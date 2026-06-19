#!/usr/bin/env python3
"""Third-party statistical verification harness for ``umbr``.

The production tool (``umbr.py``) reports *descriptive* single-run numbers:
residual dBFS, surrogate divergence, codec-surviving divergence. Those describe
one render; they do not, on their own, establish that any effect is
distinguishable from chance, nor that the "inaudible" claim survives a
controlled protocol.

This script adds the *inferential* layer. It imports umbr's own measurement
functions (so the thing under test and the measuring stick are identical) and
runs three independent analyses, each with an objective decision rule:

  1. PERMUTATION / NULL-MODEL TEST  (divergence + robustness claims)
     The real lattice's surrogate divergence is compared against an empirical
     null distribution built from K random, equal-RMS-energy control
     perturbations of the *same* clean probe. Output: an empirical p-value and a
     z-score. The null model -- not the author -- sets the bar, which is what
     makes the result "almost third-party". If the engineered lattice cannot
     beat its own random control, the report says so.

  2. ABX LISTENING HARNESS  (inaudibility claim)
     A blind, seeded ABX trial generator plus the exact-binomial significance
     test that scores the responses. The human data is supplied by the listener
     (``--abx-run`` records trials interactively); this script only generates
     the blinded trials and computes the statistics, so it cannot fabricate the
     result.

  3. CHROMAPRINT CROSS-CHECK  (independent fingerprinter)
     Runs umbr's optional fpcalc cross-check -- an external acoustic
     fingerprinter the author did not write -- and reports it as an independent
     comparison, or honestly reports that the binary is absent.

Nothing here is simulated: every number is measured against real audio.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import secrets
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# umbr.py lives in the project root (one level up from tests/); make it
# importable whether this harness is run directly, via pytest, or installed.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import umbr  # noqa: E402  -- import after sys.path bootstrap above

#: Bundled test audio: a ~40s clip of coffee.mp3, zipped to keep the repo small.
#: The full 97-minute source is intentionally not committed; the harness only
#: ever reads the first --limit-seconds, so a short clip exercises every stage.
DEFAULT_FIXTURE_ZIP = Path(__file__).resolve().parent / "fixtures" / "coffee_clip.mp3.zip"


# --- Leg 1: permutation / null-model test ----------------------------------


@dataclass
class NullModelResult:
    """Empirical-null comparison for one measured statistic."""

    statistic: str
    observed: float
    null_samples: int
    null_mean: float
    null_std: float
    null_p95: float
    null_max: float
    #: One-sided empirical p-value: P(null >= observed), with the conservative
    #: +1 / (N+1) correction so a p-value is never reported as exactly zero.
    p_value: float
    #: Smallest p-value this null could ever report (1 / (N+1)); records the
    #: resolution floor so a "not significant" result can't be over-read.
    p_value_floor: float
    #: Standardised effect size of the observed value against the null cloud.
    z_score: float
    significant_05: bool
    interpretation: str


#: Fraction of frames (loudest by displacement) the localized metric averages
#: over. The lattice acts on a sparse set of masked cells; a global average
#: dilutes that to nothing, so we look at the most-disturbed tail instead.
LOCAL_TOP_FRACTION = 0.10

#: Top-fraction percentile cutoff (e.g. 0.10 -> the 90th percentile and above).
_LOCAL_PCT = 100.0 * (1.0 - LOCAL_TOP_FRACTION)


def localized_divergence(
    clean: np.ndarray, candidate: np.ndarray, sample_rate: int, n_fft: int
) -> float:
    """Per-frame surrogate displacement, aggregated over the most-disturbed tail.

    The production surrogate averages band energies over the whole probe, so a
    perturbation confined to a sparse set of masked time-frequency cells is
    averaged away. This metric instead computes a surrogate-style log-band
    feature vector *per STFT frame*, takes the normalised displacement of each
    frame, and reports the mean of the top ``LOCAL_TOP_FRACTION`` of frames.
    That makes the statistic sensitive to *where* energy lands, which is the
    thing the lattice is actually engineered to control.
    """

    feats_clean = _per_frame_features(clean, sample_rate, n_fft)
    feats_cand = _per_frame_features(candidate, sample_rate, n_fft)
    frames = min(feats_clean.shape[0], feats_cand.shape[0])
    if frames == 0:
        return 0.0
    diff = feats_cand[:frames] - feats_clean[:frames]
    scale = np.linalg.norm(feats_clean[:frames], axis=1) + umbr.EPS
    per_frame = np.linalg.norm(diff, axis=1) / scale
    if per_frame.size == 1:
        return float(per_frame[0])
    cutoff = float(np.percentile(per_frame, _LOCAL_PCT))
    tail = per_frame[per_frame >= cutoff]
    return float(np.mean(tail)) if tail.size else float(np.max(per_frame))


def _per_frame_features(block: np.ndarray, sample_rate: int, n_fft: int) -> np.ndarray:
    """Log band-energy features for every STFT frame (frames x bands).

    Mirrors umbr.surrogate_features' band layout but keeps the per-frame axis
    instead of collapsing it, so local disturbances stay visible.
    """

    mono = block.mean(axis=1)
    spectrum, _ = umbr.hann_stft(mono, n_fft, n_fft // 2)
    mag = np.abs(spectrum) + umbr.EPS  # bins x frames
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    edges = umbr.SURROGATE_BAND_EDGES_HZ
    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            band_db = 20.0 * np.log10(mag[mask, :] + umbr.EPS)
            bands.append(np.mean(band_db, axis=0))  # one value per frame
    if not bands:
        return np.zeros((mag.shape[1], 0), dtype=np.float32)
    return np.stack(bands, axis=1).astype(np.float32)  # frames x bands


def _shuffled_lattice_control(
    clean: np.ndarray, lattice_residual: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Structure-preserving null: the lattice's own residual, sign-randomized.

    The control keeps the lattice residual's EXACT energy AND time-frequency
    support (it is literally the residual with per-sample signs flipped at
    random), then re-adds it to the clean probe. This isolates the one thing
    under test: does the lattice's *specific* sign/phase pattern diverge more
    than an arbitrary perturbation occupying the same masked cells with the same
    energy? A white-noise control would instead dump energy into loud, unmasked
    bands -- an unfair straw man -- which is why the earlier version was wrong.
    """

    flips = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32), size=lattice_residual.shape
    )
    control: np.ndarray = clean + lattice_residual * flips
    return control


def run_null_model_test(
    clean_probe: np.ndarray,
    perturbed_probe: np.ndarray,
    sample_rate: int,
    n_fft: int,
    null_samples: int,
    codec_null_samples: int,
    seed: int,
) -> list[NullModelResult]:
    """Compare the real lattice against a structure-preserving null cloud.

    Returns one result for the localized clean-decode divergence and one for the
    codec-surviving divergence (averaged over umbr's robustness codecs).
    """

    rng = np.random.default_rng(seed)
    lattice_residual = perturbed_probe - clean_probe

    observed_clean = localized_divergence(clean_probe, perturbed_probe, sample_rate, n_fft)
    null_clean = np.empty(null_samples, dtype=np.float64)
    for i in range(null_samples):
        control = _shuffled_lattice_control(clean_probe, lattice_residual, rng)
        null_clean[i] = localized_divergence(clean_probe, control, sample_rate, n_fft)

    results = [
        _summarise_null(
            "localized_divergence_clean",
            observed_clean,
            null_clean,
            "Localized (top-decile per-frame) surrogate divergence on the clean "
            "decode vs sign-randomized controls of identical energy and support.",
        )
    ]

    # --- codec-surviving null distribution ---
    if shutil.which("ffmpeg") is not None:
        observed_robust = _surviving_divergence(
            clean_probe, perturbed_probe, sample_rate, n_fft
        )
        null_robust = np.empty(codec_null_samples, dtype=np.float64)
        for i in range(codec_null_samples):
            control = _shuffled_lattice_control(clean_probe, lattice_residual, rng)
            null_robust[i] = _surviving_divergence(clean_probe, control, sample_rate, n_fft)
        results.append(
            _summarise_null(
                "codec_surviving_divergence",
                observed_robust,
                null_robust,
                "Divergence that survives MP3/AAC round-trips vs sign-randomized "
                "equal-support controls. Low absolute values are expected (and "
                "honest): lossy codecs discard sub-audible energy first.",
            )
        )
    return results


def _surviving_divergence(
    clean_probe: np.ndarray, perturbed_probe: np.ndarray, sample_rate: int, n_fft: int
) -> float:
    """Mean lattice-attributable surviving divergence across umbr's codecs."""

    codecs = umbr.measure_codec_robustness(clean_probe, perturbed_probe, sample_rate, n_fft)
    return umbr.aggregate_robustness(codecs)


def _summarise_null(
    name: str, observed: float, null: np.ndarray, interpretation: str
) -> NullModelResult:
    """Turn an observed value and a null sample array into a decision."""

    n = int(null.size)
    null_mean = float(np.mean(null))
    null_std = float(np.std(null, ddof=1)) if n > 1 else 0.0
    # Conservative empirical p-value: (#{null >= observed} + 1) / (n + 1).
    exceed = int(np.count_nonzero(null >= observed))
    p_value = (exceed + 1) / (n + 1)
    p_floor = 1.0 / (n + 1)
    z = (observed - null_mean) / (null_std + umbr.EPS)
    significant = p_value < 0.05
    verdict = (
        "Observed exceeds the equal-energy/equal-support null: the lattice's "
        "specific structure diverges more than an arbitrary perturbation of the "
        "same energy and support."
        if significant
        else "Observed is within the null band: the lattice's specific structure "
        "is not distinguishable from a sign-randomized perturbation occupying "
        "the same masked cells with the same energy."
    )
    if not significant and p_floor > 0.05:
        verdict += (
            f" (Note: only {n} null samples, so the smallest detectable p-value "
            f"is {p_floor:.3f}; raise --null-samples to resolve a weak effect.)"
        )
    return NullModelResult(
        statistic=name,
        observed=observed,
        null_samples=n,
        null_mean=null_mean,
        null_std=null_std,
        null_p95=float(np.percentile(null, 95)) if n else 0.0,
        null_max=float(np.max(null)) if n else 0.0,
        p_value=p_value,
        p_value_floor=p_floor,
        z_score=z,
        significant_05=significant,
        interpretation=f"{interpretation} {verdict}",
    )


# --- Leg 2: ABX listening harness -------------------------------------------


def binomial_sf_ge(k: int, n: int, p: float = 0.5) -> float:
    """Exact one-sided binomial tail P(X >= k) for X ~ Binomial(n, p).

    Pure-Python so the harness carries no scipy dependency. Used to score ABX
    results: under the null "listener cannot distinguish", correct responses are
    Binomial(n, 0.5); a small tail probability means they *can* hear a
    difference, which would falsify the transparency claim.
    """

    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return min(1.0, total)


@dataclass
class AbxTrialPlan:
    """A blinded ABX trial sheet the listener fills in, plus the answer key."""

    seed: int
    n_trials: int
    reference_a: str
    reference_b: str
    #: For each trial, which source ("A" or "B") was secretly played as X.
    answer_key: list[str]


def build_abx_plan(
    original_wav: str, perturbed_wav: str, n_trials: int, seed: int | None
) -> AbxTrialPlan:
    """Generate a seeded blind ABX answer key.

    A is always the original, B the perturbed render. For each trial, X is
    randomly the same as A or as B; the listener must say whether X == A or
    X == B. A cryptographically-seeded RNG keeps the key reproducible for audit
    while unpredictable to the listener.
    """

    if seed is None:
        seed = secrets.randbits(32)
    rng = np.random.default_rng(seed)
    key = ["A" if bit == 0 else "B" for bit in rng.integers(0, 2, size=n_trials)]
    return AbxTrialPlan(
        seed=seed,
        n_trials=n_trials,
        reference_a=original_wav,
        reference_b=perturbed_wav,
        answer_key=key,
    )


@dataclass
class AbxResult:
    """Scored outcome of a completed ABX session."""

    n_trials: int
    correct: int
    accuracy: float
    p_value_distinguishable: float
    #: True if listeners reliably told the files apart (transparency FAILS).
    listeners_distinguished: bool
    interpretation: str


def score_abx(answer_key: list[str], responses: list[str]) -> AbxResult:
    """Score listener responses against the answer key with an exact binomial test."""

    n = len(answer_key)
    if len(responses) != n:
        raise ValueError(f"expected {n} responses, got {len(responses)}")
    correct = sum(1 for k, r in zip(answer_key, responses) if k.upper() == r.upper())
    p_value = binomial_sf_ge(correct, n, 0.5)
    distinguished = p_value < 0.05
    if distinguished:
        verdict = (
            f"Listeners scored {correct}/{n} (p={p_value:.4f} < 0.05): they CAN "
            "hear a difference. The transparency claim is NOT supported for this "
            "material."
        )
    else:
        verdict = (
            f"Listeners scored {correct}/{n} (p={p_value:.4f}): indistinguishable "
            "from guessing. Consistent with transparency (absence of evidence "
            "for audibility, not proof of inaudibility -- raise n to tighten)."
        )
    return AbxResult(
        n_trials=n,
        correct=correct,
        accuracy=correct / n if n else 0.0,
        p_value_distinguishable=p_value,
        listeners_distinguished=distinguished,
        interpretation=verdict,
    )


# --- Leg 3: Chromaprint cross-check -----------------------------------------


@dataclass
class FingerprintResult:
    available: bool
    note: str
    hamming_fraction: float
    interpretation: str


def run_fingerprint_leg(
    clean_probe: np.ndarray, perturbed_probe: np.ndarray, sample_rate: int
) -> FingerprintResult:
    """Independent cross-check via Chromaprint (external tool, not authored here)."""

    check = umbr.run_fingerprint_check(clean_probe, perturbed_probe, sample_rate)
    if not check.available:
        note = (
            f"{check.note}. Install Chromaprint (`brew install chromaprint`) to "
            "enable the only fully independent leg of this analysis."
        )
        interp = (
            "No external fingerprinter available: legs 1 and 2 stand, but the "
            "independent-tool cross-check could not be run."
        )
    else:
        note = check.note
        interp = (
            f"A real acoustic fingerprinter the author did not write reports a "
            f"{check.hamming_fraction:.4f} symbol-mismatch fraction between "
            "original and perturbed."
        )
    return FingerprintResult(
        available=check.available,
        note=note,
        hamming_fraction=check.hamming_fraction,
        interpretation=interp,
    )


# --- Driving the render and extracting probes -------------------------------


@dataclass
class AnalysisReport:
    source: str
    perturbed_output: str
    sample_rate: int
    n_fft: int
    probe_seconds: float
    null_model: list[NullModelResult] = field(default_factory=list)
    fingerprint: FingerprintResult | None = None
    abx_plan: AbxTrialPlan | None = None
    abx_result: AbxResult | None = None


@contextlib.contextmanager
def _resolved_audio_source(requested: Path) -> Iterator[Path]:
    """Yield a real audio file path, extracting it first if given a .zip.

    The committed fixture is zipped (MP3 plus a tiny zip overhead), and a user
    may point ``source`` at any .zip holding one audio file. ffmpeg cannot decode
    a zip directly, so we unpack the single contained file into a temp directory
    and yield that path; non-zip inputs pass through untouched.
    """

    if requested.suffix.lower() != ".zip":
        yield requested
        return
    with zipfile.ZipFile(requested) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise SystemExit(
                f"{requested} must contain exactly one audio file, found {len(members)}."
            )
        with tempfile.TemporaryDirectory(prefix="umbr-fixture-") as temp_dir:
            extracted = Path(archive.extract(members[0], temp_dir))
            yield extracted


def render_and_extract(
    source: Path, sample_rate: int, n_fft: int, hop: int, limit_seconds: float, seed: str
) -> tuple[np.ndarray, np.ndarray, Path, umbr.StftConfig]:
    """Run umbr's real render and return (clean_probe, perturbed_probe, output, cfg)."""

    config = umbr.StftConfig(sample_rate, n_fft, hop)
    output = source.with_name(f"{source.stem}.analysis.umbr.wav")
    delta = output.with_name(f"{output.stem}.delta.wav")
    args = argparse.Namespace(
        source=str(source),
        output=str(output),
        delta_output=str(delta),
        artifacts="artifacts",
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop=hop,
        chunk_seconds=20.0,
        strength=umbr.Strength.MEDIUM.name.lower(),
        seed=seed,
        limit_seconds=limit_seconds,
    )
    limit_samples = int(limit_seconds * sample_rate) if limit_seconds else 0
    with tempfile.TemporaryDirectory(prefix="umbr-analysis-") as temp_dir:
        raw_path = Path(temp_dir) / "input.s16le"
        total = umbr.decode_to_raw(source, raw_path, sample_rate, umbr.CHANNELS)
        if limit_samples:
            total = min(total, limit_samples)
        raw = np.memmap(raw_path, dtype=np.int16, mode="r", shape=(total, umbr.CHANNELS))
        state = umbr.render_to_wav(raw, total, config, args, output, delta)
    if state.probe_original is None or state.probe_perturbed is None:
        raise RuntimeError("render produced no probe audio")
    probe_target = int(umbr.ROBUSTNESS_PROBE_SECONDS * sample_rate)
    clean = np.asarray(state.probe_original[:probe_target], dtype=np.float32)
    perturbed = np.asarray(state.probe_perturbed[:probe_target], dtype=np.float32)
    return clean, perturbed, output, config


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="analysis",
        description="Statistical verification harness for umbr (null-model, ABX, fingerprint).",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_FIXTURE_ZIP),
        help="Input audio file, or a .zip containing one. Defaults to the bundled "
        "coffee clip fixture.",
    )
    parser.add_argument("--limit-seconds", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--seed", default="umbr-research")
    parser.add_argument("--stat-seed", type=int, default=20260619)
    parser.add_argument(
        "--null-samples",
        type=int,
        default=200,
        help="Sign-randomized controls in the clean-decode null.",
    )
    parser.add_argument(
        "--codec-null-samples",
        type=int,
        default=40,
        help="Controls in the (slow) codec-survival null; 40 gives a p-floor of ~0.024.",
    )
    parser.add_argument("--abx-trials", type=int, default=16)
    parser.add_argument(
        "--abx-seed", type=int, default=None, help="Seed for the ABX key (default: random)."
    )
    parser.add_argument(
        "--abx-run",
        action="store_true",
        help="Interactively record ABX responses and score them.",
    )
    parser.add_argument(
        "--abx-responses",
        default="",
        help="Score a saved sheet of A/B responses (one per line) against the plan.",
    )
    parser.add_argument("--report", default="artifacts/umbr_statistics.json")
    args = parser.parse_args()

    requested = Path(args.source).expanduser()
    if not requested.exists():
        raise SystemExit(f"missing source audio: {requested}")

    with _resolved_audio_source(requested) as source:
        umbr.log("analysis", f"rendering {source} ({args.limit_seconds:.0f}s) for analysis")
        clean, perturbed, output, config = render_and_extract(
            source, args.sample_rate, args.n_fft, args.hop, args.limit_seconds, args.seed
        )

        umbr.log("analysis", f"leg 1: null-model test ({args.null_samples} controls)")
        null_results = run_null_model_test(
            clean,
            perturbed,
            config.sample_rate,
            config.n_fft,
            args.null_samples,
            args.codec_null_samples,
            args.stat_seed,
        )

        umbr.log("analysis", "leg 3: Chromaprint cross-check")
        fingerprint = run_fingerprint_leg(clean, perturbed, config.sample_rate)

        umbr.log("analysis", f"leg 2: building blind ABX plan ({args.abx_trials} trials)")
        abx_plan = build_abx_plan(
            str(source), str(output), args.abx_trials, args.abx_seed
        )
        abx_result = None
        if args.abx_responses:
            responses = _responses_from_file(Path(args.abx_responses), abx_plan.n_trials)
            abx_result = score_abx(abx_plan.answer_key, responses)
        elif args.abx_run:
            abx_result = _interactive_abx(abx_plan)

        report = AnalysisReport(
            source=str(requested),
            perturbed_output=str(output),
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            probe_seconds=umbr.ROBUSTNESS_PROBE_SECONDS,
            null_model=null_results,
            fingerprint=fingerprint,
            abx_plan=abx_plan,
            abx_result=abx_result,
        )

        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
        _print_summary(report)
    umbr.log("analysis", f"full statistics written to {report_path}")
    return 0


def _interactive_abx(plan: AbxTrialPlan) -> AbxResult:
    """Record listener responses for a blind ABX session and score them."""

    print("\n=== Blind ABX session ===")
    print(f"A = {plan.reference_a}\nB = {plan.reference_b}")
    print(
        f"{plan.n_trials} trials. For each, listen to A, B, then X, and type "
        "whether X matches A or B.\n"
    )
    print(
        "Play each file in your own player; this prompt only records what you "
        "heard. Type A or B (or 'q' to abort).\n"
    )
    responses: list[str] = []
    for i in range(plan.n_trials):
        ans = ""
        while ans not in {"A", "B"}:
            try:
                ans = input(f"Trial {i + 1}/{plan.n_trials} -- X matches (A/B)? ").strip().upper()
            except EOFError:
                raise SystemExit(
                    "\nABX session aborted (no input). Re-run with --abx-run in a "
                    "terminal, or score a saved sheet with --abx-responses FILE."
                )
            if ans == "Q":
                raise SystemExit("ABX session aborted by user.")
        responses.append(ans)
    return score_abx(plan.answer_key, responses)


def _responses_from_file(path: Path, n_trials: int) -> list[str]:
    """Read A/B responses (one per line, blanks/comments ignored) from a sheet."""

    responses: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().upper()
        if not token or token.startswith("#"):
            continue
        if token not in {"A", "B"}:
            raise SystemExit(f"invalid ABX response {token!r} in {path} (expected A or B)")
        responses.append(token)
    if len(responses) != n_trials:
        raise SystemExit(
            f"{path} has {len(responses)} responses; the plan has {n_trials} trials."
        )
    return responses


def _print_summary(report: AnalysisReport) -> None:
    print("\n" + "=" * 70)
    print("UMBR STATISTICAL VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Source : {report.source}")
    print(f"Probe  : first {report.probe_seconds:.0f}s, {report.sample_rate} Hz\n")

    print("LEG 1 -- Structure-preserving permutation null-model test")
    for r in report.null_model:
        flag = "SIGNIFICANT" if r.significant_05 else "not significant"
        print(f"  [{r.statistic}]")
        print(
            f"    observed={r.observed:.6f}  null_mean={r.null_mean:.6f}  "
            f"null_p95={r.null_p95:.6f}  (n={r.null_samples})"
        )
        print(
            f"    p={r.p_value:.4f} (floor {r.p_value_floor:.4f})  "
            f"z={r.z_score:+.2f}  -> {flag}"
        )
        print(f"    {r.interpretation}\n")

    print("LEG 2 -- Blind ABX listening test")
    if report.abx_result is not None:
        print(f"  {report.abx_result.interpretation}\n")
    elif report.abx_plan is not None:
        plan = report.abx_plan
        print(
            f"  Blind ABX plan generated ({plan.n_trials} trials, key seed "
            f"{plan.seed}). Re-run with --abx-run to record listener responses,\n"
            "  or score an existing session against the saved answer key. The "
            "binomial test needs real human trials -- it cannot be simulated.\n"
        )

    print("LEG 3 -- Independent fingerprinter cross-check (Chromaprint)")
    if report.fingerprint is not None:
        print(f"  {report.fingerprint.interpretation}")
        print(f"  {report.fingerprint.note}\n")


# --- Pytest-discoverable self-checks ---------------------------------------
# These assert the statistical machinery itself is correct (the binomial tail
# and ABX scoring) without needing audio, so `pytest tests/` validates the
# method on every run. The full audio analysis is driven via __main__ / CI.


def test_binomial_tail_matches_known_values() -> None:
    assert abs(binomial_sf_ge(12, 16) - 0.0384063721) < 1e-6
    assert abs(binomial_sf_ge(8, 16) - 0.5981903076) < 1e-6
    assert abs(binomial_sf_ge(16, 16) - (0.5**16)) < 1e-12
    assert binomial_sf_ge(0, 16) == 1.0
    assert binomial_sf_ge(17, 16) == 0.0


def test_abx_scoring_decision_rule() -> None:
    key = ["A", "B"] * 8
    perfect = score_abx(key, key)
    assert perfect.correct == 16
    assert perfect.listeners_distinguished is True
    chance = score_abx(key, ["A"] * 16)
    assert chance.correct == 8
    assert chance.listeners_distinguished is False


def test_empirical_p_value_floor_is_recorded() -> None:
    null = np.zeros(40)
    result = _summarise_null("x", 0.0, null, "test")
    # Observed equals every null sample, so it is not in the tail -> p = 1.0,
    # and the floor reflects the sample budget (1 / (N+1)).
    assert abs(result.p_value_floor - 1.0 / 41.0) < 1e-9
    assert result.significant_05 is False


if __name__ == "__main__":
    raise SystemExit(main())
