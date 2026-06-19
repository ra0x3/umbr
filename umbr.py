#!/usr/bin/env python3
"""Umbr audio perturbation research console.

The implementation operates in the short-time Fourier domain. For each chunk it
estimates a conservative psychoacoustic masking budget, lays a deterministic
non-semantic perturbation under that budget, and reconstructs clean audio.

Unlike the earlier prototype, every headline number in the audit is *measured*
rather than asserted:

* Representation divergence is measured with an explicitly labelled internal
  surrogate proxy, and (when the ``fpcalc`` Chromaprint binary is available)
  cross-checked against a real acoustic fingerprinter.
* Processing robustness is measured by round-tripping the rendered audio through
  real lossy codecs (MP3 and AAC) with ffmpeg, decoding back, and recomputing
  surrogate coherence on the degraded signal.
* A small closed refinement loop adjusts per-chunk perturbation strength so the
  transparency gate is satisfied first, then divergence is taken second.

It does not encode messages or payloads. The lattice is a deterministic,
non-semantic sign pattern with no recoverable content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# --- Numerical and audio constants -----------------------------------------

#: Floor added before logs/divisions to avoid ``log(0)`` and divide-by-zero.
EPS = 1.0e-12

#: Full-scale value of signed 16-bit PCM used when decoding to float.
INT16_FULL_SCALE = 32768.0

#: Largest 16-bit sample magnitude written back out (leaves one LSB of margin).
INT16_PEAK = 32767.0

#: dBFS reported for a digitally silent signal (linear amplitude of zero).
SILENCE_DBFS = -160.0

#: Decibel gain applied to the residual before writing the inspection WAV, so a
#: sub-audible delta becomes visible on a spectrogram (60 dB == 1000x).
DELTA_INSPECTION_GAIN_DB = 60.0

#: Suffix appended to the source stem when ``--output`` is not given, e.g.
#: ``song.mp3`` -> ``song.umbr.wav``.
DEFAULT_OUTPUT_SUFFIX = ".umbr.wav"

#: Suffix for the amplified inspection WAV when ``--delta-output`` is not given.
DEFAULT_DELTA_SUFFIX = ".delta_x60dB.wav"

#: Default directory for the CSV/JSON/Markdown audit artifacts.
DEFAULT_ARTIFACTS_DIR = "artifacts"

#: Output channel count. The pipeline always renders interleaved stereo.
CHANNELS = 2

# --- Psychoacoustic masking constants --------------------------------------

#: Lower edge (Hz) of the band eligible for perturbation. Below this the ear is
#: insensitive and codecs spend few bits, so hidden energy is fragile and risky.
PERTURB_BAND_LOW_HZ = 120.0

#: Upper edge (Hz) of the eligible band. Above this lies fragile cymbal air that
#: lossy codecs aggressively discard.
PERTURB_BAND_HIGH_HZ = 15_500.0

#: A bin is treated as masked only if it sits within this many dB below the
#: per-frame spectral peak (a coarse simultaneous-masking proxy).
NEAR_MASKER_DB = -38.0

#: Density (0..1) a frame must reach before it is considered "dense" enough to
#: hide perturbation energy. Sparse/exposed frames are passed through untouched.
DENSE_FRAME_THRESHOLD = 0.43

#: Absolute-threshold-of-hearing penalty multipliers applied per region. These
#: down-weight bands where the ear is more sensitive or codecs are more lossy.
ATH_PENALTY_SUB_250 = 0.40
ATH_PENALTY_250_1000 = 0.70
ATH_PENALTY_ABOVE_12K = 0.55

#: Density mixing weights for loudness, spectral flux, and spectral flatness.
DENSITY_WEIGHT_LOUDNESS = 0.50
DENSITY_WEIGHT_FLUX = 0.30
DENSITY_WEIGHT_FLATNESS = 0.20

#: Percentiles used to normalise per-frame log energy into a 0..1 loudness.
LOUDNESS_LOW_PCT = 35.0
LOUDNESS_HIGH_PCT = 92.0

#: Quarter-turn phase offset applied to the perturbation relative to the host,
#: spreading energy toward the masked quadrature component.
PERTURB_PHASE_OFFSET = math.pi / 2.0

#: Cover-classification thresholds for the human-readable audit only.
COVER_SPARSE_DENSITY = 0.42
COVER_TRANSIENT_FLUX = 0.74
COVER_TRANSIENT_LOUDNESS = 0.42
COVER_TEXTURE_FLATNESS = 0.62
COVER_SWELL_LOUDNESS = 0.78


class CoverType(Enum):
    """Dominant psychoacoustic cover named for a modified frame in the audit.

    The value is the human-readable phrase written verbatim into the report, so
    the audit text and the classifier never drift apart.
    """

    SPARSE = "protected sparse passage"
    TRANSIENT = "drum or bass transient cover"
    TEXTURE = "cymbal wash or distorted texture"
    SWELL = "bass, chord cluster, or vocal swell"
    DENSE = "dense mixed phrase"


# --- Strength and refinement constants -------------------------------------


class Strength(Enum):
    """Perturbation strength preset selectable on the command line.

    The value is the dB level of the base perturbation relative to the local
    host magnitude (more negative is quieter and more conservative).
    """

    CONSERVATIVE = -62.0
    MEDIUM = -56.0
    RESEARCH = -50.0

    @classmethod
    def from_name(cls, name: str) -> "Strength":
        """Resolve a lowercase CLI choice (e.g. ``"medium"``) to a member."""

        return cls[name.upper()]

#: Transparency gate: per-chunk residual RMS must stay at or below this dBFS
#: ceiling. The refinement loop attenuates strength until the gate is met.
TRANSPARENCY_GATE_DBFS = -85.0

#: Maximum refinement attempts per chunk before accepting the quietest result.
MAX_REFINE_STEPS = 4

#: Multiplicative attenuation applied to strength on each failed refine step.
REFINE_ATTENUATION = 0.5

# --- Surrogate model constants ---------------------------------------------

#: Band edges (Hz) for the surrogate log-band embedding. The surrogate is an
#: explicit *proxy* for an audio-analysis system, not the real target.
SURROGATE_BAND_EDGES_HZ = np.array(
    [80, 160, 320, 640, 1280, 2560, 5120, 10_240, 15_500], dtype=np.float32
)

#: Sensitivity of the coherence kernel ``exp(-k * relative_displacement)``.
#: Larger ``k`` means small feature shifts read as large divergence.
SURROGATE_COHERENCE_K = 18.0

# --- Robustness (codec) constants ------------------------------------------

#: Codecs exercised by the robustness stage: (label, ffmpeg encoder, bitrate).
ROBUSTNESS_CODECS = (
    ("mp3_128k", "libmp3lame", "128k"),
    ("aac_128k", "aac", "128k"),
)

#: Seconds of the rendered output sampled for the robustness round-trip. Codec
#: round-tripping the whole track would be slow and is unnecessary for a proxy.
ROBUSTNESS_PROBE_SECONDS = 30.0


@dataclass(frozen=True)
class StftConfig:
    """Immutable bundle of the STFT parameters threaded through the pipeline."""

    sample_rate: int
    n_fft: int
    hop: int


@dataclass(frozen=True)
class LatticeContext:
    """Per-channel context controlling lattice strength and determinism."""

    strength_db: float
    seed: int
    chunk_index: int


@dataclass
class RegionAudit:
    """Summary of one modified STFT analysis frame, used in the audit report."""

    index: int
    start: float
    end: float
    low_hz: float
    high_hz: float
    density: float
    budget_db: float
    perturbation_dbfs: float
    safe_bins: int
    total_bins: int
    cover: str
    reason: str


@dataclass
class CodecRobustness:
    """Surrogate coherence of the rendered output after one codec round-trip."""

    label: str
    encoder: str
    bitrate: str
    #: Coherence of (original probe) vs (original probe through the codec).
    baseline_coherence: float
    #: Coherence of (perturbed probe through codec) vs (original probe).
    perturbed_coherence: float
    #: How much of the codec-survived divergence is attributable to the lattice
    #: rather than to the codec itself (perturbed minus baseline displacement).
    surviving_divergence: float


@dataclass
class FingerprintCheck:
    """Optional Chromaprint cross-check of the internal surrogate."""

    available: bool
    note: str
    #: Fraction of fingerprint symbols that differ, original vs perturbed.
    hamming_fraction: float = 0.0


@dataclass
class Metrics:
    """Whole-render quality, divergence, and robustness metrics (all measured)."""

    source: str
    output: str
    delta_inspection_output: str
    sample_rate: int
    channels: int
    duration_seconds: float
    processed_seconds: float
    n_fft: int
    hop_size: int
    modified_frames: int
    modified_time_ratio: float
    delta_rms_dbfs: float
    delta_peak_dbfs: float
    output_peak_dbfs: float
    log_spectral_distance_db: float
    transparency_gate_dbfs: float
    transparency_gate_pass: bool
    surrogate_self_coherence: float
    surrogate_post_coherence: float
    surrogate_divergence: float
    fingerprint: FingerprintCheck
    codec_robustness: list[CodecRobustness] = field(default_factory=list)
    robustness_score: float = 0.0
    clipping_samples: int = 0


def dbfs(value: float) -> float:
    """Convert a linear full-scale amplitude to dBFS."""

    if value <= 0.0:
        return SILENCE_DBFS
    return 20.0 * math.log10(value)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a scalar value into a closed interval."""

    return max(low, min(high, value))


def log(stage: str, message: str) -> None:
    """Emit a compact progress line."""

    print(f"[{stage:<12}] {message}", flush=True)


def run_checked(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Run a subprocess and raise a readable error on failure."""

    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"command failed: {' '.join(command)}")
    return result


def ffprobe_duration(path: Path) -> float:
    """Return media duration in seconds using ffprobe."""

    result = run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ]
    )
    return float(result.stdout.decode("utf-8").strip())


def decode_to_raw(source: Path, raw_path: Path, sample_rate: int, channels: int) -> int:
    """Decode arbitrary ffmpeg-readable audio to signed 16-bit interleaved PCM.

    Returns the number of stereo sample frames written.
    """

    run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            str(raw_path),
        ]
    )
    bytes_per_frame = np.dtype(np.int16).itemsize * channels
    return raw_path.stat().st_size // bytes_per_frame


def window_coherent_gain(n_fft: int) -> float:
    """Return the coherent gain (mean) of the Hann analysis window.

    Used to convert summed STFT-bin magnitude back to a time-domain amplitude
    so that reported per-region levels are physically meaningful.
    """

    return float(np.mean(np.hanning(n_fft).astype(np.float32))) + EPS


def hann_stft(channel: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, int]:
    """Compute an analysis STFT with Hann windows.

    Returns the complex spectrum (bins x frames) and the original sample count.
    """

    if channel.size == 0:
        return np.empty((n_fft // 2 + 1, 0), dtype=np.complex64), 0
    pad_left = n_fft // 2
    centered_len = channel.size + 2 * pad_left
    frame_count = max(1, int(math.ceil(max(0, centered_len - n_fft) / hop)) + 1)
    padded_len = (frame_count - 1) * hop + n_fft
    padded = np.pad(channel, (pad_left, padded_len - channel.size - pad_left))
    windows = sliding_window_view(padded, n_fft)[::hop]
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(windows * window, axis=1).T.astype(np.complex64)
    return spectrum, channel.size


def hann_istft(spectrum: np.ndarray, original_len: int, n_fft: int, hop: int) -> np.ndarray:
    """Invert a Hann-windowed STFT with overlap-add normalization."""

    frame_count = spectrum.shape[1]
    out_len = (frame_count - 1) * hop + n_fft
    output = np.zeros(out_len, dtype=np.float32)
    weight = np.zeros(out_len, dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.fft.irfft(spectrum.T, n=n_fft, axis=1).astype(np.float32)
    for frame_index in range(frame_count):
        start = frame_index * hop
        end = start + n_fft
        output[start:end] += frames[frame_index] * window
        weight[start:end] += window * window
    valid = weight > 1.0e-8
    output[valid] /= weight[valid]
    pad_left = n_fft // 2
    return output[pad_left : pad_left + original_len]


def smooth_1d(values: np.ndarray, width: int) -> np.ndarray:
    """Box-smooth one-dimensional values without changing length."""

    if width <= 1 or values.size == 0:
        return values
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values, kernel, mode="same")


def classify_cover(density: float, flux: float, flatness: float, loudness: float) -> CoverType:
    """Name the dominant psychoacoustic cover for the audit report only."""

    if density < COVER_SPARSE_DENSITY:
        return CoverType.SPARSE
    if flux > COVER_TRANSIENT_FLUX and loudness > COVER_TRANSIENT_LOUDNESS:
        return CoverType.TRANSIENT
    if flatness > COVER_TEXTURE_FLATNESS:
        return CoverType.TEXTURE
    if loudness > COVER_SWELL_LOUDNESS:
        return CoverType.SWELL
    return CoverType.DENSE


def lattice_signs(shape: tuple[int, int], seed: int, chunk_index: int) -> np.ndarray:
    """Create a deterministic +/- lattice without embedding semantic content."""

    rng = np.random.default_rng(seed + chunk_index * 104_729)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=shape)
    # Lightly braid diagonal bands so the pattern has coarse spatial structure.
    rows, cols = np.indices(shape)
    braid = np.where(((rows * 3 + cols * 5 + seed) % 11) < 5, 1.0, -1.0).astype(np.float32)
    return signs * braid


def estimate_density(
    mag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate per-frame loudness, flux, flatness, and a combined density.

    All four arrays are normalised to 0..1 and have one value per STFT frame.
    """

    frame_energy = np.sqrt(np.mean(mag * mag, axis=0) + EPS)
    log_energy = 20.0 * np.log10(frame_energy + EPS)
    low = float(np.percentile(log_energy, LOUDNESS_LOW_PCT))
    high = float(np.percentile(log_energy, LOUDNESS_HIGH_PCT))
    loudness = np.clip((log_energy - low) / max(1.0, high - low), 0.0, 1.0)

    flux_raw = np.zeros_like(frame_energy)
    if mag.shape[1] > 1:
        flux_raw[1:] = np.maximum(mag[:, 1:] - mag[:, :-1], 0.0).mean(axis=0)
    flux = np.clip(flux_raw / (np.percentile(flux_raw, 92) + EPS), 0.0, 1.0)

    flatness = np.exp(np.mean(np.log(mag), axis=0)) / (np.mean(mag, axis=0) + EPS)
    flatness = np.clip(flatness / (np.percentile(flatness, 90) + EPS), 0.0, 1.0)

    density = np.clip(
        DENSITY_WEIGHT_LOUDNESS * loudness
        + DENSITY_WEIGHT_FLUX * flux
        + DENSITY_WEIGHT_FLATNESS * flatness,
        0.0,
        1.0,
    )
    density = smooth_1d(density.astype(np.float32), 5)
    return density, loudness, flux, flatness


def build_perturbation(
    spectrum: np.ndarray,
    config: StftConfig,
    context: LatticeContext,
    density_bundle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a bounded complex perturbation and its per-bin amplitude ratio.

    ``density_bundle`` is reused across refinement steps so density estimation
    is not recomputed on every attempt.
    """

    freqs = np.fft.rfftfreq(config.n_fft, 1.0 / config.sample_rate).astype(np.float32)
    mag = np.abs(spectrum).astype(np.float32) + EPS
    phase = np.angle(spectrum).astype(np.float32)
    density, _loudness, _flux, _flatness = density_bundle

    band = (freqs[:, None] >= PERTURB_BAND_LOW_HZ) & (freqs[:, None] <= PERTURB_BAND_HIGH_HZ)
    frame_peak = np.max(mag, axis=0, keepdims=True)
    near_masker = mag >= frame_peak * (10.0 ** (NEAR_MASKER_DB / 20.0))
    dense_frame = density[None, :] >= DENSE_FRAME_THRESHOLD

    ath_penalty = np.ones_like(freqs, dtype=np.float32)
    ath_penalty[freqs < 250.0] = ATH_PENALTY_SUB_250
    ath_penalty[(freqs >= 250.0) & (freqs < 1000.0)] = ATH_PENALTY_250_1000
    ath_penalty[freqs > 12_000.0] = ATH_PENALTY_ABOVE_12K
    tonal_guard = np.clip((mag / (frame_peak + EPS)) ** 0.35, 0.18, 1.0)
    safe = band & near_masker & dense_frame

    base_ratio = 10.0 ** (context.strength_db / 20.0)
    ratio = base_ratio * density[None, :] * ath_penalty[:, None] * tonal_guard
    ratio = np.where(safe, ratio, 0.0).astype(np.float32)

    signs = lattice_signs(ratio.shape, context.seed, context.chunk_index)
    perturbation = (
        mag * ratio * signs * np.exp(1j * (phase + PERTURB_PHASE_OFFSET))
    ).astype(np.complex64)
    return perturbation, ratio


def collect_region_audits(
    ratio: np.ndarray,
    perturbation: np.ndarray,
    config: StftConfig,
    strength_db: float,
    density_bundle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[RegionAudit], dict[str, float]]:
    """Build per-frame audits and summary stats from a finalized perturbation."""

    density, loudness, flux, flatness = density_bundle
    coherent_gain = window_coherent_gain(config.n_fft)
    freqs = np.fft.rfftfreq(config.n_fft, 1.0 / config.sample_rate)
    audits: list[RegionAudit] = []
    frame_hop_seconds = config.n_fft / config.sample_rate
    for frame_index in range(ratio.shape[1]):
        safe_bins = int(np.count_nonzero(ratio[:, frame_index]))
        if safe_bins == 0:
            continue
        bin_indices = np.flatnonzero(ratio[:, frame_index])
        # Convert summed bin magnitude to a time-domain amplitude using the
        # window coherent gain, then express as an RMS-equivalent dBFS level.
        frame_amp = (
            float(np.sum(np.abs(perturbation[:, frame_index]))) / config.n_fft / coherent_gain
        )
        perturb_rms = frame_amp / math.sqrt(2.0)
        cover = classify_cover(
            float(density[frame_index]),
            float(flux[frame_index]),
            float(flatness[frame_index]),
            float(loudness[frame_index]),
        )
        audits.append(
            RegionAudit(
                index=frame_index,
                start=0.0,
                end=frame_hop_seconds,
                low_hz=float(freqs[bin_indices[0]]),
                high_hz=float(freqs[bin_indices[-1]]),
                density=float(density[frame_index]),
                budget_db=float(
                    strength_db + 20.0 * math.log10(max(0.05, density[frame_index]))
                ),
                perturbation_dbfs=dbfs(perturb_rms),
                safe_bins=safe_bins,
                total_bins=ratio.shape[0],
                cover=cover.value,
                reason=(
                    "masked by local spectral density, transient/noise energy, "
                    "and frequency-band headroom"
                ),
            )
        )

    stats = {
        "safe_bin_ratio": float(np.count_nonzero(ratio) / ratio.size),
        "mean_density": float(np.mean(density)),
        "mean_ratio": float(np.mean(ratio[ratio > 0.0])) if np.any(ratio > 0.0) else 0.0,
    }
    return audits, stats


def refine_channel(
    spectrum: np.ndarray,
    config: StftConfig,
    context: LatticeContext,
    original_len: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Closed refinement loop: attenuate strength until the gate is satisfied.

    Implements the spec priority of transparency first, divergence second. The
    loop reduces strength while the per-channel residual RMS exceeds the gate,
    then returns the transformed channel, the final ratio map, and the strength
    actually used (in dB).
    """

    density_bundle = estimate_density(np.abs(spectrum).astype(np.float32) + EPS)
    original = hann_istft(spectrum, original_len, config.n_fft, config.hop)
    current_db = context.strength_db
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(MAX_REFINE_STEPS):
        step_context = LatticeContext(current_db, context.seed, context.chunk_index)
        perturbation, ratio = build_perturbation(
            spectrum, config, step_context, density_bundle
        )
        transformed = hann_istft(
            spectrum + perturbation, original_len, config.n_fft, config.hop
        )
        residual = transformed - original
        residual_rms = float(np.sqrt(np.mean(residual * residual) + EPS))
        best = (transformed, ratio, current_db)
        if dbfs(residual_rms) <= TRANSPARENCY_GATE_DBFS:
            break
        current_db += 20.0 * math.log10(REFINE_ATTENUATION)
    assert best is not None
    return best


def log_spectral_distance(original: np.ndarray, transformed: np.ndarray, n_fft: int) -> float:
    """Compute a compact log-spectral distance on a representative chunk."""

    mono_a = original.mean(axis=1)
    mono_b = transformed.mean(axis=1)
    spec_a, _ = hann_stft(mono_a, n_fft, n_fft // 2)
    spec_b, _ = hann_stft(mono_b, n_fft, n_fft // 2)
    log_a = 20.0 * np.log10(np.abs(spec_a) + EPS)
    log_b = 20.0 * np.log10(np.abs(spec_b) + EPS)
    return float(np.sqrt(np.mean((log_a - log_b) ** 2)))


def surrogate_features(block: np.ndarray, sample_rate: int, n_fft: int) -> np.ndarray:
    """Extract a lightweight surrogate embedding from log spectral bands.

    NOTE: this is an explicit *proxy* for an audio-analysis system, not a model
    of any specific real retrieval/fingerprinting system. Divergence measured
    against it is suggestive, not conclusive; the Chromaprint cross-check and
    codec round-trip exist precisely because this surrogate is self-defined.
    """

    mono = block.mean(axis=1)
    spectrum, _ = hann_stft(mono, n_fft, n_fft // 2)
    mag = np.abs(spectrum) + EPS
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    edges = SURROGATE_BAND_EDGES_HZ
    features: list[float] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            band = 20.0 * np.log10(mag[mask] + EPS)
            features.extend([float(np.mean(band)), float(np.std(band))])
    flux = (
        np.maximum(mag[:, 1:] - mag[:, :-1], 0.0).mean(axis=0)
        if mag.shape[1] > 1
        else mag[0]
    )
    features.extend([float(np.mean(flux)), float(np.std(flux))])
    return np.asarray(features, dtype=np.float32)


def surrogate_coherence(original: np.ndarray, transformed: np.ndarray) -> float:
    """Convert surrogate feature displacement into a 0..1 coherence score."""

    scale = float(np.linalg.norm(original)) + EPS
    relative = float(np.linalg.norm(transformed - original) / scale)
    return clamp(math.exp(-SURROGATE_COHERENCE_K * relative), 0.0, 1.0)


def read_wav_float(path: Path, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV into a float32 (samples x channels) array."""

    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        total = handle.getnframes()
        if max_seconds is not None:
            total = min(total, int(max_seconds * sample_rate))
        raw = handle.readframes(total)
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
    if channels > 1:
        data = data.reshape(-1, channels)
    else:
        data = data.reshape(-1, 1)
    return data, sample_rate


def codec_round_trip(
    pcm_path: Path, encoder: str, bitrate: str, sample_rate: int, work_dir: Path
) -> np.ndarray:
    """Encode a WAV with ``encoder`` then decode back to float PCM.

    This exercises real lossy compression so robustness is measured, not
    asserted. Returns the decoded (samples x channels) float array.
    """

    suffix = "m4a" if encoder == "aac" else "mp3"
    encoded = work_dir / f"probe.{suffix}"
    decoded_raw = work_dir / "probe.decoded.s16le"
    run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(pcm_path),
            "-codec:a",
            encoder,
            "-b:a",
            bitrate,
            str(encoded),
        ]
    )
    run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(encoded),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            str(CHANNELS),
            "-ar",
            str(sample_rate),
            str(decoded_raw),
        ]
    )
    data = np.fromfile(decoded_raw, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
    return data.reshape(-1, CHANNELS)


def write_probe_wav(block: np.ndarray, path: Path, sample_rate: int) -> None:
    """Write a float (samples x channels) array as a 16-bit PCM WAV."""

    clipped = np.clip(block, -0.999969, 0.999969)
    i16 = np.round(clipped * INT16_PEAK).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(i16.tobytes())


def measure_codec_robustness(
    original_probe: np.ndarray,
    perturbed_probe: np.ndarray,
    sample_rate: int,
    n_fft: int,
) -> list[CodecRobustness]:
    """Round-trip both probes through each codec and measure surviving divergence.

    For each codec we compute:
    * baseline coherence = original vs (original through codec), i.e. how much
      the codec alone perturbs the surrogate features, and
    * perturbed coherence = (perturbed through codec) vs original.
    The surviving divergence is the displacement attributable to the lattice
    over and above the codec's own distortion.
    """

    if shutil.which("ffmpeg") is None:
        return []
    base_features = surrogate_features(original_probe, sample_rate, n_fft)
    results: list[CodecRobustness] = []
    with tempfile.TemporaryDirectory(prefix="umbr-codec-") as temp_dir:
        work = Path(temp_dir)
        original_wav = work / "original.wav"
        perturbed_wav = work / "perturbed.wav"
        write_probe_wav(original_probe, original_wav, sample_rate)
        write_probe_wav(perturbed_probe, perturbed_wav, sample_rate)
        for label, encoder, bitrate in ROBUSTNESS_CODECS:
            try:
                orig_rt = codec_round_trip(original_wav, encoder, bitrate, sample_rate, work)
                pert_rt = codec_round_trip(perturbed_wav, encoder, bitrate, sample_rate, work)
            except RuntimeError as error:
                log("robustness", f"{label} skipped: {error}")
                continue
            orig_rt_feat = surrogate_features(orig_rt, sample_rate, n_fft)
            pert_rt_feat = surrogate_features(pert_rt, sample_rate, n_fft)
            baseline = surrogate_coherence(base_features, orig_rt_feat)
            perturbed = surrogate_coherence(base_features, pert_rt_feat)
            scale = float(np.linalg.norm(base_features)) + EPS
            base_disp = float(np.linalg.norm(orig_rt_feat - base_features)) / scale
            pert_disp = float(np.linalg.norm(pert_rt_feat - base_features)) / scale
            results.append(
                CodecRobustness(
                    label=label,
                    encoder=encoder,
                    bitrate=bitrate,
                    baseline_coherence=baseline,
                    perturbed_coherence=perturbed,
                    surviving_divergence=clamp(pert_disp - base_disp, 0.0, 1.0),
                )
            )
    return results


def fingerprint_string(path: Path) -> str | None:
    """Return the Chromaprint fingerprint of ``path`` if ``fpcalc`` exists."""

    if shutil.which("fpcalc") is None:
        return None
    result = subprocess.run(
        ["fpcalc", "-raw", "-plain", str(path)],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace").strip()


def measure_fingerprint(original_wav: Path, perturbed_wav: Path) -> FingerprintCheck:
    """Cross-check the surrogate with a real fingerprinter when available."""

    if shutil.which("fpcalc") is None:
        return FingerprintCheck(
            available=False,
            note="fpcalc (Chromaprint) not installed; surrogate not cross-checked",
        )
    original_fp = fingerprint_string(original_wav)
    perturbed_fp = fingerprint_string(perturbed_wav)
    if not original_fp or not perturbed_fp:
        return FingerprintCheck(available=False, note="fpcalc produced no fingerprint")
    a = original_fp.split(",")
    b = perturbed_fp.split(",")
    width = min(len(a), len(b))
    if width == 0:
        return FingerprintCheck(available=False, note="empty fingerprint")
    diffs = sum(1 for i in range(width) if a[i] != b[i])
    return FingerprintCheck(
        available=True,
        note=f"compared {width} Chromaprint symbols",
        hamming_fraction=diffs / width,
    )


def aggregate_robustness(codecs: list[CodecRobustness]) -> float:
    """Combine per-codec surviving divergence into a single 0..1 score.

    The score reflects how much lattice-induced representation change *survives*
    real compression, averaged over the exercised codecs. Zero means the codec
    erased the perturbation entirely.
    """

    if not codecs:
        return 0.0
    return clamp(float(np.mean([codec.surviving_divergence for codec in codecs])), 0.0, 1.0)


@dataclass
class RenderState:
    """Mutable accumulators carried across chunks during a render."""

    all_audits: list[RegionAudit] = field(default_factory=list)
    delta_sum: float = 0.0
    delta_peak: float = 0.0
    output_peak: float = 0.0
    clipping_samples: int = 0
    processed_samples: int = 0
    safe_ratios: list[float] = field(default_factory=list)
    densities: list[float] = field(default_factory=list)
    sample_a: np.ndarray | None = None
    sample_b: np.ndarray | None = None
    probe_original: np.ndarray | None = None
    probe_perturbed: np.ndarray | None = None


def transform_block(
    block: np.ndarray, config: StftConfig, strength_db: float, seed: int, chunk_index: int
) -> tuple[np.ndarray, list[RegionAudit], dict[str, float]]:
    """Perturb one stereo block and return transformed audio, audits, and stats.

    Both channels are refined independently; audits and stats are derived from
    the left channel using the strength the refinement loop actually settled on.
    """

    transformed_channels = []
    chunk_audits: list[RegionAudit] = []
    stats: dict[str, float] = {"safe_bin_ratio": 0.0, "mean_density": 0.0, "mean_ratio": 0.0}
    for channel_index in range(CHANNELS):
        spectrum, original_len = hann_stft(block[:, channel_index], config.n_fft, config.hop)
        context = LatticeContext(strength_db, seed + channel_index * 8191, chunk_index)
        transformed, ratio, used_db = refine_channel(
            spectrum, config, context, original_len
        )
        transformed_channels.append(transformed)
        if channel_index == 0:
            density_bundle = estimate_density(np.abs(spectrum).astype(np.float32) + EPS)
            audit_context = LatticeContext(used_db, seed, chunk_index)
            perturbation, _ = build_perturbation(
                spectrum, config, audit_context, density_bundle
            )
            chunk_audits, stats = collect_region_audits(
                ratio, perturbation, config, used_db, density_bundle
            )
    transformed_block = np.column_stack(transformed_channels)
    return transformed_block, chunk_audits, stats


def render_chunk(
    block: np.ndarray,
    start: int,
    chunk_index: int,
    config: StftConfig,
    strength_db: float,
    seed: int,
    state: RenderState,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one chunk, update ``state``, and return (output, delta) arrays."""

    transformed_block, chunk_audits, stats = transform_block(
        block, config, strength_db, seed, chunk_index
    )

    block_peak = float(np.max(np.abs(transformed_block))) if transformed_block.size else 0.0
    if block_peak > 0.999:
        transformed_block *= 0.999 / block_peak
    transformed_block = np.clip(transformed_block, -0.999969, 0.999969)
    delta = transformed_block - block

    state.delta_sum += float(np.sum(delta * delta))
    state.delta_peak = max(state.delta_peak, float(np.max(np.abs(delta))) if delta.size else 0.0)
    state.output_peak = max(
        state.output_peak,
        float(np.max(np.abs(transformed_block))) if transformed_block.size else 0.0,
    )
    state.clipping_samples += int(np.count_nonzero(np.abs(transformed_block) >= 0.999969))
    state.safe_ratios.append(stats["safe_bin_ratio"])
    state.densities.append(stats["mean_density"])

    sr = config.sample_rate
    if state.sample_a is None:
        state.sample_a = block[: min(block.shape[0], sr * 20)].copy()
        state.sample_b = transformed_block[: min(transformed_block.shape[0], sr * 20)].copy()

    probe_target = int(ROBUSTNESS_PROBE_SECONDS * sr)
    if state.probe_original is None or state.probe_perturbed is None:
        state.probe_original = block.copy()
        state.probe_perturbed = transformed_block.copy()
    elif state.probe_original.shape[0] < probe_target:
        state.probe_original = np.vstack([state.probe_original, block])
        state.probe_perturbed = np.vstack([state.probe_perturbed, transformed_block])

    chunk_start_seconds = start / sr
    for audit in chunk_audits:
        audit.start = chunk_start_seconds + audit.index * config.hop / sr
        audit.end = audit.start + config.n_fft / sr
        audit.index = len(state.all_audits)
        state.all_audits.append(audit)

    state.processed_samples += block.shape[0]
    return transformed_block, delta


def render_to_wav(
    raw: np.ndarray,
    total_samples: int,
    config: StftConfig,
    args: argparse.Namespace,
    output: Path,
    delta_output: Path,
) -> RenderState:
    """Stream every chunk to the output and inspection WAVs, returning state."""

    chunk_samples = int(args.chunk_seconds * config.sample_rate)
    strength = Strength.from_name(args.strength).value
    seed = int(hashlib.sha256(args.seed.encode("utf-8")).hexdigest()[:8], 16)
    delta_gain = 10.0 ** (DELTA_INSPECTION_GAIN_DB / 20.0)
    state = RenderState()

    with wave.open(str(output), "wb") as wav, wave.open(str(delta_output), "wb") as delta_wav:
        for handle in (wav, delta_wav):
            handle.setnchannels(CHANNELS)
            handle.setsampwidth(2)
            handle.setframerate(config.sample_rate)

        for chunk_index, start in enumerate(range(0, total_samples, chunk_samples)):
            end = min(total_samples, start + chunk_samples)
            block = np.asarray(raw[start:end], dtype=np.int16).astype(np.float32) / INT16_FULL_SCALE
            transformed_block, delta = render_chunk(
                block, start, chunk_index, config, strength, seed, state
            )
            out_i16 = np.round(transformed_block * INT16_PEAK).astype("<i2")
            wav.writeframes(out_i16.tobytes())
            delta_inspection = np.clip(delta * delta_gain, -0.999969, 0.999969)
            delta_i16 = np.round(delta_inspection * INT16_PEAK).astype("<i2")
            delta_wav.writeframes(delta_i16.tobytes())
            if chunk_index == 0 or (chunk_index + 1) % 10 == 0:
                log("process", f"{state.processed_samples / config.sample_rate:.1f}s rendered")
    return state


def build_metrics(
    state: RenderState,
    config: StftConfig,
    source: Path,
    output: Path,
    delta_output: Path,
    duration: float,
) -> Metrics:
    """Measure divergence, robustness, and quality from the finished render."""

    assert state.sample_a is not None and state.sample_b is not None
    assert state.probe_original is not None and state.probe_perturbed is not None
    probe_target = int(ROBUSTNESS_PROBE_SECONDS * config.sample_rate)
    original_probe = state.probe_original[:probe_target]
    perturbed_probe = state.probe_perturbed[:probe_target]

    delta_rms = math.sqrt(state.delta_sum / max(1, state.processed_samples * CHANNELS))

    log("robustness", "round-tripping a probe through MP3 and AAC")
    codecs = measure_codec_robustness(
        original_probe, perturbed_probe, config.sample_rate, config.n_fft
    )
    robustness = aggregate_robustness(codecs)

    # Honest before/after surrogate coherence on the audible (clean-decode) probe.
    base_feat = surrogate_features(original_probe, config.sample_rate, config.n_fft)
    post_feat = surrogate_features(perturbed_probe, config.sample_rate, config.n_fft)
    # Self-coherence: identical input compared to itself, i.e. the numerical
    # ceiling of the metric (should be 1.0); reported instead of hardcoding it.
    self_coherence = surrogate_coherence(base_feat, base_feat)
    post_coherence = surrogate_coherence(base_feat, post_feat)
    divergence = clamp(self_coherence - post_coherence, 0.0, 1.0)

    log("fingerprint", "checking for Chromaprint cross-check")
    fingerprint = run_fingerprint_check(original_probe, perturbed_probe, config.sample_rate)

    lsd = log_spectral_distance(state.sample_a, state.sample_b, config.n_fft)
    # De-overlapped, single-channel modified-time fraction: distinct frames are
    # counted once and scaled by hop, divided by the per-channel sample count.
    distinct_frames = len({(audit.start, audit.low_hz) for audit in state.all_audits})
    modified_time_ratio = min(
        1.0, distinct_frames * config.hop / max(1, state.processed_samples)
    )

    return Metrics(
        source=str(source),
        output=str(output),
        delta_inspection_output=str(delta_output),
        sample_rate=config.sample_rate,
        channels=CHANNELS,
        duration_seconds=duration,
        processed_seconds=state.processed_samples / config.sample_rate,
        n_fft=config.n_fft,
        hop_size=config.hop,
        modified_frames=len(state.all_audits),
        modified_time_ratio=modified_time_ratio,
        delta_rms_dbfs=dbfs(delta_rms),
        delta_peak_dbfs=dbfs(state.delta_peak),
        output_peak_dbfs=dbfs(state.output_peak),
        log_spectral_distance_db=lsd,
        transparency_gate_dbfs=TRANSPARENCY_GATE_DBFS,
        transparency_gate_pass=dbfs(delta_rms) <= TRANSPARENCY_GATE_DBFS,
        surrogate_self_coherence=self_coherence,
        surrogate_post_coherence=post_coherence,
        surrogate_divergence=divergence,
        fingerprint=fingerprint,
        codec_robustness=codecs,
        robustness_score=robustness,
        clipping_samples=state.clipping_samples,
    )


def resolve_output_paths(args: argparse.Namespace, source: Path) -> tuple[Path, Path]:
    """Derive the output and delta-inspection paths, honouring explicit flags.

    With no ``--output``, the rendered WAV lands next to the source as
    ``<stem>.umbr.wav``; the delta WAV defaults to ``<output-stem>.delta_x60dB.wav``.
    """

    output = (
        Path(args.output)
        if args.output
        else source.with_name(f"{source.stem}{DEFAULT_OUTPUT_SUFFIX}")
    )
    delta_output = (
        Path(args.delta_output)
        if args.delta_output
        else output.with_name(f"{output.stem}{DEFAULT_DELTA_SUFFIX}")
    )
    return output, delta_output


def process_audio(args: argparse.Namespace) -> Metrics:
    """Decode, perturb (with refinement), reconstruct, and export the audio."""

    source = Path(args.source).expanduser()
    output, delta_output = resolve_output_paths(args, source)
    artifacts = Path(args.artifacts)
    output.parent.mkdir(parents=True, exist_ok=True)
    delta_output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    duration = ffprobe_duration(source)
    config = StftConfig(args.sample_rate, args.n_fft, args.hop)
    limit_samples = int(args.limit_seconds * config.sample_rate) if args.limit_seconds else 0

    log("decode", f"{source} -> temporary {config.sample_rate} Hz PCM")
    with tempfile.TemporaryDirectory(prefix="umbr-") as temp_dir:
        raw_path = Path(temp_dir) / "input.s16le"
        total_samples = decode_to_raw(source, raw_path, config.sample_rate, CHANNELS)
        if limit_samples:
            total_samples = min(total_samples, limit_samples)
        raw = np.memmap(raw_path, dtype=np.int16, mode="r", shape=(total_samples, CHANNELS))
        state = render_to_wav(raw, total_samples, config, args, output, delta_output)

    if state.sample_a is None or state.probe_original is None:
        raise RuntimeError("no audio frames were processed")

    metrics = build_metrics(state, config, source, output, delta_output, duration)
    write_artifacts(artifacts, state.all_audits, metrics)
    return metrics


def run_fingerprint_check(
    probe_original: np.ndarray, probe_perturbed: np.ndarray, sample_rate: int
) -> FingerprintCheck:
    """Write both probes to temp WAVs and run the Chromaprint cross-check."""

    if shutil.which("fpcalc") is None:
        return FingerprintCheck(
            available=False,
            note="fpcalc (Chromaprint) not installed; surrogate not cross-checked",
        )
    with tempfile.TemporaryDirectory(prefix="umbr-fp-") as temp_dir:
        work = Path(temp_dir)
        original_wav = work / "original.wav"
        perturbed_wav = work / "perturbed.wav"
        write_probe_wav(probe_original, original_wav, sample_rate)
        write_probe_wav(probe_perturbed, perturbed_wav, sample_rate)
        return measure_fingerprint(original_wav, perturbed_wav)


def write_artifacts(artifacts: Path, audits: list[RegionAudit], metrics: Metrics) -> None:
    """Write CSV, JSON, and Markdown audit artifacts."""

    regions_csv = artifacts / "umbr_regions.csv"
    metrics_json = artifacts / "umbr_metrics.json"
    report_md = artifacts / "umbr_audit.md"
    bins_csv = artifacts / "umbr_spectrogram_bins.csv"

    with regions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RegionAudit.__dataclass_fields__))
        writer.writeheader()
        for audit in audits:
            writer.writerow(asdict(audit))

    with bins_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "start",
                "end",
                "low_hz",
                "high_hz",
                "perturbation_dbfs",
                "density",
                "cover",
            ],
        )
        writer.writeheader()
        for audit in sorted(audits, key=lambda item: item.perturbation_dbfs, reverse=True)[:5000]:
            writer.writerow(
                {
                    "start": f"{audit.start:.3f}",
                    "end": f"{audit.end:.3f}",
                    "low_hz": f"{audit.low_hz:.1f}",
                    "high_hz": f"{audit.high_hz:.1f}",
                    "perturbation_dbfs": f"{audit.perturbation_dbfs:.2f}",
                    "density": f"{audit.density:.4f}",
                    "cover": audit.cover,
                }
            )

    metrics_json.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_report(audits, metrics), encoding="utf-8")


def render_report(audits: list[RegionAudit], metrics: Metrics) -> str:
    """Build the human-readable Markdown audit text from measured metrics."""

    top = sorted(audits, key=lambda item: item.perturbation_dbfs, reverse=True)[:12]
    gate = "PASS" if metrics.transparency_gate_pass else "FAIL"
    lines = [
        "# Umbr Audit",
        "",
        f"Source: `{metrics.source}`",
        f"Output: `{metrics.output}`",
        f"Amplified delta inspection WAV: `{metrics.delta_inspection_output}`",
        f"Processed: {metrics.processed_seconds:.2f}s of {metrics.duration_seconds:.2f}s",
        f"STFT: n_fft={metrics.n_fft}, hop={metrics.hop_size}, sample_rate={metrics.sample_rate}",
        "",
        "## Perceptual Transparency Gates",
        "",
        f"Transparency gate: {metrics.transparency_gate_dbfs:.1f} dBFS ceiling -> {gate}",
        f"Delta RMS: {metrics.delta_rms_dbfs:.2f} dBFS",
        f"Delta peak: {metrics.delta_peak_dbfs:.2f} dBFS",
        f"Output peak: {metrics.output_peak_dbfs:.2f} dBFS",
        f"Log-spectral distance proxy: {metrics.log_spectral_distance_db:.4f} dB",
        f"Clipping samples: {metrics.clipping_samples}",
        "",
        "## Surrogate Readout (internal proxy)",
        "",
        "The surrogate is a self-defined band-energy proxy, not a model of any "
        "specific real retrieval system. Treat divergence as suggestive.",
        "",
        f"Self-coherence ceiling: {metrics.surrogate_self_coherence:.4f}",
        f"Post-lattice coherence: {metrics.surrogate_post_coherence:.4f}",
        f"Surrogate divergence: {metrics.surrogate_divergence:.4f}",
        "",
        "## Fingerprint Cross-Check (Chromaprint)",
        "",
        f"Available: {metrics.fingerprint.available}",
        f"Note: {metrics.fingerprint.note}",
        f"Symbol mismatch fraction: {metrics.fingerprint.hamming_fraction:.4f}",
        "",
        "## Processing Robustness (measured via real codecs)",
        "",
        f"Aggregate surviving divergence: {metrics.robustness_score:.4f}",
        "",
    ]
    if metrics.codec_robustness:
        for codec in metrics.codec_robustness:
            lines.extend(
                [
                    f"### {codec.label} ({codec.encoder} @ {codec.bitrate})",
                    "",
                    f"Codec-only baseline coherence: {codec.baseline_coherence:.4f}",
                    f"Perturbed coherence: {codec.perturbed_coherence:.4f}",
                    f"Surviving divergence (lattice over codec): {codec.surviving_divergence:.4f}",
                    "",
                ]
            )
    else:
        lines.extend(["No codec round-trip results (ffmpeg unavailable).", ""])
    lines.extend(
        [
            "## Spectrogram Inspection Coordinates",
            "",
            f"Modified STFT frames: {metrics.modified_frames}",
            f"Modified time ratio: {metrics.modified_time_ratio * 100.0:.2f}%",
            "",
        ]
    )
    for index, audit in enumerate(top, 1):
        lines.extend(
            [
                f"### Region {index}",
                "",
                f"Time: {audit.start:.3f}s - {audit.end:.3f}s",
                f"Band: {audit.low_hz:.1f} Hz - {audit.high_hz:.1f} Hz",
                f"Perturbation: {audit.perturbation_dbfs:.2f} dBFS",
                f"Density: {audit.density:.4f}",
                f"Cover: {audit.cover}",
                f"Reason: {audit.reason}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Listening Verification",
            "",
            "The exported file is intended for blind listening and external "
            "spectrogram inspection. Sparse passages are passed through "
            "unchanged; modified bins are constrained to dense, masked "
            "time-frequency neighborhoods. The lattice is deterministic and "
            "non-semantic: it does not encode text, speech, or a recoverable "
            "message.",
            "",
            "## Known Failure Modes",
            "",
            "- The internal surrogate is self-defined; divergence against it does "
            "not guarantee divergence against a real retrieval system.",
            "- Sub-audible energy is, by construction, the first thing lossy "
            "codecs discard: low surviving divergence is the expected result and "
            "is reported honestly rather than masked.",
            "- Exposed vocals, fades, and sparse instruments overrule every "
            "numeric score and must be confirmed by blind listening.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(
        prog="umbr",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Apply a masked STFT-domain perturbation lattice to an input track.",
    )
    parser.add_argument(
        "source",
        help="Input audio file (any format ffmpeg can decode).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Output WAV; empty derives '<source-stem>.umbr.wav' next to the source.",
    )
    parser.add_argument(
        "--delta-output",
        default="",
        help="Amplified difference WAV for spectrogram inspection; empty derives from output name.",
    )
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--chunk-seconds", type=float, default=20.0)
    parser.add_argument(
        "--strength",
        choices=[member.name.lower() for member in Strength],
        default=Strength.MEDIUM.name.lower(),
    )
    parser.add_argument("--seed", default="umbr-research")
    parser.add_argument(
        "--limit-seconds",
        type=float,
        default=0.0,
        help="Process only the first N seconds; 0 means the full track.",
    )
    args = parser.parse_args()
    if args.n_fft < 512 or args.n_fft > 8192 or args.n_fft & (args.n_fft - 1):
        parser.error("--n-fft must be a power of two between 512 and 8192")
    if args.hop <= 0 or args.hop > args.n_fft:
        parser.error("--hop must be positive and no larger than --n-fft")
    if args.chunk_seconds < 1.0:
        parser.error("--chunk-seconds must be at least 1")
    if args.sample_rate < 16_000 or args.sample_rate > 96_000:
        parser.error("--sample-rate must be between 16000 and 96000")
    return args


def main() -> int:
    """Run the umbr pipeline."""

    args = parse_args()
    if not Path(args.source).expanduser().exists():
        raise SystemExit(f"missing source audio: {args.source}")
    log("boot", "STFT umbr console initialized")
    metrics = process_audio(args)
    log("export", metrics.output)
    log("audit", f"{args.artifacts}/umbr_audit.md")
    log(
        "quality",
        f"delta_rms={metrics.delta_rms_dbfs:.2f} dBFS "
        f"divergence={metrics.surrogate_divergence:.4f} "
        f"robustness={metrics.robustness_score:.4f}",
    )
    return 0


def cli() -> None:
    """Console-script entry point invoked by the ``umbr`` command."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
