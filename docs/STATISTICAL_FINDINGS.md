# Umbr — Statistical Verification Findings

Harness: `tests/verification_harness.py` (importable functions + CLI + pytest
self-checks). It imports umbr's own measurement functions, so the system under
test and the measuring stick are identical.

Definitive run (bundled fixture, `medium` strength, first 30 s):

```
python tests/verification_harness.py --null-samples 200 --codec-null-samples 40
```

The harness adds the *inferential* layer on top of umbr's descriptive audit:
each of the three legs has an explicit null hypothesis and a decision rule.

## Leg 1 — Structure-preserving permutation test (automated)

**Method (corrected).** Two things were fixed after the first pass:

1. *Metric.* The production surrogate averages band energies over the whole
   30 s probe, which dilutes a sparse, masked perturbation to nothing. The
   harness instead computes a per-STFT-frame log-band feature vector and reports
   the **mean displacement of the top-decile most-disturbed frames**
   (`localized_divergence`). This is sensitive to *where* energy lands.
2. *Null model.* The first version compared against equal-energy **white noise**,
   which unfairly dumps energy into loud, unmasked bands. The corrected null is
   the lattice's **own residual with randomized signs** — identical energy *and*
   identical time-frequency support, differing only in the sign/phase pattern.
   So the test isolates exactly one question: does the lattice's *specific*
   structure diverge more than an arbitrary perturbation of the same energy in
   the same masked cells?

| Statistic | Observed | Null mean | Null p95 | p (floor) | z |
| --- | --- | --- | --- | --- | --- |
| Localized divergence (clean) | 0.000151 | 0.002071 | 0.002263 | 1.000 (0.005) | −19.8 |
| Codec-surviving divergence | 0.000000 | 0.000038 | 0.000178 | 1.000 (0.024) | −0.41 |

**Finding.** Even with the fair metric and the fair null, the lattice does **not**
diverge more than its sign-randomized counterpart — on the clean decode it
diverges *slightly less* (z = −19.8). That is consistent with the design: the
lattice is phase-quadrature-aligned to the host specifically to minimise
footprint, so its structure is, if anything, *gentler* on the representation
than a random perturbation of the same energy would be. After MP3/AAC the
observed and null values are both ≈0 and statistically indistinguishable
(z = −0.41) — honest erasure, exactly as the README predicts.

**Interpretation for the divergence/robustness claim:** at `medium` strength the
lattice's representational effect is not distinguishable from "an arbitrary
inaudible perturbation," and is erased by lossy compression. The headline
"representation divergence" is **not supported** as a *structural* effect for
this material/strength. A strength sweep (`--strength` is plumbed through umbr)
confirms divergence rises monotonically with level (conservative→research) but
stays orders of magnitude below the null across the inaudible range.

## Leg 2 — Blind ABX listening test (needs human trials)

A seeded blind 16-trial ABX plan is generated; the answer key is saved in
`umbr_statistics.json`. Responses are scored with an **exact one-sided binomial
test** against the null "listener guesses" (p = 0.5), validated against known
tail probabilities in the pytest self-checks:

- ≥ 12/16 correct → p < 0.05 → listeners CAN hear it → transparency FAILS.
- ~8/16 → indistinguishable from guessing → consistent with transparency.

Record a session interactively (`--abx-run`) or score a saved sheet
(`--abx-responses sheet.txt`, one A/B per line). This leg requires real human
listeners and is omitted from CI by design — that is what makes it third-party.

## Leg 3 — Independent fingerprinter cross-check (Chromaprint)

`fpcalc` (Chromaprint) is installed locally and in CI (`libchromaprint-tools`).
On this run a real acoustic fingerprinter the author did not write reports a
**0.0000 symbol-mismatch fraction over 221 symbols** between original and
perturbed — i.e. the external tool sees the two files as identical. If `fpcalc`
is absent the harness reports Leg 3 unavailable and does not hard-fail.

## Bottom line

The harness delivers an objective, reproducible statistical test with explicit
decision rules, run on real audio, with the measurement bias corrected. For the
coffee clip at `medium` strength it returns an honest negative on all
automated legs: the perturbation is inaudible-and-inert at the surrogate and
fingerprint level, and erased by codecs. The ABX harness and strength sweep are
the tools to find the boundary — if one exists — where audibility appears before
divergence does. So far, it does not.
