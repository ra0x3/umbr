# umbr

Sub-audible, STFT-domain audio perturbation research console.

`umbr` (from *umbra*, the fully shadowed core of a shadow) takes an input track
and produces a transformed version that is intended to sound perceptually
identical to a human listener, while its machine-readable representation is
nudged away from the original. Every claim the tool makes about transparency,
representation divergence, and processing robustness is **measured** and written
to an audit report - nothing is asserted or simulated.

This is a research instrument for studying the tension between *inaudibility*
and *survivability under lossy processing*. It does **not** encode messages or
payloads: the perturbation is a deterministic, non-semantic sign lattice with no
recoverable content.

> Status: alpha / research prototype. Expect the honest finding that a
> perturbation quiet enough to be inaudible is largely erased by ordinary lossy
> compression. The point of the tool is to quantify that, not to hide it.

## Installation

```bash
pip install umbr
```

`umbr` requires Python 3.12+ and depends on [NumPy](https://numpy.org/) (>= 2.0).

It also shells out to **FFmpeg** for decoding, codec round-trips, and duration
probing, so `ffmpeg` and `ffprobe` must be on your `PATH`
(<https://ffmpeg.org/download.html>). The optional fingerprint cross-check uses
the Chromaprint `fpcalc` binary if present
(<https://acoustid.org/chromaprint>); when it is missing the stage degrades
gracefully and says so in the audit.

## Usage

The source track is required; everything else is derived or optional:

```bash
# Output derives to <source-stem>.umbr.wav next to the source.
umbr song.flac

# Explicit output, a research-strength preset, first 60s only.
umbr song.flac -o out/song.umbr.wav --strength research --limit-seconds 60
```

Key options (`umbr --help` for the full list):

| Option | Default | Meaning |
| --- | --- | --- |
| `source` | *(required)* | Input audio, any format FFmpeg can decode. |
| `-o, --output` | `<stem>.umbr.wav` | Rendered WAV. |
| `--delta-output` | `<output-stem>.delta_x60dB.wav` | Residual amplified +60 dB for spectrogram inspection. |
| `--artifacts` | `artifacts` | Directory for the CSV/JSON/Markdown audit. |
| `--strength` | `medium` | `conservative` / `medium` / `research` (quieter to louder). |
| `--n-fft` | `2048` | STFT window size (power of two, 512-8192). |
| `--hop` | `512` | STFT hop size. |
| `--sample-rate` | `44100` | Internal working rate (16000-96000). |
| `--limit-seconds` | `0` (full) | Process only the first N seconds. |

## How it works

The pipeline mirrors the spec stages, all in the short-time Fourier domain.

1. **Ingestion / normalization** - FFmpeg decodes the source to 16-bit PCM at
   the working sample rate.
2. **Spectrogram analysis** - a Hann-windowed
   [Short-Time Fourier Transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
   with overlap-add resynthesis (the
   [Constant-OverLap-Add / COLA](https://www.dsprelated.com/freebooks/sasp/Overlap_Add_OLA_STFT_Processing.html)
   condition keeps reconstruction transparent).
3. **Psychoacoustic masking map** - a conservative
   [simultaneous-masking](https://en.wikipedia.org/wiki/Auditory_masking) proxy
   combining a near-masker threshold, an
   [absolute-threshold-of-hearing](https://en.wikipedia.org/wiki/Absolute_threshold_of_hearing)
   penalty, per-frame
   [spectral flux](https://en.wikipedia.org/wiki/Spectral_flux) and
   [spectral flatness](https://en.wikipedia.org/wiki/Spectral_flatness), and a
   loudness estimate. Only dense, masked time-frequency regions are eligible.
4. **Candidate perturbation** - a deterministic +/- lattice
   ([NumPy PCG64 generator](https://numpy.org/doc/stable/reference/random/generator.html)),
   phase-shifted a quarter turn from the host and scaled under the masking
   budget.
5. **Surrogate evaluation** - a small, explicitly labelled band-energy *proxy*
   for an audio-analysis system. Divergence against it is suggestive, not
   conclusive (see *Caveats*).
6. **Robustness refinement** - the rendered probe is round-tripped through real
   lossy codecs ([MP3 / LAME](https://lame.sourceforge.io/) and
   [AAC](https://en.wikipedia.org/wiki/Advanced_Audio_Coding)) via FFmpeg and
   re-measured, so robustness is observed rather than modelled.
7. **Perceptual quality scoring** - residual RMS / peak in
   [dBFS](https://en.wikipedia.org/wiki/DBFS), a
   [log-spectral distance](https://en.wikipedia.org/wiki/Log-spectral_distance)
   proxy, and a transparency gate that the refinement loop must satisfy.
8. **Export** - the transformed WAV plus a +60 dB amplified residual WAV for
   spectrogram inspection.
9. **Human-listening verification** - the audit lists exactly which regions were
   modified and why they were judged psychoacoustically safe, for blind ABX
   listening.

## Documented constants

All tunables live as documented module-level constants near the top of
`umbr.py`. The most important:

| Constant | Value | Reference |
| --- | --- | --- |
| `STRENGTH` presets (dB) | `-62 / -56 / -50` | dB relative to local host magnitude; see [dBFS](https://en.wikipedia.org/wiki/DBFS). |
| `TRANSPARENCY_GATE_DBFS` | `-85.0` | Residual RMS ceiling the refinement loop must meet. |
| `NEAR_MASKER_DB` | `-38.0` | Masking threshold below the per-frame peak ([auditory masking](https://en.wikipedia.org/wiki/Auditory_masking)). |
| `PERTURB_BAND_LOW/HIGH_HZ` | `120 / 15500` | Eligible band; avoids fragile sub-bass and codec-stripped air. |
| `ATH_PENALTY_*` | `0.40 / 0.70 / 0.55` | [Absolute threshold of hearing](https://en.wikipedia.org/wiki/Absolute_threshold_of_hearing) weighting. |
| `PERTURB_PHASE_OFFSET` | `pi / 2` | Quadrature offset from the host phase. |
| `SURROGATE_BAND_EDGES_HZ` | `80 .. 15500` | Log-band edges of the surrogate embedding. |
| `ROBUSTNESS_CODECS` | MP3 128k, AAC 128k | Codecs exercised by the robustness round-trip. |

`Strength` and `CoverType` are `enum.Enum` types rather than bare strings, so the
CLI choices, the dB levels, and the audit phrasing stay in sync.

## Audit output

Written to the `--artifacts` directory:

- `umbr_audit.md` - human-readable report: transparency gates, surrogate
  readout, codec robustness, and the top modified regions with their
  psychoacoustic rationale.
- `umbr_metrics.json` - the full `Metrics` record.
- `umbr_regions.csv` - every modified STFT frame.
- `umbr_spectrogram_bins.csv` - the loudest modified bins for spectrogram
  overlay.

## Caveats

- The internal surrogate is **self-defined**; divergence against it does not
  prove divergence against a real retrieval/fingerprinting system. Install
  Chromaprint to enable the independent cross-check.
- Sub-audible energy is, by construction, the first thing lossy codecs discard,
  so low surviving divergence is the expected result and is reported honestly.
- Exposed vocals, fades, and sparse instruments overrule every numeric score and
  must be confirmed by blind listening.

## License

MIT.
