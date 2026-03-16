# guitar-strings

> Vibed into existence with Claude Code.

A Python analysis pipeline that measures how pluck intensity affects pitch on guitar strings — the phenomenon where harder plucks produce slightly sharper notes due to increased string tension.

## What it does

For each recording in `data/`, the pipeline:

1. **Detects pluck onsets** using spectral flux with adaptive thresholding
2. **Estimates pitch** via autocorrelation (robust against harmonics dominating the spectrum)
3. **Extracts RMS amplitude** per pluck as a proxy for pluck intensity
4. **Fits a Huber robust regression** of pitch vs. amplitude
5. **Generates three figures** per string:
   - Waveform overview with detected onsets and pitch overlay
   - Scatter plot with regression line showing the pitch-intensity relationship
   - Side-by-side comparison of a soft vs. hard pluck

## Strings supported

| String | Nominal frequency |
|--------|------------------|
| Bass E (E1) | 41.20 Hz |
| Low E (E2) | 82.41 Hz |
| G (G3) | 196.00 Hz |
| High E (E4) | 329.63 Hz |

WAV filenames are matched to strings by keywords (`bass`, `low-e`, `g-string`, `high-e`).

## Usage

```bash
# Place .wav recordings in data/
uv run python main.py
```

Output PNGs are written to the project root, prefixed by string name (e.g., `e1_fig1_waveform.png`).

## Dependencies

```
numpy
scipy
statsmodels
matplotlib
```
