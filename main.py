import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=wav.WavFileWarning)


# Standard guitar tuning open string frequencies
# One whole tone = 2 semitones = factor of 2^(2/12) ≈ 1.122
STRING_CONFIGS = {
    "low-e": {"nominal_hz": 82.41, "name": "Low E (E2)"},
    "g-string": {"nominal_hz": 196.00, "name": "G (G3)"},
    "high-e": {"nominal_hz": 329.63, "name": "High E (E4)"},
    "bass": {"nominal_hz": 41.20, "name": "Bass E (E1)"},
}


def guess_string_config(filename: str) -> dict:
    """Match a wav filename to a string configuration."""
    name = filename.lower()
    if "bass" in name:
        return STRING_CONFIGS["bass"]
    if "high-e" in name or "high_e" in name:
        return STRING_CONFIGS["high-e"]
    if "low-e" in name or "low_e" in name:
        return STRING_CONFIGS["low-e"]
    if "g-string" in name or "g_string" in name:
        return STRING_CONFIGS["g-string"]
    raise ValueError(f"Cannot determine string from filename: {filename}")


def one_whole_tone_range(nominal_hz: float) -> tuple[float, float]:
    """Return (low, high) frequency bounds: one whole tone below and above nominal."""
    factor = 2 ** (2 / 12)
    return nominal_hz / factor, nominal_hz * factor


def load_wav(path: str) -> tuple[int, np.ndarray]:
    """Load a wav file and return (sample_rate, mono_float_signal)."""
    rate, data = wav.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64) / np.iinfo(np.int16).max
    return rate, data


def detect_onsets(
    audio: np.ndarray,
    rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    pre_max_frames: int = 6,
    post_max_frames: int = 6,
    pre_avg_frames: int = 50,
    post_avg_frames: int = 50,
    wait_frames: int = 20,
    delta_ratio: float = 3.0,
    min_delta: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Detect pluck onsets using spectral flux with adaptive thresholding.

    Instead of a fixed delta on the globally-normalized flux, we use a
    multiplicative threshold: a peak must exceed delta_ratio * local_mean.
    This lets us detect quiet plucks in low-energy regions while still
    rejecting noise in loud sustained sections.
    """
    # Compute STFT magnitudes
    _, _, Zxx = signal.stft(audio, fs=rate, nperseg=frame_size, noverlap=frame_size - hop_size)
    mag = np.abs(Zxx)

    # Spectral flux: sum of positive magnitude differences across frames
    diff = np.diff(mag, axis=1)
    diff = np.maximum(diff, 0)
    flux = diff.sum(axis=0)

    # Peak picking with adaptive ratio threshold
    onsets = []
    for i in range(pre_max_frames, len(flux) - post_max_frames):
        # Local maximum check
        window = flux[max(0, i - pre_max_frames):i + post_max_frames + 1]
        if flux[i] < window.max():
            continue

        # Adaptive threshold: must exceed local mean by a ratio
        avg_start = max(0, i - pre_avg_frames)
        avg_end = min(len(flux), i + post_avg_frames + 1)
        local_mean = flux[avg_start:avg_end].mean()
        threshold = max(local_mean * delta_ratio, min_delta)
        if flux[i] < threshold:
            continue

        # Minimum wait between onsets
        if onsets and (i - onsets[-1]) < wait_frames:
            continue

        onsets.append(i)

    onset_frames = np.array(onsets)
    onset_samples = (onset_frames + 1) * hop_size

    return onset_samples, flux, hop_size


def autocorrelation_pitch(segment: np.ndarray, rate: int,
                          f_min: float = 20.0, f_max: float = 500.0) -> float | None:
    """
    Estimate fundamental frequency using autocorrelation.

    Finds the highest autocorrelation peak in the lag range corresponding
    to [f_min, f_max], which reliably finds the fundamental even when
    harmonics are stronger in the spectrum.
    """
    lag_min = int(rate / f_max)
    lag_max = int(rate / f_min)
    lag_max = min(lag_max, len(segment) - 1)

    if lag_min >= lag_max:
        return None

    # Normalized autocorrelation
    windowed = segment * np.hanning(len(segment))
    corr = np.correlate(windowed, windowed, mode='full')
    corr = corr[len(windowed) - 1:]  # keep positive lags only
    corr = corr / (corr[0] + 1e-15)

    # Find the highest peak in the valid lag range
    search_region = corr[lag_min:lag_max + 1]
    if len(search_region) == 0:
        return None

    peak_lag = lag_min + np.argmax(search_region)

    # Require a reasonable correlation strength to reject noise
    if corr[peak_lag] < 0.2:
        return None

    # Parabolic interpolation around the peak for sub-sample accuracy
    if 0 < peak_lag < len(corr) - 1:
        alpha = corr[peak_lag - 1]
        beta = corr[peak_lag]
        gamma = corr[peak_lag + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            correction = 0.5 * (alpha - gamma) / denom
            correction = np.clip(correction, -0.5, 0.5)
            peak_lag = peak_lag + correction

    if peak_lag <= 0:
        return None

    return rate / peak_lag


def extract_features(
    audio: np.ndarray,
    rate: int,
    onset_samples: np.ndarray,
    nominal_hz: float,
    skip_duration: float = 0.05,
    analysis_duration: float = 0.15,
    min_rms: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each pluck, extract dominant frequency (autocorrelation),
    RMS amplitude, and corresponding onset sample indices.
    Skips the initial transient (skip_duration) and then
    analyzes a window of analysis_duration seconds.

    Filters to pitches within one whole tone of nominal_hz.
    """
    n_skip = int(skip_duration * rate)
    n_analysis = int(analysis_duration * rate)

    # Autocorrelation search range: 2 octaves around nominal
    ac_f_min = nominal_hz / 4
    ac_f_max = nominal_hz * 4

    freq_lo, freq_hi = one_whole_tone_range(nominal_hz)

    frequencies = []
    amplitudes = []
    matched_onsets = []

    for onset in onset_samples:
        start = onset + n_skip
        segment = audio[start:start + n_analysis]
        if len(segment) < n_analysis // 2:
            continue

        rms = np.sqrt(np.mean(segment**2))

        freq = autocorrelation_pitch(segment, rate, f_min=ac_f_min, f_max=ac_f_max)
        if freq is None:
            continue

        frequencies.append(freq)
        amplitudes.append(rms)
        matched_onsets.append(onset)

    frequencies = np.array(frequencies)
    amplitudes = np.array(amplitudes)
    matched_onsets = np.array(matched_onsets)

    if len(frequencies) == 0:
        return frequencies, amplitudes, matched_onsets

    # Filter outliers: within one whole tone of nominal, above min RMS
    mask = (frequencies >= freq_lo) & (frequencies <= freq_hi) & (amplitudes >= min_rms)
    return frequencies[mask], amplitudes[mask], matched_onsets[mask]


def rms_envelope(audio: np.ndarray, rate: int, window_ms: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute a smoothed RMS envelope for display."""
    win = int(window_ms / 1000 * rate)
    hop = win // 2
    n_frames = (len(audio) - win) // hop
    env = np.zeros(n_frames)
    t = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        env[i] = np.sqrt(np.mean(audio[start:start + win] ** 2))
        t[i] = (start + win / 2) / rate
    return t, env


def compare_amp_vs_power_fit(freqs: np.ndarray, amps: np.ndarray, string_name: str):
    """
    Compare Huber M-estimator fits for freq ~ RMS vs freq ~ RMS².

    R² is inappropriate for robust regression (it measures OLS residuals, not
    the Huber objective). Instead we compare result.scale, which is the robust
    scale estimate (MAD-based) from each M-estimator — directly analogous to σ̂
    in OLS. Lower scale = tighter residuals under the Huber norm = better fit.
    """
    huber = sm.robust.norms.HuberT()
    result_amp = sm.RLM(freqs, sm.add_constant(amps),      M=huber).fit()
    result_pow = sm.RLM(freqs, sm.add_constant(amps ** 2), M=huber).fit()
    s_amp = result_amp.scale
    s_pow = result_pow.scale
    winner = "RMS" if s_amp <= s_pow else "RMS²"
    print(f"  [{string_name}] Robust scale: RMS={s_amp:.4f}  RMS²={s_pow:.4f}  → {winner} fits better")


def plot_blog_figures(audio, rate, onset_samples, freqs, amps, matched_onsets,
                      flux, hop_size, string_name: str, nominal_hz: float,
                      prefix: str):
    """Generate three publication-quality figures for a given string."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
    })

    BLUE = "#3B82F6"
    RED = "#EF4444"
    ORANGE = "#F59E0B"
    GRAY = "#6B7280"
    DARK = "#1F2937"

    # ── Figure 1: Waveform overview with detected plucks ──
    fig1, ax1 = plt.subplots(figsize=(12, 3.5))
    t_env, env = rms_envelope(audio, rate)
    t_audio = np.arange(len(audio)) / rate

    ax1.fill_between(t_audio, audio, alpha=0.15, color=BLUE, linewidth=0)
    ax1.plot(t_env, env, color=BLUE, linewidth=1.2, label="RMS envelope")
    ax1.plot(t_env, -env, color=BLUE, linewidth=1.2)

    onset_times = onset_samples / rate
    for i, t in enumerate(onset_times):
        ax1.axvline(t, color=RED, alpha=0.35, linewidth=0.6,
                     label=f"{len(onset_samples)} detected plucks" if i == 0 else None)

    ax1.set_xlim(0, len(audio) / rate)
    ax1.set_ylim(-1.05, env.max() * 1.35)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{string_name} String: Recording with Detected Pluck Onsets",
                  fontsize=13, fontweight="bold", pad=12)

    # Right axis: pitch, matching figure 2's y-range
    ax1r = ax1.twinx()
    ax1r.spines["top"].set_visible(False)
    pitch_times = matched_onsets / rate
    ax1r.plot(pitch_times, freqs, marker="o", linestyle="none", color=ORANGE,
              markersize=4, alpha=0.7, zorder=5, label="Estimated pitch")
    # Match figure 2 y-limits
    freq_margin = (freqs.max() - freqs.min()) * 0.1
    ax1r.set_ylim(freqs.min() - freq_margin, freqs.max() + freq_margin)
    ax1r.set_ylabel("Pitch (Hz)", color=ORANGE)
    ax1r.tick_params(axis="y", colors=ORANGE)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", framealpha=0.9, fontsize=9)

    fig1.tight_layout()
    fig1.savefig(f"{prefix}_fig1_waveform.png", dpi=180, bbox_inches="tight")
    print(f"  Saved {prefix}_fig1_waveform.png")
    plt.close(fig1)

    # ── Figure 2: Scatter + Huber regression (the main result) ──
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.scatter(amps, freqs, alpha=0.55, s=35, color=ORANGE, edgecolors=DARK,
                linewidths=0.4, zorder=3)

    huber = sm.robust.norms.HuberT()
    result_quad = sm.RLM(freqs, sm.add_constant(amps ** 2), M=huber).fit()

    amp_range = np.linspace(amps.min(), amps.max(), 200)
    fit_quad = result_quad.params[0] + result_quad.params[1] * amp_range ** 2

    ax2.plot(amp_range, fit_quad, color="#7C3AED", linewidth=2.5, zorder=2)

    # Annotation: formula + p-value (replaces separate legend)
    a, b = result_quad.params
    pval = result_quad.pvalues[1]
    pval_str = "< 0.001" if pval < 0.001 else f"= {pval:.3f}"
    stats_text = (
        f"f\u0302 = {a:.2f} + {b:.4f}\u00b7RMS\u00b2\n"
        f"p {pval_str}  (Huber robust fit)"
    )
    ax2.text(0.03, 0.97, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor=ORANGE, alpha=0.9))

    ax2.set_ylim(freqs.min() - freq_margin, freqs.max() + freq_margin)
    ax2.set_xlabel("Pluck Intensity (RMS amplitude)", fontsize=11)
    ax2.set_ylabel("Detected Pitch (Hz)", fontsize=11)
    ax2.set_title(f"{string_name} String: Harder Plucks Go Sharper",
                  fontsize=14, fontweight="bold", pad=12)

    # Right axis: cents from zero-intensity baseline (fit intercept) — dashed
    baseline_hz = result_quad.params[0]
    ax2r = ax2.twinx()
    hz_lo, hz_hi = ax2.get_ylim()
    ax2r.set_ylim(1200 * np.log2(hz_lo / baseline_hz),
                  1200 * np.log2(hz_hi / baseline_hz))
    ax2r.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:+.0f}")
    )
    ax2r.set_ylabel("Cents from baseline", fontsize=11)
    ax2r.axhline(0, color=GRAY, linestyle="--", linewidth=1, alpha=0.6)

    fig2.tight_layout()
    fig2.savefig(f"{prefix}_fig2_scatter.png", dpi=180, bbox_inches="tight")
    print(f"  Saved {prefix}_fig2_scatter.png")
    plt.close(fig2)

    # ── Figure 3: Soft vs hard pluck waveform comparison ──
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4))

    sorted_idx = np.argsort(amps)
    quiet_idx = sorted_idx[len(sorted_idx) // 10]
    loud_idx = sorted_idx[-len(sorted_idx) // 10]

    # Show ~3 periods
    period_ms = 1000 / nominal_hz
    show_ms = max(period_ms * 3.5, 10)  # at least 3.5 periods, min 10ms
    show_samples = int(show_ms / 1000 * rate)

    soft_freq = freqs[quiet_idx]
    hard_freq = freqs[loud_idx]
    cents_diff = 1200 * np.log2(hard_freq / soft_freq)

    for ax, idx, label, color in [
        (ax3a, quiet_idx, "Soft Pluck", BLUE),
        (ax3b, loud_idx, "Hard Pluck", RED),
    ]:
        target_amp = amps[idx]
        target_freq = freqs[idx]
        onset = matched_onsets[idx]
        n_skip = int(0.05 * rate)

        seg = audio[onset + n_skip:onset + n_skip + show_samples]
        skip_ms = n_skip / rate * 1000
        t_ms = np.arange(len(seg)) / rate * 1000 + skip_ms

        ax.plot(t_ms, seg, linewidth=0.8, color=color, alpha=0.85)
        ax.fill_between(t_ms, seg, alpha=0.1, color=color)
        ax.set_xlabel("Time after onset (ms)")
        ax.set_title(
            f"{label}:  {target_freq:.1f} Hz,  RMS = {target_amp:.3f}",
            fontsize=11, fontweight="bold", pad=8,
        )
        ax.set_xlim(skip_ms, skip_ms + show_ms)

        # Mark one period between actual peaks
        min_lag = int(0.5 / target_freq * rate)
        peaks, _ = signal.find_peaks(seg, distance=min_lag, height=seg.max() * 0.3)
        if len(peaks) >= 2:
            p1, p2 = peaks[0], peaks[1]
            t1 = p1 / rate * 1000 + skip_ms
            t2 = p2 / rate * 1000 + skip_ms
            y_arrow = max(seg[p1], seg[p2]) * 1.15
            ax.annotate("", xy=(t2, y_arrow), xytext=(t1, y_arrow),
                         arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.5))
            ax.text((t1 + t2) / 2, y_arrow * 1.08, f"T = {t2 - t1:.2f} ms",
                    ha="center", fontsize=9, color=DARK)

    # Give both panels some headroom above the waveform
    for ax in (ax3a, ax3b):
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.4)

    ax3a.set_ylabel("Amplitude")

    fig3.text(0.5, 1.04, f"{string_name} String: Comparing a Soft and Hard Pluck",
              ha="center", fontsize=13, fontweight="bold", transform=fig3.transFigure)
    fig3.text(0.5, 1.00, f"hard pluck is {cents_diff:+.1f} cents vs. soft",
              ha="center", fontsize=10, color=GRAY, transform=fig3.transFigure)
    fig3.tight_layout()
    fig3.savefig(f"{prefix}_fig3_comparison.png", dpi=180, bbox_inches="tight")
    print(f"  Saved {prefix}_fig3_comparison.png")
    plt.close(fig3)

    print(f"\n  --- Huber Regression ({string_name}): quadratic (freq ~ RMS²) ---")
    print(result_quad.summary2())


def analyze_file(wav_path: Path):
    """Run full analysis pipeline on a single wav file."""
    config = guess_string_config(wav_path.name)
    string_name = config["name"]
    nominal_hz = config["nominal_hz"]

    # Output file prefix from the string name
    prefix = string_name.split("(")[1].rstrip(")").replace(" ", "").lower()

    print(f"\n{'='*60}")
    print(f"  {string_name} — {nominal_hz:.1f} Hz")
    print(f"  File: {wav_path.name}")
    print(f"{'='*60}")

    rate, audio = load_wav(str(wav_path))
    print(f"  {rate} Hz, {len(audio)/rate:.1f}s")

    onset_samples, flux, hop_size = detect_onsets(audio, rate)
    print(f"  {len(onset_samples)} onsets detected")

    freqs, amps, matched_onsets = extract_features(audio, rate, onset_samples, nominal_hz)
    print(f"  {len(freqs)} plucks after filtering")

    if len(freqs) < 5:
        print(f"  WARNING: Too few plucks ({len(freqs)}) — skipping plots")
        return

    compare_amp_vs_power_fit(freqs, amps, string_name)
    plot_blog_figures(audio, rate, onset_samples, freqs, amps, matched_onsets,
                      flux, hop_size, string_name, nominal_hz, prefix)

    # Return stats for summary table
    huber = sm.robust.norms.HuberT()
    r_lin  = sm.RLM(freqs, sm.add_constant(amps),    M=huber).fit()
    r_quad = sm.RLM(freqs, sm.add_constant(amps**2), M=huber).fit()
    p95_amp = np.percentile(amps, 95)
    baseline_hz = r_quad.params[0]
    cents_p95 = 1200 * np.log2((baseline_hz + r_quad.params[1] * p95_amp**2) / baseline_hz)
    pval = r_quad.pvalues[1]
    return {
        "name": string_name,
        "nominal_hz": nominal_hz,
        "n": len(freqs),
        "a": r_quad.params[0],
        "b": r_quad.params[1],
        "pval": pval,
        "scale_lin": r_lin.scale,
        "scale_quad": r_quad.scale,
        "p95_amp": p95_amp,
        "cents_p95": cents_p95,
    }


def print_summary_table(rows: list[dict]):
    """Print a markdown summary table of fit results across strings."""
    header = (
        r"| String | $n$ | $\hat{f} = a + b\cdot\text{RMS}^2$ | $p$ | "
        r"95th pct RMS | $\Delta$ cents at 95th pct |"
    )
    sep = "|---|---|---|---|---|---|"
    print("\n\n## Summary\n")
    print(header)
    print(sep)
    for r in rows:
        pval_str = r"$< 0.001$" if r["pval"] < 0.001 else f"${r['pval']:.3f}$"
        print(
            f"| {r['name']} "
            f"| ${r['n']}$ "
            f"| ${r['a']:.2f} + {r['b']:.4f}\\cdot\\text{{RMS}}^2$ "
            f"| {pval_str} "
            f"| ${r['p95_amp']:.3f}$ "
            f"| ${r['cents_p95']:+.1f}$ |"
        )


def main():
    wav_files = sorted(Path("data").glob("*.wav"))
    print(f"Found {len(wav_files)} wav files")
    rows = []
    for wav_path in wav_files:
        result = analyze_file(wav_path)
        if result:
            rows.append(result)
    print_summary_table(rows)


if __name__ == "__main__":
    main()
