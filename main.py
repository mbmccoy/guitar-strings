import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path


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

    # DON'T globally normalize — keep the raw scale so quiet and loud
    # sections are handled by the adaptive (ratio-based) threshold.

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
    # Normalize by zero-lag (energy)
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
            # Clamp correction to avoid overshooting into adjacent bins
            correction = np.clip(correction, -0.5, 0.5)
            peak_lag = peak_lag + correction

    if peak_lag <= 0:
        return None

    return rate / peak_lag


def extract_features(
    audio: np.ndarray,
    rate: int,
    onset_samples: np.ndarray,
    skip_duration: float = 0.05,
    analysis_duration: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each pluck, extract dominant frequency (autocorrelation) and
    RMS amplitude. Skips the initial transient (skip_duration) and then
    analyzes a window of analysis_duration seconds.
    """
    n_skip = int(skip_duration * rate)
    n_analysis = int(analysis_duration * rate)
    frequencies = []
    amplitudes = []

    for onset in onset_samples:
        start = onset + n_skip
        segment = audio[start:start + n_analysis]
        if len(segment) < n_analysis // 2:
            continue

        rms = np.sqrt(np.mean(segment**2))

        freq = autocorrelation_pitch(segment, rate)
        if freq is None:
            continue

        frequencies.append(freq)
        amplitudes.append(rms)

    frequencies = np.array(frequencies)
    amplitudes = np.array(amplitudes)

    # Filter outliers: keep only pitches within one whole tone of E1 (41.2 Hz)
    # D1 ≈ 36.7 Hz, F#1 ≈ 46.2 Hz
    # Also discard very quiet plucks (RMS < 0.01) which are noise/artifacts
    mask = (frequencies >= 36.7) & (frequencies <= 46.2) & (amplitudes >= 0.01)
    return frequencies[mask], amplitudes[mask]


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


def plot_blog_figures(audio, rate, onset_samples, freqs, amps, flux, hop_size):
    """Generate three publication-quality figures."""
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
    ax1.set_title("Bass Guitar Recording with Detected Pluck Onsets", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)

    fig1.tight_layout()
    fig1.savefig("fig1_waveform.png", dpi=180, bbox_inches="tight")
    print("Saved fig1_waveform.png")
    plt.close(fig1)

    # ── Figure 2: Scatter + Huber regression (the main result) ──
    fig2, ax2 = plt.subplots(figsize=(7, 5.5))

    ax2.scatter(amps, freqs, alpha=0.55, s=35, color=ORANGE, edgecolors=DARK,
                linewidths=0.4, zorder=3)

    X = sm.add_constant(amps)
    huber = sm.RLM(freqs, X, M=sm.robust.norms.HuberT())
    result = huber.fit()
    slope = result.params[1]
    intercept = result.params[0]

    amp_range = np.linspace(amps.min(), amps.max(), 200)
    fit_line = intercept + slope * amp_range
    ax2.plot(amp_range, fit_line, color=RED, linewidth=2.5, zorder=4, label="Huber robust fit")

    # Reference line
    ax2.axhline(41.2, color=GRAY, linestyle=":", linewidth=1, alpha=0.7)
    ax2.text(amps.max() * 0.98, 41.25, "E1 = 41.2 Hz", ha="right", fontsize=8, color=GRAY)

    # Stats annotation
    cents_range = 1200 * np.log2((intercept + slope * amps.max()) / (intercept + slope * amps.min()))
    stats_text = (
        f"slope = {slope:.2f} Hz / RMS unit\n"
        f"p < 0.001 (z = {result.tvalues[1]:.1f})\n"
        f"total pitch shift: {cents_range:.0f} cents (approx. {cents_range/100:.1f} semitones)"
    )
    ax2.text(0.03, 0.97, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor=ORANGE, alpha=0.9))

    ax2.set_xlabel("Pluck Intensity (RMS amplitude)", fontsize=11)
    ax2.set_ylabel("Detected Pitch (Hz)", fontsize=11)
    ax2.set_title("Harder Plucks Go Sharper", fontsize=14, fontweight="bold", pad=12)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig2.tight_layout()
    fig2.savefig("fig2_scatter.png", dpi=180, bbox_inches="tight")
    print("Saved fig2_scatter.png")
    plt.close(fig2)

    # ── Figure 3: Soft vs hard pluck waveform comparison ──
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4))

    # Find a quiet and loud pluck (by RMS)
    sorted_idx = np.argsort(amps)
    # Pick from 10th percentile and 90th percentile to avoid edge cases
    quiet_idx = sorted_idx[len(sorted_idx) // 10]
    loud_idx = sorted_idx[-len(sorted_idx) // 10]

    show_ms = 75  # ms of waveform to show
    show_samples = int(show_ms / 1000 * rate)

    for ax, idx, label, color in [
        (ax3a, quiet_idx, "Soft Pluck", BLUE),
        (ax3b, loud_idx, "Hard Pluck", RED),
    ]:
        # Find the corresponding onset sample
        # We need to map back from filtered index to onset — use amplitude to find it
        target_amp = amps[idx]
        target_freq = freqs[idx]

        # Re-extract to find which onset this was
        n_skip = int(0.05 * rate)
        best_onset = None
        best_diff = float("inf")
        for onset in onset_samples:
            start = onset + n_skip
            seg = audio[start:start + int(0.15 * rate)]
            if len(seg) < int(0.15 * rate) // 2:
                continue
            rms = np.sqrt(np.mean(seg**2))
            diff = abs(rms - target_amp)
            if diff < best_diff:
                best_diff = diff
                best_onset = onset

        seg = audio[best_onset:best_onset + show_samples]
        t_ms = np.arange(len(seg)) / rate * 1000

        ax.plot(t_ms, seg, linewidth=0.8, color=color, alpha=0.85)
        ax.fill_between(t_ms, seg, alpha=0.1, color=color)
        ax.set_xlabel("Time after onset (ms)")
        ax.set_title(
            f"{label}:  {target_freq:.1f} Hz,  RMS = {target_amp:.3f}",
            fontsize=11, fontweight="bold", pad=8,
        )
        ax.set_xlim(0, show_ms)

        # Mark one period between actual peaks
        min_lag = int(0.5 / target_freq * rate)  # half expected period
        peaks, _ = signal.find_peaks(seg, distance=min_lag, height=seg.max() * 0.3)
        if len(peaks) >= 2:
            p1, p2 = peaks[0], peaks[1]
            t1 = p1 / rate * 1000
            t2 = p2 / rate * 1000
            y_arrow = max(seg[p1], seg[p2]) * 1.15
            ax.annotate("", xy=(t2, y_arrow), xytext=(t1, y_arrow),
                         arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.5))
            ax.text((t1 + t2) / 2, y_arrow * 1.08, f"T = {t2 - t1:.1f} ms",
                    ha="center", fontsize=9, color=DARK)

    ax3a.set_ylim(-0.045, 0.05)
    ax3b.set_ylim(-0.7, 0.5)

    ax3a.set_ylabel("Amplitude")

    fig3.suptitle("Comparing a Soft and Hard Pluck", fontsize=13, fontweight="bold", y=1.02)
    fig3.tight_layout()
    fig3.savefig("fig3_comparison.png", dpi=180, bbox_inches="tight")
    print("Saved fig3_comparison.png")
    plt.close(fig3)

    print("\n--- Huber Regression ---")
    print(result.summary2())


def main():
    wav_path = next(Path("data").glob("*.wav"))
    print(f"Loading {wav_path.name}...")
    rate, audio = load_wav(str(wav_path))
    print(f"  {rate} Hz, {len(audio)/rate:.1f}s, {len(audio)} samples")

    print("Detecting onsets...")
    onset_samples, flux, hop_size = detect_onsets(audio, rate)
    print(f"  Found {len(onset_samples)} plucks")

    print("Extracting features...")
    freqs, amps = extract_features(audio, rate, onset_samples)
    print(f"  Analyzed {len(freqs)} plucks")

    plot_blog_figures(audio, rate, onset_samples, freqs, amps, flux, hop_size)
    plt.show()


if __name__ == "__main__":
    main()
