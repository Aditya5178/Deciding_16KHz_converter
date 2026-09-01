"""Waveform and spectrogram plots for NLMS output verification."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from nlms import DEFAULT_SAMPLE_RATE, default_input_path, load_audio, match_lengths


def _time_axis(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    return np.arange(len(samples)) / sample_rate


def save_waveforms(
    primary: np.ndarray,
    reference: np.ndarray,
    estimated_noise: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
) -> Path:
    """Save the four requested time-domain signals in one directly comparable figure."""
    signals = (
        ("Primary — speech + noise", primary, "#1f77b4"),
        ("Reference — correlated noise", reference, "#ff7f0e"),
        ("Estimated noise — NLMS output", estimated_noise, "#d62728"),
        ("Enhanced signal — primary − estimated noise", enhanced, "#2ca02c"),
    )
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 10), constrained_layout=True)
    times = _time_axis(primary, sample_rate)
    for axis, (title, samples, color) in zip(axes, signals):
        axis.plot(times, samples, color=color, linewidth=0.55)
        axis.set_title(title)
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (seconds)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _draw_spectrogram(axis: plt.Axes, samples: np.ndarray, sample_rate: int, title: str):
    frequencies, times, spectrum = signal.spectrogram(
        samples, fs=sample_rate, nperseg=512, noverlap=384, scaling="density", mode="psd"
    )
    spectrum_db = 10 * np.log10(spectrum + np.finfo(float).tiny)
    image = axis.pcolormesh(times, frequencies, spectrum_db, shading="auto", cmap="magma")
    axis.set_title(title)
    axis.set_ylabel("Frequency (Hz)")
    axis.set_ylim(0, sample_rate / 2)
    return image


def save_spectrograms(
    primary: np.ndarray,
    estimated_noise: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
) -> Path:
    """Save spectrograms for primary, estimated noise, and enhanced output."""
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 10), constrained_layout=True)
    images = (
        _draw_spectrogram(axes[0], primary, sample_rate, "Primary — speech + noise"),
        _draw_spectrogram(axes[1], estimated_noise, sample_rate, "Estimated noise — NLMS output"),
        _draw_spectrogram(axes[2], enhanced, sample_rate, "Enhanced signal"),
    )
    axes[-1].set_xlabel("Time (seconds)")
    for axis, image in zip(axes, images):
        figure.colorbar(image, ax=axis, pad=0.01, label="Power/frequency (dB/Hz)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_visualizations(
    primary: np.ndarray,
    reference: np.ndarray,
    estimated_noise: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create all plots required to inspect an NLMS run."""
    output_dir = Path(output_dir)
    return {
        "waveforms": save_waveforms(
            primary, reference, estimated_noise, enhanced, sample_rate, output_dir / "waveforms.png"
        ),
        "spectrograms": save_spectrograms(
            primary, estimated_noise, enhanced, sample_rate, output_dir / "spectrograms.png"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate NLMS waveform and spectrogram figures.")
    parser.add_argument("--primary", default=default_input_path("primary.wav"))
    parser.add_argument("--reference", default=default_input_path("reference.wav"))
    parser.add_argument("--estimated-noise", default="outputs/estimated_noise.wav")
    parser.add_argument("--enhanced", default="outputs/nlms_enhanced.wav")
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    args = parser.parse_args()

    primary, sample_rate = load_audio(args.primary, args.sample_rate)
    reference, _ = load_audio(args.reference, args.sample_rate)
    estimated_noise, _ = load_audio(args.estimated_noise, args.sample_rate)
    enhanced, _ = load_audio(args.enhanced, args.sample_rate)
    primary, reference = match_lengths(primary, reference)
    primary, estimated_noise = match_lengths(primary, estimated_noise)
    primary, enhanced = match_lengths(primary, enhanced)
    paths = save_visualizations(primary, reference, estimated_noise, enhanced, sample_rate, args.output_dir)
    print(f"Waveforms:    {paths['waveforms']}")
    print(f"Spectrograms: {paths['spectrograms']}")


if __name__ == "__main__":
    main()
