"""Create an STFT spectrogram from a WAV file.

Usage:
    python stft_fft.py input.wav
    python stft_fft.py input.wav --show
    python stft_fft.py input.wav --window-size 400 --overlap 240 --show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an STFT spectrogram from a WAV file."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input WAV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path "
             "(default: <input-name>_spectrogram.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open the plot window",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=400,
        help="STFT window size in samples (default: 400, "
             "25 ms at 16 kHz)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=240,
        help="Window overlap in samples (default: 240, "
             "15 ms at 16 kHz)",
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        help="Highest frequency to display in Hz "
             "(default: Nyquist frequency)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    if args.window_size < 2:
        parser.error("--window-size must be at least 2")

    if not 0 <= args.overlap < args.window_size:
        parser.error("--overlap must be at least 0 and smaller "
                     "than --window-size")

    audio, sample_rate = sf.read(args.input, always_2d=False)

    # Convert stereo/multi-channel audio to mono.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Explicit Hann window for reproducible STFT processing.
    frequencies, times, spectrum = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=args.window_size,
        noverlap=args.overlap,
    )

    # Convert STFT magnitude to decibels.
    magnitude_db = 20 * np.log10(
        np.maximum(np.abs(spectrum), 1e-10)
    )

    output = args.output or args.input.with_name(
        f"{args.input.stem}_spectrogram.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    window_ms = 1000 * args.window_size / sample_rate
    hop_size = args.window_size - args.overlap
    hop_ms = 1000 * hop_size / sample_rate
    max_frequency = args.max_frequency or sample_rate / 2

    figure, axis = plt.subplots(
        figsize=(12, 6),
        constrained_layout=True,
    )

    plot = axis.pcolormesh(
        times,
        frequencies,
        magnitude_db,
        shading="gouraud",
        cmap="magma",
    )

    axis.set_title(
        f"STFT Spectrogram: {args.input.name}\n"
        f"SR={sample_rate} Hz | "
        f"Window={window_ms:.1f} ms | "
        f"Hop={hop_ms:.1f} ms | Hann"
    )
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_ylim(0, min(max_frequency, sample_rate / 2))

    figure.colorbar(
        plot,
        ax=axis,
        label="Magnitude (dB)",
    )

    figure.savefig(output, dpi=200)

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Window: {args.window_size} samples ({window_ms:.2f} ms)")
    print(f"Overlap: {args.overlap} samples")
    print(f"Hop: {hop_size} samples ({hop_ms:.2f} ms)")
    print("Window function: Hann")
    print(f"Spectrogram saved to: {output.resolve()}")

    if args.show:
        plt.show()

    plt.close(figure)


if __name__ == "__main__":
    main()
