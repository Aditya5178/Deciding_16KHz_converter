"""Convert WAV or MP3 audio files to 16 kHz mono PCM WAV.

Usage:
    python resample_to_16k.py input.wav output.wav
    python resample_to_16k.py input_folder output_folder
"""

import argparse
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16_000
SUPPORTED_INPUT_SUFFIXES = {".wav", ".mp3"}


def convert_file(input_path: Path, output_path: Path) -> None:
    """Convert one WAV file to 16 kHz, mono, 16-bit PCM."""
    audio, sample_rate = sf.read(input_path, always_2d=False)

    # Convert stereo or multi-channel audio to mono.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    print(
        f"{input_path.name}: {sample_rate} Hz -> {TARGET_SAMPLE_RATE} Hz | "
        f"saved to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WAV or MP3 files to 16 kHz mono WAV."
    )
    parser.add_argument("input", type=Path, help="Input WAV/MP3 file or folder")
    parser.add_argument("output", type=Path, help="Output WAV file or folder")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")

    if args.input.is_file():
        if args.input.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            parser.error("Input file must be a WAV or MP3 file.")
        convert_file(args.input, args.output)
        return

    if args.output.resolve().is_relative_to(args.input.resolve()):
        parser.error("Output folder must not be inside the input folder.")

    audio_files = sorted(
        path
        for path in args.input.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )
    if not audio_files:
        parser.error("No WAV or MP3 files found in the input folder.")

    for input_file in audio_files:
        relative_output = input_file.relative_to(args.input).with_suffix(".wav")
        convert_file(input_file, args.output / relative_output)

    print(f"\nConverted {len(audio_files)} audio file(s).")
    print(f"Target sample rate: {TARGET_SAMPLE_RATE} Hz")


if __name__ == "__main__":
    main()
