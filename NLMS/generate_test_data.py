"""Generate reproducible controlled and held-out NLMS WAV test recordings.

The speech source here is a clearly labelled speech-like surrogate, not a claim
of a human speech recording. Replace these files with recorded two-channel
audio for field evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import signal

from nlms import DEFAULT_SAMPLE_RATE, write_wav


def speech_like_surrogate(sample_rate: int, duration_seconds: float, seed: int) -> np.ndarray:
    """Create a deterministic voiced/unvoiced, syllabic clean-speech surrogate."""
    rng = np.random.default_rng(seed)
    count = round(sample_rate * duration_seconds)
    time_axis = np.arange(count) / sample_rate
    syllable_rate = 2.2 + 0.35 * np.sin(2 * np.pi * 0.13 * time_axis)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * syllable_rate * time_axis - 1.1))
    envelope = np.power(np.maximum(envelope, 0), 1.8)
    formant_envelope = (
        0.58 * np.sin(2 * np.pi * (118 + 9 * np.sin(2 * np.pi * 0.27 * time_axis)) * time_axis)
        + 0.25 * np.sin(2 * np.pi * 242 * time_axis)
        + 0.12 * np.sin(2 * np.pi * 370 * time_axis)
    )
    unvoiced = signal.lfilter([1.0], [1.0, -0.82], rng.normal(0, 0.04, count))
    consonant_mask = np.power(np.maximum(np.sin(2 * np.pi * 3.1 * time_axis + 0.7), 0), 8)
    clean = envelope * formant_envelope + consonant_mask * unvoiced
    clean /= max(np.max(np.abs(clean)), np.finfo(float).eps)
    return 0.18 * clean


def correlated_noise(sample_rate: int, duration_seconds: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a reference noise signal and its differently filtered primary counterpart."""
    rng = np.random.default_rng(seed)
    count = round(sample_rate * duration_seconds)
    white = rng.normal(0, 1, count)
    reference = signal.lfilter([1.0], [1.0, -0.93], white)
    bandpass = signal.butter(4, [80, 5500], btype="bandpass", fs=sample_rate, output="sos")
    reference = signal.sosfilt(bandpass, reference)
    reference /= max(np.std(reference), np.finfo(float).eps)
    acoustic_path = np.array([0.72, -0.29, 0.16, 0.08, -0.04])
    primary_noise = signal.lfilter(acoustic_path, [1.0], reference)
    delay_samples = 9
    primary_noise = np.concatenate((np.zeros(delay_samples), primary_noise[:-delay_samples]))
    primary_noise += rng.normal(0, 0.025, count)
    return reference, primary_noise


def make_case(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_seconds: float = 5.0,
    snr_db: float = 0.0,
    seed: int = 2026,
) -> dict[str, np.ndarray | int]:
    """Build a known clean plus known noise two-channel ANC case."""
    clean = speech_like_surrogate(sample_rate, duration_seconds, seed)
    reference, noise = correlated_noise(sample_rate, duration_seconds, seed + 1)
    desired_noise_power = np.mean(np.square(clean)) / (10 ** (snr_db / 10))
    scale = np.sqrt(desired_noise_power / np.mean(np.square(noise)))
    return {
        "clean": clean,
        "reference": reference * scale,
        "noise": noise * scale,
        "primary": clean + noise * scale,
        "sample_rate": sample_rate,
    }


def write_case(case: dict[str, np.ndarray | int], destination: str | Path) -> None:
    """Persist a controlled case as WAV files while keeping labels for evaluation."""
    destination = Path(destination)
    sample_rate = int(case["sample_rate"])
    write_wav(destination / "primary.wav", np.asarray(case["primary"]), sample_rate)
    write_wav(destination / "reference.wav", np.asarray(case["reference"]), sample_rate)
    write_wav(destination / "clean_speech.wav", np.asarray(case["clean"]), sample_rate)
    write_wav(destination / "known_primary_noise.wav", np.asarray(case["noise"]), sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible controlled NLMS test WAVs.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    controlled = make_case(duration_seconds=args.duration, snr_db=0.0, seed=2026)
    write_case(controlled, args.data_dir)
    held_out = make_case(duration_seconds=args.duration, snr_db=-3.0, seed=98765)
    write_case(held_out, Path(args.data_dir) / "unseen")
    print(f"Controlled data: {Path(args.data_dir).resolve()}")
    print(f"Held-out data:   {(Path(args.data_dir) / 'unseen').resolve()}")


if __name__ == "__main__":
    main()
