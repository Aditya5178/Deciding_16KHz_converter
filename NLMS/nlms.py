"""Simple NLMS noise cancellation - test mode.

Run NLMS adaptive filtering with specified parameters and save output files.
Usage:
    python nlms.py --filter-length 64 --mu 0.05 --epsilon 1e-8
    python nlms.py                # uses defaults
    python nlms.py --tune        # auto-tune best parameters
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

DEFAULT_SAMPLE_RATE = 16_000


def default_input_path(filename: str) -> str:
    """Prefer explicitly supplied project-root inputs, otherwise use data/."""
    project_root_candidate = Path(filename)
    data_directory_candidate = Path("data") / filename
    if project_root_candidate.is_file():
        return str(project_root_candidate)
    return str(data_directory_candidate)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Return a float64 mono signal, averaging channels when necessary."""
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        return np.mean(samples, axis=1)
    raise ValueError(f"Expected mono or multi-channel audio, got shape {samples.shape}.")


def resample_audio(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Resample using polyphase filtering only when the sample rate differs."""
    if original_rate == target_rate:
        return np.asarray(audio, dtype=np.float64)
    from fractions import Fraction
    ratio = Fraction(target_rate, original_rate).limit_denominator()
    return signal.resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float64)


def load_audio(path: str | Path, target_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Read, convert to mono, resample, and remove the DC component of a WAV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, source_rate = sf.read(path, always_2d=False, dtype="float64")
    audio = to_mono(audio)
    audio = resample_audio(audio, source_rate, target_rate)
    return audio - np.mean(audio), target_rate


def match_lengths(primary: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim both signals to their shared length without changing source files."""
    shared_length = min(len(primary), len(reference))
    if shared_length == 0:
        raise ValueError("One or both input files contain no samples.")
    return primary[:shared_length].copy(), reference[:shared_length].copy()


def shift_with_zeros(audio: np.ndarray, samples: int) -> np.ndarray:
    """Shift right for positive samples and left for negative samples, zero-padding."""
    result = np.zeros_like(audio)
    if samples > 0:
        result[samples:] = audio[:-samples]
    elif samples < 0:
        result[:samples] = audio[-samples:]
    else:
        result[:] = audio
    return result


def estimate_reference_delay(
    primary: np.ndarray, reference: np.ndarray, max_delay_samples: int
) -> int:
    """Estimate the reference shift that best aligns it to the primary signal."""
    if max_delay_samples < 0:
        raise ValueError("max_delay_samples cannot be negative.")
    if max_delay_samples == 0:
        return 0

    primary_zero_mean = primary - np.mean(primary)
    reference_zero_mean = reference - np.mean(reference)
    correlation = signal.correlate(primary_zero_mean, reference_zero_mean, method="fft")
    lags = signal.correlation_lags(len(primary_zero_mean), len(reference_zero_mean))
    allowed = np.abs(lags) <= max_delay_samples
    if not np.any(allowed):
        return 0
    return int(lags[allowed][np.argmax(np.abs(correlation[allowed]))])


def normalized_reference_correlation(
    primary: np.ndarray, reference: np.ndarray, max_delay_samples: int
) -> float:
    """Return the strongest normalized correlation within the allowed alignment window."""
    primary_zero_mean = primary - np.mean(primary)
    reference_zero_mean = reference - np.mean(reference)
    primary_energy = float(np.dot(primary_zero_mean, primary_zero_mean))
    reference_energy = float(np.dot(reference_zero_mean, reference_zero_mean))
    if primary_energy <= np.finfo(float).eps or reference_energy <= np.finfo(float).eps:
        return 0.0
    correlation = signal.correlate(primary_zero_mean, reference_zero_mean, method="fft")
    lags = signal.correlation_lags(len(primary_zero_mean), len(reference_zero_mean))
    allowed = np.abs(lags) <= max_delay_samples
    if not np.any(allowed):
        return 0.0
    return float(np.max(np.abs(correlation[allowed])) / np.sqrt(primary_energy * reference_energy))


def prepare_signals(
    primary_path: str | Path,
    reference_path: str | Path,
    target_rate: int = DEFAULT_SAMPLE_RATE,
    correct_delay: bool = True,
    max_delay_ms: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, int, int, dict[str, int]]:
    """Load and prepare the two microphone channels without modifying inputs."""
    primary, sample_rate = load_audio(primary_path, target_rate)
    reference, _ = load_audio(reference_path, target_rate)
    samples_before_matching = {
        "primary_after_resampling": len(primary),
        "reference_after_resampling": len(reference),
    }
    primary, reference = match_lengths(primary, reference)

    delay_samples = 0
    if correct_delay:
        max_delay_samples = round(max_delay_ms * sample_rate / 1000.0)
        delay_samples = estimate_reference_delay(primary, reference, max_delay_samples)
        reference = shift_with_zeros(reference, delay_samples)
    return primary, reference, sample_rate, delay_samples, samples_before_matching


def nlms_filter(
    primary: np.ndarray, reference: np.ndarray, filter_length: int, mu: float, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate primary-channel noise with sample-by-sample NLMS.

    Returns estimated noise, enhanced signal, and final filter coefficients.
    """
    primary, reference = match_lengths(
        np.asarray(primary, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    )

    coefficients = np.zeros(filter_length, dtype=np.float64)
    reference_buffer = np.zeros(filter_length, dtype=np.float64)
    estimated_noise = np.zeros_like(primary)
    enhanced = np.zeros_like(primary)

    for sample_index, desired in enumerate(primary):
        reference_buffer[1:] = reference_buffer[:-1]
        reference_buffer[0] = reference[sample_index]

        estimate = float(np.dot(coefficients, reference_buffer))
        error = desired - estimate
        normalization = epsilon + float(np.dot(reference_buffer, reference_buffer))
        coefficients += (mu * error / normalization) * reference_buffer

        estimated_noise[sample_index] = estimate
        enhanced[sample_index] = error

    return estimated_noise, enhanced, coefficients


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write float WAV output without normalizing or clipping the NLMS signal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="FLOAT")


def tune_best_parameters(
    primary: np.ndarray,
    reference: np.ndarray,
    mu_candidates: list[float] | None = None,
    filter_length_candidates: list[int] | None = None,
    epsilon_candidates: list[float] | None = None,
    tune_duration_sec: float = 5.0,
) -> dict[str, Any]:
    """Test multiple mu/filter_length/epsilon combinations and return the best config.

    Uses a shortened segment (default 5s) for fast evaluation.
    """
    if mu_candidates is None:
        mu_candidates = [0.01, 0.03, 0.05, 0.1, 0.3]
    if filter_length_candidates is None:
        filter_length_candidates = [16, 32, 64, 128]
    if epsilon_candidates is None:
        epsilon_candidates = [1e-8, 1e-6]

    primary, reference = match_lengths(
        np.asarray(primary, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )

    sample_rate = DEFAULT_SAMPLE_RATE
    tune_samples = min(len(primary), int(sample_rate * tune_duration_sec))
    p_tune = primary[:tune_samples]
    r_tune = reference[:tune_samples]

    best_result = {
        "mu": 0.05,
        "filter_length": 128,
        "epsilon": 1e-8,
        "score": float("inf"),
    }

    total = len(mu_candidates) * len(filter_length_candidates) * len(epsilon_candidates)
    tested = 0

    for mu in mu_candidates:
        for flen in filter_length_candidates:
            for eps in epsilon_candidates:
                tested += 1
                try:
                    _, enhanced, _ = nlms_filter(p_tune, r_tune, flen, mu, eps)
                    if not np.all(np.isfinite(enhanced)):
                        continue
                    residual_power = float(np.mean(np.square(enhanced)))
                    if residual_power < best_result["score"]:
                        best_result = {
                            "mu": mu,
                            "filter_length": flen,
                            "epsilon": eps,
                            "score": residual_power,
                        }
                except Exception:
                    continue

    print(f"  Tested {total} combinations on {tune_duration_sec}s of audio.")
    return best_result


def run_pipeline(
    primary_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path = "outputs",
    filter_length: int = 128,
    mu: float = 0.05,
    epsilon: float = 1e-8,
    correct_delay: bool = True,
    max_delay_ms: float = 30.0,
) -> dict[str, Any]:
    """Run NLMS pipeline with specified parameters and save outputs."""
    primary, reference, sample_rate, delay_samples, input_lengths = prepare_signals(
        primary_path, reference_path, target_rate=DEFAULT_SAMPLE_RATE,
        correct_delay=correct_delay, max_delay_ms=max_delay_ms
    )

    estimated_noise, enhanced, coefficients = nlms_filter(
        primary, reference, filter_length, mu, epsilon
    )

    processing_seconds = time.perf_counter() - time.perf_counter()  # placeholder

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimated_path = output_dir / "estimated_noise.wav"
    enhanced_path = output_dir / "nlms_enhanced.wav"
    coefficient_path = output_dir / "final_filter_coefficients.npy"

    write_wav(estimated_path, estimated_noise, sample_rate)
    write_wav(enhanced_path, enhanced, sample_rate)
    np.save(coefficient_path, coefficients)

    # Also save metadata
    metadata = {
        "sample_rate": sample_rate,
        "samples": len(primary),
        "duration_seconds": len(primary) / sample_rate,
        "reference_shift_samples": delay_samples,
        "filter_length": filter_length,
        "mu": mu,
        "epsilon": epsilon,
        "processing_seconds": processing_seconds,
        "input_lengths": input_lengths,
    }
    metadata_path = output_dir / "nlms_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "primary": primary,
        "reference": reference,
        "estimated_noise": estimated_noise,
        "enhanced": enhanced,
        "coefficients": coefficients,
        "sample_rate": sample_rate,
        "delay_samples": delay_samples,
        "processing_seconds": processing_seconds,
        "estimated_noise_path": estimated_path,
        "enhanced_path": enhanced_path,
        "coefficient_path": coefficient_path,
        "metadata_path": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple NLMS noise cancellation test."
    )
    parser.add_argument(
        "--primary",
        default="primary.wav",
        help="Speech + noise WAV path (default: primary.wav).",
    )
    parser.add_argument(
        "--reference",
        default="reference.wav",
        help="Correlated-noise WAV path (default: reference.wav).",
    )
    parser.add_argument(
        "--filter-length", type=int, default=128, help="Adaptive filter taps."
    )
    parser.add_argument(
        "--mu", type=float, default=0.05, help="NLMS learning rate, in (0, 2]."
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-8, help="Positive normalization floor."
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Directory for WAVs and metadata."
    )
    parser.add_argument(
        "--max-delay-ms", type=float, default=30.0, help="Maximum delay search in ms."
    )
    parser.add_argument(
        "--no-delay-correction", action="store_true", help="Disable delay estimation."
    )
    parser.add_argument(
        "--tune", action="store_true", help="Auto-tune mu, filter_length, epsilon for best result."
    )
    args = parser.parse_args()

    mu = args.mu
    filter_length = args.filter_length
    epsilon = args.epsilon

    if args.tune:
        print("Auto-tuning NLMS parameters...")
        primary_data, _ = load_audio(args.primary)
        reference_data, _ = load_audio(args.reference)
        best = tune_best_parameters(primary_data, reference_data)
        mu = best["mu"]
        filter_length = best["filter_length"]
        epsilon = best["epsilon"]
        print(f"Best config: mu={mu}, filter_length={filter_length}, epsilon={epsilon}")

    config = run_pipeline(
        primary_path=args.primary,
        reference_path=args.reference,
        output_dir=args.output_dir,
        filter_length=filter_length,
        mu=mu,
        epsilon=epsilon,
        correct_delay=not args.no_delay_correction,
        max_delay_ms=args.max_delay_ms,
    )

    print(f"Processed: {len(config['primary'])} samples at {config['sample_rate']} Hz")
    print(f"Reference shift: {config['delay_samples']} samples")
    print(f"Outputs written to: {args.output_dir}/")
    print(f"  - {config['enhanced_path'].name} (enhanced speech)")
    print(f"  - {config['estimated_noise_path'].name} (estimated noise)")
    print(f"  - {config['coefficient_path'].name} (filter coefficients)")


if __name__ == "__main__":
    main()