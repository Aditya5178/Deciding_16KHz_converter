"""Objective metrics for runs where a clean speech reference is available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from nlms import DEFAULT_SAMPLE_RATE, default_input_path, load_audio


def _shared_length(*signals: np.ndarray) -> tuple[np.ndarray, ...]:
    length = min(len(samples) for samples in signals)
    if length == 0:
        raise ValueError("Cannot evaluate empty signals.")
    return tuple(samples[:length] for samples in signals)


def snr_db(clean: np.ndarray, residual_noise: np.ndarray) -> float:
    """Calculate energy SNR in dB, returning infinity for a zero residual."""
    clean_power = float(np.mean(np.square(clean)))
    noise_power = float(np.mean(np.square(residual_noise)))
    if clean_power == 0:
        return float("nan")
    if noise_power == 0:
        return float("inf")
    return 10.0 * np.log10(clean_power / noise_power)


def si_sdr_db(estimate: np.ndarray, target: np.ndarray) -> float:
    """Scale-invariant SDR, implemented directly to avoid a metrics dependency."""
    estimate, target = _shared_length(estimate, target)
    target = target - np.mean(target)
    estimate = estimate - np.mean(estimate)
    target_energy = float(np.dot(target, target))
    if target_energy <= np.finfo(float).eps:
        return float("nan")
    projected_target = np.dot(estimate, target) * target / target_energy
    distortion = estimate - projected_target
    distortion_energy = float(np.dot(distortion, distortion))
    if distortion_energy <= np.finfo(float).eps:
        return float("inf")
    return 10.0 * np.log10(float(np.dot(projected_target, projected_target)) / distortion_energy)


def stoi_score(clean: np.ndarray, enhanced: np.ndarray, sample_rate: int) -> float | None:
    """Compute STOI when the optional pystoi package is installed."""
    try:
        from pystoi import stoi
    except ImportError:
        return None
    clean, enhanced = _shared_length(clean, enhanced)
    return float(stoi(clean, enhanced, sample_rate, extended=False))


def evaluate_signals(
    primary: np.ndarray, enhanced: np.ndarray, clean: np.ndarray, sample_rate: int
) -> dict[str, float | None]:
    """Calculate input/output SNR, improvement, SI-SDR, and optional STOI."""
    primary, enhanced, clean = _shared_length(primary, enhanced, clean)
    input_snr = snr_db(clean, primary - clean)
    output_snr = snr_db(clean, enhanced - clean)
    return {
        "input_snr_db": input_snr,
        "output_snr_db": output_snr,
        "snr_improvement_db": output_snr - input_snr,
        "input_si_sdr_db": si_sdr_db(primary, clean),
        "output_si_sdr_db": si_sdr_db(enhanced, clean),
        "stoi": stoi_score(clean, enhanced, sample_rate),
    }


def evaluate_from_paths(
    primary_path: str | Path,
    enhanced_path: str | Path,
    clean_path: str | Path,
    target_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict[str, float | None]:
    primary, sample_rate = load_audio(primary_path, target_rate)
    enhanced, _ = load_audio(enhanced_path, target_rate)
    clean, _ = load_audio(clean_path, target_rate)
    return evaluate_signals(primary, enhanced, clean, sample_rate)


def _format_value(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "N/A (install pystoi for STOI)"
    if np.isnan(value):
        return "N/A"
    if np.isinf(value):
        return f"inf{suffix}"
    return f"{value:.2f}{suffix}"


def format_metrics(metrics: dict[str, Any]) -> str:
    """Format result text without fabricating unavailable measurements."""
    lines = [
        f"Input SNR:       {_format_value(metrics.get('input_snr_db'), ' dB')}",
        f"Output SNR:      {_format_value(metrics.get('output_snr_db'), ' dB')}",
        f"Improvement:     {_format_value(metrics.get('snr_improvement_db'), ' dB')}",
        f"Input SI-SDR:    {_format_value(metrics.get('input_si_sdr_db'), ' dB')}",
        f"Output SI-SDR:   {_format_value(metrics.get('output_si_sdr_db'), ' dB')}",
        f"STOI:            {_format_value(metrics.get('stoi'))}",
    ]
    if "processing_seconds" in metrics and metrics["processing_seconds"] is not None:
        lines.append(f"Processing time: {_format_value(metrics['processing_seconds'], ' seconds')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NLMS output against clean speech.")
    parser.add_argument("--primary", default=default_input_path("primary.wav"))
    parser.add_argument("--enhanced", default="outputs/nlms_enhanced.wav")
    parser.add_argument("--clean", required=True, help="Clean-speech WAV aligned to primary.wav.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--metadata", default="outputs/nlms_metadata.json")
    parser.add_argument("--save-json", default="outputs/evaluation.json")
    args = parser.parse_args()

    metrics = evaluate_from_paths(args.primary, args.enhanced, args.clean, args.sample_rate)
    metadata_path = Path(args.metadata)
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics["processing_seconds"] = metadata.get("processing_seconds")
    print(format_metrics(metrics))

    output_path = Path(args.save_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics JSON:    {output_path}")


if __name__ == "__main__":
    main()
