"""lms.py
Plain (non-normalized) LMS adaptive noise canceller - the classical baseline.

WHY THIS EXISTS
---------------
The SIH brief explicitly contrasts the AI system against "classical LMS-based
ANC". The team already has a strong NLMS implementation; this adds the plain LMS
so the report can show the full progression:

        LMS (fragile)  ->  NLMS (robust)  ->  AI model (learns non-linear noise)

It deliberately REUSES nlms.py for loading, delay alignment, correlation guard,
writing and evaluation, so LMS vs NLMS is a true apples-to-apples comparison -
only the weight-update rule differs.

LMS update:   w(n+1) = w(n) + mu * e(n) * x(n)
NLMS update:  w(n+1) = w(n) + mu * e(n) * x(n) / (eps + ||x(n)||^2)

The plain LMS step size is NOT normalized, so a good `mu` depends on the input
power - that is exactly the weakness that motivates NLMS (and why loud impulsive
noise like gunshots destabilises LMS).

Run:
    python lms.py --primary data/primary.wav --reference data/reference.wav \
                  --clean data/clean_speech.wav
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

# Reuse everything already built and tested in nlms.py (same folder).
from nlms import (
    DEFAULT_SAMPLE_RATE,
    default_input_path,
    match_lengths,
    prepare_signals,
    write_wav,
)


@dataclass(frozen=True)
class LMSConfig:
    """Parameters for a sample-by-sample plain LMS filter."""

    filter_length: int = 128
    mu: float = 1e-3          # NOT normalized -> keep small; power-dependent

    def validate(self) -> None:
        if self.filter_length < 1:
            raise ValueError("filter_length must be at least 1.")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive.")


def lms_filter(primary: np.ndarray, reference: np.ndarray,
               config: LMSConfig | None = None):
    """Estimate primary-channel noise with plain LMS.

    Returns (estimated_noise, enhanced, coefficients), matching nlms_filter's
    return contract so the two are drop-in comparable.
    """
    config = config or LMSConfig()
    config.validate()
    primary, reference = match_lengths(
        np.asarray(primary, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    )

    w = np.zeros(config.filter_length, dtype=np.float64)
    x_buf = np.zeros(config.filter_length, dtype=np.float64)
    estimated_noise = np.zeros_like(primary)
    enhanced = np.zeros_like(primary)

    for i, desired in enumerate(primary):
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = reference[i]
        estimate = float(np.dot(w, x_buf))
        error = desired - estimate
        w += config.mu * error * x_buf          # <-- the only difference vs NLMS
        estimated_noise[i] = estimate
        enhanced[i] = error

    return estimated_noise, enhanced, w


def main() -> None:
    ap = argparse.ArgumentParser(description="Plain LMS adaptive noise cancellation (baseline).")
    ap.add_argument("--primary", default=default_input_path("primary.wav"))
    ap.add_argument("--reference", default=default_input_path("reference.wav"))
    ap.add_argument("--output-dir", default="outputs_lms")
    ap.add_argument("--clean", help="Optional clean-speech WAV for evaluation.")
    ap.add_argument("--filter-length", type=int, default=128)
    ap.add_argument("--mu", type=float, default=1e-3)
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    ap.add_argument("--no-delay-correction", action="store_true")
    args = ap.parse_args()

    from pathlib import Path
    primary, reference, sr, delay, _ = prepare_signals(
        args.primary, args.reference, args.sample_rate,
        correct_delay=not args.no_delay_correction,
    )
    _, enhanced, _ = lms_filter(primary, reference, LMSConfig(args.filter_length, args.mu))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_path = out_dir / "lms_enhanced.wav"
    write_wav(enhanced_path, enhanced, sr)
    print(f"Reference shift: {delay} samples")
    print(f"Enhanced speech: {enhanced_path}")

    if args.clean:
        from evaluate import evaluate_from_paths, format_metrics
        metrics = evaluate_from_paths(args.primary, enhanced_path, args.clean, sr)
        print()
        print("LMS baseline metrics:")
        print(format_metrics(metrics))


if __name__ == "__main__":
    main()
