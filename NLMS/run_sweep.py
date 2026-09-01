"""Run the same NLMS core across noise levels, filter lengths, and mu values."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluate import evaluate_signals
from generate_test_data import make_case
from nlms import NLMSConfig, nlms_filter


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled NLMS parameter sweep.")
    parser.add_argument("--output", default="outputs/parameter_sweep.csv")
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    rows: list[dict[str, float | int]] = []
    for noise_snr_db in (-5.0, 0.0, 5.0):
        case = make_case(duration_seconds=args.duration, snr_db=noise_snr_db, seed=4000 + int(noise_snr_db))
        primary = np.asarray(case["primary"])
        reference = np.asarray(case["reference"])
        clean = np.asarray(case["clean"])
        for filter_length in (32, 64, 128):
            for mu in (0.1, 0.3, 0.6):
                _, enhanced, _ = nlms_filter(primary, reference, NLMSConfig(filter_length, mu))
                metrics = evaluate_signals(primary, enhanced, clean, int(case["sample_rate"]))
                rows.append(
                    {
                        "input_snr_db": noise_snr_db,
                        "filter_length": filter_length,
                        "mu": mu,
                        "output_snr_db": float(metrics["output_snr_db"]),
                        "improvement_db": float(metrics["snr_improvement_db"]),
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} experiments: {output_path}")
    for row in rows:
        print(
            "Input SNR={input_snr_db:>4.1f} dB | taps={filter_length:>3} | "
            "mu={mu:.1f} | improvement={improvement_db:>6.2f} dB".format(**row)
        )


if __name__ == "__main__":
    main()
