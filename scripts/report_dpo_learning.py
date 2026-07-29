#!/usr/bin/env python
"""Summarize DPO train versus validation preference accuracy by epoch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen3-0.6b-all-datasets-dpo-2epochs"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = args.output_dir / "trainer_state.json"
    if not state_path.exists():
        checkpoints = sorted(
            args.output_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        if not checkpoints:
            raise SystemExit(f"No trainer state found below {args.output_dir}")
        state_path = checkpoints[-1] / "trainer_state.json"

    history = json.loads(state_path.read_text(encoding="utf-8"))["log_history"]
    train_by_epoch: dict[int, list[float]] = {}
    validation: list[dict] = []

    for row in history:
        if "rewards/accuracies" in row:
            epoch = max(1, math.ceil(float(row["epoch"]) - 1e-9))
            train_by_epoch.setdefault(epoch, []).append(float(row["rewards/accuracies"]))
        if "eval_rewards/accuracies" in row:
            validation.append(
                {
                    "epoch": round(float(row.get("epoch", 0.0)), 6),
                    "validation_accuracy": float(row["eval_rewards/accuracies"]),
                    "validation_loss": float(row["eval_loss"]),
                    "step": int(row["step"]),
                }
            )

    train_rows = [
        {
            "epoch": epoch,
            "training_accuracy": sum(values) / len(values),
            "training_batches": len(values),
        }
        for epoch, values in sorted(train_by_epoch.items())
    ]
    train_frame = pd.DataFrame(train_rows)
    validation_frame = pd.DataFrame(validation)
    if validation_frame.empty:
        raise SystemExit("Training has not produced a validation result yet.")

    report = validation_frame.merge(train_frame, on="epoch", how="left")
    report["generalization_gap"] = (
        report["training_accuracy"] - report["validation_accuracy"]
    )
    report_path = args.output_dir / "train_vs_validation_accuracy.csv"
    report.to_csv(report_path, index=False)

    axis = report.plot(
        x="epoch",
        y=["training_accuracy", "validation_accuracy"],
        marker="o",
        ylim=(0, 1.02),
        title="DPO preference accuracy: training vs validation",
    )
    axis.set_ylabel("Preference accuracy")
    axis.grid(alpha=0.25)
    figure_path = args.output_dir / "train_vs_validation_accuracy.png"
    axis.figure.tight_layout()
    axis.figure.savefig(figure_path, dpi=160)
    plt.close(axis.figure)

    display = report.copy()
    for column in ("training_accuracy", "validation_accuracy", "generalization_gap"):
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.1%}"
        )
    print(display.to_string(index=False))
    print(f"\nCSV: {report_path}")
    print(f"Plot: {figure_path}")


if __name__ == "__main__":
    main()
