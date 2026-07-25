#!/usr/bin/env python3
"""Validate and deterministically split the CoT-controllability DPO data."""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path


THINK_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)\s*$", re.DOTALL)
NORMALIZED_INSTRUCTION = (
    "Think step-by-step. Format only your reasoning according to this rule: "
    "your reasoning must be in English and in ALL CAPITAL LETTERS. "
    "The final answer does not need to be in capital letters."
)


def split_response(text: str) -> tuple[str, str]:
    match = THINK_RE.match(text)
    if not match:
        raise ValueError("response must contain one leading <think>...</think> block")
    return match.group(1), match.group(2)


def uppercase_ratio(text: str) -> float:
    # Ignore uncased scripts (for example Chinese); they are alphabetic but
    # cannot satisfy or violate an uppercase instruction.
    letters = [
        char
        for char in text
        if char.isalpha() and char.lower() != char.upper()
    ]
    if not letters:
        return 1.0
    return sum(char.isupper() for char in letters) / len(letters)


def validate(
    records: list[dict], min_chosen_uppercase: float, max_rejected_uppercase: float
) -> list[str]:
    errors: list[str] = []
    required = {"instruction", "input", "chosen", "rejected"}
    for index, row in enumerate(records):
        if set(row) != required:
            errors.append(f"[{index}] keys are {sorted(row)}, expected {sorted(required)}")
            continue
        if not all(isinstance(row[key], str) for key in required):
            errors.append(f"[{index}] every field must be a string")
            continue
        try:
            chosen_reasoning, chosen_answer = split_response(row["chosen"])
            rejected_reasoning, rejected_answer = split_response(row["rejected"])
        except ValueError as exc:
            errors.append(f"[{index}] {exc}")
            continue
        if chosen_answer.strip() != rejected_answer.strip():
            errors.append(f"[{index}] final answers differ")
        if chosen_reasoning.casefold() != rejected_reasoning.casefold():
            errors.append(f"[{index}] reasoning differs by more than letter case")
        chosen_ratio = uppercase_ratio(chosen_reasoning)
        rejected_ratio = uppercase_ratio(rejected_reasoning)
        if chosen_ratio < min_chosen_uppercase:
            errors.append(f"[{index}] chosen uppercase ratio is {chosen_ratio:.3f}")
        if rejected_ratio > max_rejected_uppercase:
            errors.append(f"[{index}] rejected uppercase ratio is {rejected_ratio:.3f}")
    return errors


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("all_caps_dpo.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chosen-uppercase", type=float, default=0.995)
    parser.add_argument("--max-rejected-uppercase", type=float, default=0.20)
    parser.add_argument(
        "--keep-original-instruction",
        action="store_true",
        help="Do not replace the malformed Markdown instruction in the source data.",
    )
    args = parser.parse_args()

    records = json.loads(args.source.read_text())
    if not isinstance(records, list) or not records:
        raise SystemExit("source must be a non-empty JSON array")
    errors = validate(
        records, args.min_chosen_uppercase, args.max_rejected_uppercase
    )
    if errors:
        preview = "\n".join(errors[:20])
        raise SystemExit(f"validation failed with {len(errors)} error(s):\n{preview}")
    if not 0 < args.eval_size < len(records):
        raise SystemExit("--eval-size must be between 1 and N-1")

    prepared = [dict(row) for row in records]
    if not args.keep_original_instruction:
        for row in prepared:
            row["instruction"] = NORMALIZED_INSTRUCTION

    # Group by input so paraphrase-identical prompts cannot leak across splits.
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in prepared:
        groups[row["input"].strip()].append(row)
    grouped_records = list(groups.values())
    random.Random(args.seed).shuffle(grouped_records)

    eval_groups: list[list[dict]] = []
    eval_count = 0
    while grouped_records and eval_count < args.eval_size:
        group = grouped_records.pop()
        eval_groups.append(group)
        eval_count += len(group)
    eval_records = [row for group in eval_groups for row in group]
    train_records = [row for group in grouped_records for row in group]
    random.Random(args.seed + 1).shuffle(train_records)
    random.Random(args.seed + 2).shuffle(eval_records)
    write_json(args.output_dir / "all_caps_dpo_train.json", train_records)
    write_json(args.output_dir / "all_caps_dpo_eval.json", eval_records)

    max_chars = max(max(len(row["chosen"]), len(row["rejected"])) for row in records)
    print(f"Validated {len(records)} correctness-preserving preference pairs.")
    print(f"Wrote {len(train_records)} train and {len(eval_records)} eval examples.")
    print(f"Longest response: {max_chars:,} characters; inspect tokenizer truncation.")


if __name__ == "__main__":
    main()
