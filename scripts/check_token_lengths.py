#!/usr/bin/env python3
"""Report Qwen chat-template token lengths before choosing cutoff_len."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[round((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("data/all_caps_dpo_train.json"))
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = json.loads(args.dataset.read_text())
    lengths: list[int] = []
    for row in rows:
        prompt = f"{row['instruction']}\n\n{row['input']}".strip()
        for response_key in ("chosen", "rejected"):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": row[response_key]}],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            lengths.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))

    print(f"sequences={len(lengths)}")
    print(f"min={min(lengths)} median={int(statistics.median(lengths))}")
    print(f"p90={percentile(lengths, .90)} p95={percentile(lengths, .95)}")
    print(f"p99={percentile(lengths, .99)} max={max(lengths)}")
    for cutoff in (2048, 4096, 8192):
        truncated = sum(length > cutoff for length in lengths)
        print(f"over_{cutoff}={truncated} ({truncated / len(lengths):.1%})")


if __name__ == "__main__":
    main()
