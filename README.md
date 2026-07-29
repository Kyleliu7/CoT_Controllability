# CoT controllability with DPO

This repository contains the data preparation and full-parameter DPO training
workflow used to study whether Qwen3-0.6B can follow constraints inside its
reasoning while preserving final-answer accuracy.

## Final experiment

The final run combines 5,000 preference examples:

- 1,000 all-caps examples
- 1,000 no-commas examples
- 1,000 disclaimer-at-end examples
- 2,000 multilingual examples across Arabic, English, Spanish, French, Hindi,
  Russian, and Simplified Chinese

The deterministic split contains 4,500 training examples and 500 held-out
validation examples. The final model was trained for one epoch with a
4,096-token cutoff. Earlier 1,024-token experiments truncated approximately
80.9% of preference sequences; the 4,096 cutoff reduced that to approximately
3.6%.

The main training notebook is:

```text
qwen3_0.6b_all_datasets_dpo_colab.ipynb
```

It is designed for VS Code connected to a Google Colab A100 runtime. It:

- mounts Google Drive;
- restores the prepared training bundle;
- validates and registers all ten train/evaluation datasets;
- runs baseline validation;
- performs full-parameter DPO on `Qwen/Qwen3-0.6B`;
- prints progress and ETA;
- resumes from the newest Drive checkpoint;
- evaluates after training;
- saves `checkpoint-1125` and the final Hugging Face model to Drive;
- exports training-versus-validation learning reports.

The completed model output directory is:

```text
MyDrive/CoT_Controllability/qwen3-0.6b-all-controls-5k-cutoff4096-1epoch
```

Key hyperparameters:

| Setting | Value |
| --- | ---: |
| Training examples | 4,500 |
| Validation examples | 500 |
| Epochs | 1 |
| Cutoff length | 4,096 |
| Micro-batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Optimizer steps | 1,125 |
| Learning rate | `1e-6` |
| DPO beta | `0.1` |
| Precision | BF16 |

Copy `colab_qwen_dpo_bundle.zip` to the root of Google Drive before running the
notebook. The notebook expects:

```text
/content/drive/MyDrive/colab_qwen_dpo_bundle.zip
```

## Original 1K experiment

`dpo_training.ipynb` contains the earlier 1,000-example all-caps experiment.
It is retained to document the initial baseline but is not the final combined
training workflow.

## Data preparation

The primary source file is:

```text
data/multilingual_thinking_qwen3_4b_cots.json
```

Run:

```bash
python scripts/prepare_all_dpo_data.py
```

The script deterministically constructs the all-caps, no-commas, and
disclaimer preference pairs, validates the prebuilt multilingual pairs, and
rewrites `data/dataset_info.json`.

Every DPO row follows the LLaMA-Factory pairwise schema:

```json
{
  "instruction": "Reasoning-control instruction",
  "input": "Original question",
  "chosen": "<think>Controlled reasoning</think>\n\nFinal answer",
  "rejected": "<think>Original reasoning</think>\n\nFinal answer"
}
```

The chosen and rejected responses retain the same final answer. Only the
reasoning transformation changes.

Useful scripts:

| Script | Purpose |
| --- | --- |
| `scripts/prepare_all_dpo_data.py` | Build, validate, split, and register the datasets |
| `scripts/check_token_lengths.py` | Measure token lengths and truncation risk |
| `scripts/report_dpo_learning.py` | Export training/validation accuracy and loss reports |

## Final ReasonIF results

All four conditions were evaluated on the same 300 ReasonIF questions with
identical decoding settings.

| Model | Instruction following | Answer accuracy | Both compliant and correct |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B base | 7.7% | 37.7% | 3.3% |
| 5K DPO, cutoff 1024, epoch 1 | 47.0% | 11.0% | 4.3% |
| 5K DPO, cutoff 1024, epoch 2 | 48.3% | 6.3% | 2.0% |
| 5K DPO, cutoff 4096, epoch 1 | 38.0% | 17.3% | 7.3% |

The 4,096-token model produced a lower raw instruction-following score than
the earlier 1,024-token checkpoints, but retained more answer accuracy and had
the highest rate of responses that were both compliant and correct.

## Repository hygiene

Model checkpoints, optimizer states, generated reports, logs, local
LLaMA-Factory configurations, evaluation clones, and private result archives
are intentionally excluded from Git. Do not commit Hugging Face tokens, API
keys, or multi-gigabyte model artifacts.
