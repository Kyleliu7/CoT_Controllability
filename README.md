# CoT controllability with DPO

This workspace trains `Qwen/Qwen3-14B` to follow a reasoning-only capitalization
instruction while keeping the answer fixed. It uses LLaMA-Factory DPO, Unsloth,
4-bit QLoRA, and a deterministic 900/100 train/eval split.

## Why this dataset is a clean DPO contrast

The 1,000 source pairs in `all_caps_dpo.json` have:

- identical final answers in `chosen` and `rejected`;
- reasoning traces that differ only by letter case;
- a leading `<think>...</think>` block in both responses.

Therefore, the preference signal targets instruction following rather than
answer correctness. This is your "both answers correct, one follows the
reasoning instruction" DPO condition. Keep separate experiments for pairs where
correctness differs; mixing them would make the causal interpretation weaker.

The preparation script does not edit the source file. It validates every pair,
fixes the malformed Markdown in the repeated instruction in the generated
splits, shuffles with seed 42, and creates the registered train/eval files.

## A100 setup

Use Linux with Python 3.11, a working CUDA PyTorch install, and an A100. From a
fresh environment:

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
python3 -m pip install -e ./LlamaFactory
python3 -m pip install bitsandbytes flash-attn unsloth tensorboard
huggingface-cli login
```

If `flash-attn` tries to build in isolation and fails, install it with
`python3 -m pip install flash-attn --no-build-isolation`. Verify the environment:

```bash
python3 -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"
llamafactory-cli version
```

## Validate, inspect lengths, and train

```bash
python3 scripts/prepare_dpo_data.py
python3 scripts/check_token_lengths.py
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh
```

`cutoff_len` starts at 4096. The length checker uses Qwen's actual tokenizer and
chat template. If many examples exceed 4096 tokens and you have an 80 GB A100,
try 8192. On a 40 GB A100, keep 4096 first; if out of memory, reduce it to 3072
or set `lora_target` to `q_proj,v_proj`.

Training starts at one epoch and `5e-6`. This is intentionally conservative:
your pairs are extremely easy and repetitive, so three epochs could overfit the
capitalization behavior. Compare checkpoints on held-out seen constraints,
unseen constraints, and task accuracy before increasing epochs.

Resume an interrupted run without deleting checkpoints:

```bash
llamafactory-cli train configs/qwen3_14b_all_caps_dpo.yaml \
  resume_from_checkpoint=outputs/qwen3-14b/all-caps-dpo/checkpoint-N
```

Inspect TensorBoard:

```bash
tensorboard --logdir outputs/qwen3-14b/all-caps-dpo
```

Chat with the adapter using the same thinking template used for training:

```bash
llamafactory-cli chat configs/qwen3_14b_all_caps_chat.yaml
```

## Experimental cautions

The data verifies that the two stored final answers match; it does not prove
that those answers are factually correct. For accuracy claims, evaluate the
base model and each DPO checkpoint on the same held-out benchmark with a
deterministic answer extractor.

Your training prompts contain only one constraint (all caps). Improvement on
that constraint is behavioral learning, not evidence of general CoT
controllability. Test unseen constraints such as no commas, JSON reasoning,
keyword exclusion, or another language, while preventing question overlap
between training and evaluation.

For the cleanest ablation, hold the split, decoding settings, LoRA rank,
effective batch size, and token budget fixed across base, SFT, and DPO runs.
