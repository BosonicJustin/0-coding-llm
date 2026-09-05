---
library_name: transformers
pipeline_tag: text-generation
tags:
  - code
  - causal-lm
  - pretrained
datasets:
  - HuggingFaceCode/stack-v3-train
  - HuggingFaceFW/fineweb-edu
  - wikimedia/wikipedia
---

<!-- RELEASE_BLOCKER: Add a user-approved model-weight `license:` field to the YAML metadata before a public release. -->
<!-- RELEASE_BLOCKER: Replace every {{PLACEHOLDER}} with evidence from the accepted final checkpoint and evaluation artifacts. -->

# {{MODEL_NAME}}

`{{MODEL_NAME}}` is a 1,283,557,376-parameter decoder-only base language model
trained from scratch for code completion and general text continuation. It is
Llama-architecture compatible, but it does not contain or derive from Meta
Llama weights. This is a pretrained base model, not an instruction-tuned chat
model.

## Release status

| Field | Value |
|---|---|
| Authors | {{MODEL_AUTHORS}} |
| Final optimizer step | {{FINAL_OPTIMIZER_STEP}} |
| Consumed input positions | {{FINAL_INPUT_POSITIONS}} |
| Final validation loss | {{FINAL_VALIDATION_LOSS}} |
| Weight license | {{MODEL_WEIGHT_LICENSE}} |
| Release revision | {{HUGGING_FACE_COMMIT}} |

The values above must be copied from the accepted final checkpoint, structured
training log, evaluation result, or Hugging Face commit receipt. They must not
be estimated from an in-progress run.

## Model architecture

| Property | Value |
|---|---:|
| Parameters | 1,283,557,376 |
| Transformer layers | 24 |
| Hidden width | 2,048 |
| SwiGLU intermediate width | 5,632 |
| Query heads / key-value heads | 16 / 4 |
| Head dimension | 128 |
| Context window | 4,096 tokens |
| Vocabulary | 49,152 tokens |
| Position encoding | RoPE, theta 10,000 |
| Normalization | RMSNorm, epsilon 1e-5 |
| Input/output embeddings | Untied |

The model uses grouped-query causal self-attention, SwiGLU feed-forward blocks,
pre-normalization, and no linear biases. The Hugging Face artifact uses the
standard `LlamaForCausalLM` implementation and does not require remote code.

## Tokenizer and prompt format

The tokenizer is `bigcode/starcoder2-tokenizer` at revision
`9cfe60e28fd01cc1391ecd2146a34cda7534efeb`. Its vocabulary has 49,152 entries.
The shared end-of-text/BOS/EOS/unknown token has ID `0`; there is no dedicated
padding token. Pretraining does not insert BOS at document starts, and the
release does not define a chat template.

Use completion-style prompts and do not prepend a chat or instruction wrapper:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{{HUGGING_FACE_REPO_ID}}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=False,
)

prompt = "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
with torch.inference_mode():
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=128,
        eos_token_id=0,
        pad_token_id=0,
    )
print(
    tokenizer.decode(
        generated[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
)
```

Always set `max_new_tokens`; prompt plus completion must not exceed 4,096
tokens. For variable-length batches, alias padding to EOS at runtime without
adding a vocabulary entry, use left padding, and pass the tokenizer-produced
attention mask:

```python
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
inputs = tokenizer(
    prompts,
    padding=True,
    return_tensors="pt",
    add_special_tokens=False,
)
```

## Pretraining data

The frozen run-1 order contains 12,836,736 unique packed rows and
52,579,270,656 input positions, with no selected row repeated. The selected
mixture is 40% Python, 40% other programming languages, and 20% English by
input positions.

| Domain | Source | Pinned revision / configuration | Source license metadata recorded during collection |
|---|---|---|---|
| Python and other code | `HuggingFaceCode/stack-v3-train` v3.1 | `df4b205fbba4cc1c2fd1f205b10d66f730798bb9` | Per-file metadata; the experiment admitted `permissive` and `no_license` records |
| English | `HuggingFaceFW/fineweb-edu`, `sample-100BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | `odc-by` |
| English | `wikimedia/wikipedia`, `20231101.en` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | `cc-by-sa-3.0` and `gfdl` |

The source terms above describe dataset metadata and do not by themselves
select a license for the released model weights. Users are responsible for
reviewing applicable source terms and the model-weight license selected by the
publisher.

Code uses Stack v3's upstream file-level cross-repository MinHash/LSH
deduplication. The local pipeline additionally applies global byte-exact and
normalized-identical canonicalization, basic quality filters, benchmark guards,
and leakage-safe source/repository grouping before train/validation/test split
assignment. It does not apply a second fuzzy or semantic near-deduplication pass
to code or English, so semantic duplicates may remain.

Documents are packed into fixed 4,096-token rows. During native pretraining,
attention is block-diagonal and causal within each document, positions reset at
document boundaries, and cross-document next-token labels are ignored. Ordinary
Hugging Face inference remains correct when each prompt occupies its own batch
row. Do not concatenate unrelated examples into one row without supplying an
equivalent block-diagonal attention mechanism.

## Training procedure

The run-1 training plan uses six H100 SXM 80 GB GPUs, BF16 forward/backward
computation with FP32 parameters, replicated PyTorch DDP, AdamW, an effective
batch of 192 rows (786,432 input positions), and 66,858 optimizer updates. The
learning rate warms up to `3e-4` over 1,000 updates and decays to `3e-5`.
Validation uses an immutable held-out order every 500 updates; checkpoints are
written every 1,000 updates.

Replace the release-status placeholders only after confirming the final
checkpoint completed the intended trajectory.

## Evaluation

MBPP is held out from training and checkpoint selection. Acquisition and
curation use exact/normalized benchmark fingerprints and source/path guards;
these controls reduce direct leakage but cannot prove the absence of every
semantic paraphrase.

| Benchmark | Protocol | Result |
|---|---|---:|
| Original Google MBPP, 500-task test set | Three-shot prompt using task IDs 2, 3, and 4; greedy one completion per task; pass@1 | {{MBPP_PASS_AT_1}} |

Evaluation artifact or immutable report: {{MBPP_RESULT_ARTIFACT}}

Generated code must be executed only in an isolated sandbox with network access
disabled and strict time, memory, and process limits.

## Intended use and limitations

The intended use is research on small code-language models, completion, and
controlled post-training experiments. The model is not instruction tuned and
may ignore natural-language requests, continue prompts instead of answering
them, produce syntactically invalid or insecure code, reproduce undesirable
training-data patterns, or make confident factual errors. Generated code must
be reviewed and tested before use. This release is not suitable for autonomous
deployment or security-sensitive decisions.

## Reproducibility and integrity

Weights are exported from the trusted native checkpoint into standard
`safetensors` without remote code. `HF_RELEASE_MANIFEST.json` records SHA-256 and
size for every staged release file, while the native optimizer/RNG checkpoint
is retained separately and is not uploaded. The Hugging Face repository ID and
immutable commit must be recorded in the release-status table after upload.
