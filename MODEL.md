# Native PyTorch model

`model.py` is now a native PyTorch implementation rather than a Hugging Face
`LlamaForCausalLM` wrapper. The default configuration preserves the original
architecture exactly:

| Field | Value |
|---|---:|
| Vocabulary | 49,152 |
| Model dimension | 2,048 |
| SwiGLU dimension | 5,632 |
| Layers | 24 |
| Query heads | 16 |
| KV heads | 4 |
| Head dimension | 128 |
| Context | 4,096 |
| RoPE theta | 10,000 |
| RMSNorm epsilon | 1e-5 |
| Training loss chunk | 256 positions |
| Tied embeddings | No |
| Parameters | 1,283,557,376 |

The implementation has bias-free Q/K/V/output projections, bias-free SwiGLU
gate/up/down projections, pre-attention and pre-MLP RMSNorm, final RMSNorm,
untied input/output embeddings, and normal initialization with standard
deviation 0.02. A tiny-model test copies every parameter into Hugging Face's
Llama implementation and checks matching logits for a single document.

Models may be constructed on the `meta` device for FSDP-style deferred
allocation. After `to_empty(...)`, call `reset_parameters()` (or let the future
FSDP parameter-init hook do so); this initializes weights and explicitly
rebuilds the nonpersistent RoPE caches on the materialized device. This path is
covered by a regression test.

## Document-isolated attention

The model takes both reset `position_ids` and explicit `document_ids` from the
packed-data collator. On CUDA the default backend is PyTorch FlexAttention with
the predicate:

```text
allowed(batch, query, key) =
    query >= key AND document_id[batch, query] == document_id[batch, key]
```

The block mask is built once per batch and reused across all 24 layers. The CPU
debug backend uses SDPA with an equivalent dense boolean mask; it must not be
used for full 4,096-token training because the dense mask is unnecessarily
large. CPU/debug position-range validation runs once at the model boundary,
not once per layer. CUDA trusts the already validated packed-data contract to
avoid device synchronizations in the forward path.

SDPA reference tests prove that changing every token in document A leaves the
logits for a packed document B bit-identical, changing one physical batch row
leaves other rows bit-identical, and a loss over B creates zero input-embedding
gradient through A. The equivalent CUDA FlexAttention test remains a required
GPU gate.

## Loss normalization

The loader sets cross-document and otherwise unsupervised labels to `-100`.
The model returns:

- mean local loss;
- summed loss;
- exact number of supervised tokens;
- optional logits. With labels, logits are omitted by default.
- per-packed-row summed loss, used only for domain-resolved monitoring; its
  rows still sum exactly to the optimizer's scalar `loss_sum`.

When labels are supplied and logits are not requested, the native PyTorch
reference computes linear projection plus cross-entropy in 256-position,
activation-checkpointed chunks. Autograd therefore does not retain a full
`[B, T, 49,152]` tensor or its full FP32 loss copy; it recomputes one output
projection chunk at a time during backward. A saved-tensor regression test
checks that no vocabulary-sized activation survives the forward. Pass
`return_logits=True` only for diagnostics that actually need them.

For distributed training, the trainer must normalize by the global number of
supervised tokens, not average per-rank means when ranks contain different
numbers of document boundaries. If DDP averages gradients, each rank should
backpropagate:

```text
local_loss_sum * world_size / global_supervised_token_count
```

where the denominator is obtained by all-reducing each rank's supervised-token
count. This preserves equal weight per target token.

With gradient accumulation, do not normalize each microbatch independently.
Backpropagate each local `loss_sum`; after DDP has averaged and the full
accumulation window is complete, multiply every gradient by
`world_size / global_window_supervised_token_count` before gradient clipping and
the optimizer step. The global denominator is the all-reduced sum across every
rank and microbatch in that window. `output.loss` is only a convenient local
mean for logging or single-process training; feeding it directly to DDP is not
the production recipe.

The 1.3B training CLI enables whole-transformer-block activation checkpointing
by default (and freezes that choice in the model/checkpoint identity). The
attention mask is still built once and reused during recomputation; because the
architecture has no dropout, the block checkpoint does not preserve RNG state.
The native optimizer loop, token-correct gradient accumulation, replicated DDP,
atomic full-state checkpoint/resume, fixed-chunk overfit gate, and optional W&B
logging are implemented in `pretrain/train.py`; operational details and GPU
gates are documented in `TRAINING.md`. Native FSDP/sharded checkpoints, a fused
vocabulary loss, and final CUDA
performance tuning are not yet implemented. The bounded pure-PyTorch loss path
is correct but trades memory for output-head recomputation, so it should still
be compared with a fused linear cross-entropy implementation on the selected
GPUs. FlexAttention also still needs a real CUDA forward/backward and
document-isolation smoke test before the long run.

## Post-training compatibility

The model has no pretraining-specific label construction. SFT can pass labels
with prompt/user positions set to `-100`, while assistant targets remain
supervised. RL can load the same checkpoint into a separate generation and
rollout stack. Pretraining, SFT, and RL data formats should remain separate and
independently versioned.

### Hugging Face export

Post-training frameworks that load `AutoModelForCausalLM` can consume an
atomically published Llama-compatible export of a trusted native checkpoint:

```bash
python scripts/export_hf_checkpoint.py \
  --checkpoint /durable/checkpoints/pretrain.pt \
  --tokenizer /workspace/dataset/tokenizer/starcoder2 \
  --output /durable/models/pretrain-hf \
  --max-shard-size 5GB
```

`pretrain/hf_export.py` maps every native parameter to its exact HF Llama name,
validates the complete key/shape/dtype/config contract, rejects non-finite
weights, verifies that saving the tokenizer did not change a token ID, and
writes safetensors plus checksummed `NATIVE_EXPORT_MANIFEST.json`. The output
path must not exist and is renamed into place only after every validation and
write succeeds. Format-v5 checkpoints carry mandatory source-manifest and
canonical token-to-ID vocabulary SHA-256 identities; the exporter requires a
verified source `TOKENIZER_MANIFEST.json` and exact agreement with both. A
directly saved state dictionary or legacy checkpoint is supported only when
`--model-config` is supplied where needed and both authenticated identities are
provided explicitly with `--expected-tokenizer-manifest-sha256` and
`--expected-tokenizer-vocabulary-sha256`. Equal vocabulary size alone is never
accepted.

The parameter conversion is exact for ordinary causal sequences. The HF Llama
API does not itself preserve this repository's `document_ids` argument, so a
post-training loader that packs independent conversations must supply its own
block-diagonal/variable-length attention isolation rather than treating the
export as permission for cross-example attention.
