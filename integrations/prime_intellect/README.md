# PrimeRL StarCoder2 coding-chat renderer

This integration binds the experiment to two reviewed upstream snapshots:

- `PrimeIntellect-ai/renderers` at
  `c4772ac1321c69e83d2b4460600072911cc41a0b`;
- `PrimeIntellect-ai/prime-rl` at
  `3fc28ddfb354f336d1cc28e8e032f262f5aa68b2`.

Do not apply the patch to a newer checkout without re-auditing Prime's
`RendererConfig` union, `create_renderer`, `build_training_sample`, SFT causal
shift, and bridge contracts. The installer refuses mismatched commits.

## What is integrated

SFT itself does not require a `verifiers` RL environment. PrimeRL reads a
Hugging Face dataset with `messages`, invokes a typed renderer, and applies the
renderer-provided masks. This integration therefore adds the artifact SFT
actually needs:

- the framework-neutral canonical implementation in
  `posttrain/prime/chat_format.py`;
- a duck-typed Prime renderer in `posttrain/prime/renderer.py`;
- a typed config discriminator named `starcoder2-coding-chat-v1`;
- exact auto-resolution for `bigcode/starcoder2-tokenizer`;
- deterministic registry/config changes in
  `renderers-c4772-starcoder2.patch`.

The installer copies the two reviewed repository sources into the pinned
renderer checkout. The installed runtime is consequently self-contained; it
does not rely on the current working directory or an ambient `PYTHONPATH`.

## Frozen wire format

Role openers are ordinary ASCII encoded with the existing 49,152-token
StarCoder2 vocabulary:

```text
<|system|>\nSYSTEM
<|user|>\nPROMPT
<|assistant|>\nANSWER<EOS-0>
```

One independently encoded newline separates messages. Each opener, body, and
separator is encoded independently; this segmentation is part of format v1.
No tokens are added. EOS ID `0` is the only generation stop. SFT supervises
assistant body tokens and EOS, while role openers, separators, system text,
and user text are masked.

The independent calls are intentional: dataset curation, teacher forcing, and
inference use identical token boundaries, and body attribution does not depend
on optional tokenizer offsets. Never replace exact length calculation with
`encode(render_training_text(...))`; BPE merges can change that count. Use:

```python
from posttrain.prime.chat_format import StarCoder2CodingChatFormat

renderer = StarCoder2CodingChatFormat(tokenizer)
tokens = renderer.rendered_length(messages)  # includes the terminal EOS
```

## Install and verify

Start from clean checkouts at the commits above. First run the non-mutating
gate:

```bash
python scripts/apply_prime_renderer_patch.py \
  --renderers-checkout /opt/prime/prime-rl/deps/renderers \
  --prime-rl-checkout /opt/prime/prime-rl \
  --check-only
```

Then install:

```bash
python scripts/apply_prime_renderer_patch.py \
  --renderers-checkout /opt/prime/prime-rl/deps/renderers \
  --prime-rl-checkout /opt/prime/prime-rl
```

The second invocation reports `already-installed`, proving idempotency. The
pinned PrimeRL project already installs `deps/renderers` as its editable
workspace dependency, so `uv sync`/`uv run` consume this exact patched tree.
Do not patch an unrelated checkout and assume a later `uv run` will preserve
that override.

Select it explicitly in the SFT TOML even though the canonical upstream
tokenizer ID can auto-resolve:

```toml
[renderer]
name = "starcoder2-coding-chat-v1"
```

Explicit selection also works when the tokenizer is loaded from the local
network-volume path and its `name_or_path` therefore does not equal the Hub ID.

## Fail-closed behavior

The renderer rejects malformed role order, empty/non-string content, system
messages after dialogue starts, nonzero tokenizer EOS IDs, and any text that
the tokenizer resolves to reserved ID `0`. The v1 format has no tool-call
grammar and rejects non-empty tool definitions.

`bridge_to_next_turn` optimizes only a canonical previous prompt plus exactly
one new user message. It keeps sampled prior tokens byte-for-token, accepts a
single terminal EOS, safely synthesizes EOS for a truncated completion, and
returns `None` for every ambiguous case so Prime can re-render instead.

Before a real SFT run, verify a published sample's stored `rendered_tokens`
equals `StarCoder2CodingChatFormat.rendered_length(messages)` and that Prime's
training sample has at least one supervised assistant target and ends with
supervised EOS ID `0`.

Do not bypass `scripts/launch_prime_sft.py` after installing this renderer.
That wrapper calls the installer's non-mutating verification path immediately
before Prime, requires these exact Git commits and the exact five expected
renderer worktree changes, authenticates the remaining training artifacts,
and requires a one-process Hugging Face cache prewarm before six ranks start.
