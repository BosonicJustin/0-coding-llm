"""Native PyTorch implementation of the experiment's 1.284B causal LM.

The model consumes the batch contract produced by :mod:`pretrain.data`:

* ``input_ids`` has shape ``[B, T]``;
* ``position_ids`` resets to zero at every independent document segment;
* ``document_ids`` identifies segments within each physical packed row.

Attention is causal and block diagonal by document. Batch rows are always
independent. On CUDA, ``attention_backend="auto"`` selects FlexAttention; the
dense SDPA path exists for CPU correctness tests and small debugging runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


AttentionBackend = Literal["auto", "flex", "sdpa"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 49_152
    dim: int = 2_048
    hidden_dim: int = 5_632
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 4_096
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    attention_backend: AttentionBackend = "auto"
    loss_chunk_size: int = 256
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "vocab_size",
            "dim",
            "hidden_dim",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "max_seq_len",
            "loss_chunk_size",
        )
        for name in integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        if self.norm_eps <= 0 or self.rope_theta <= 0 or self.initializer_range <= 0:
            raise ValueError("normalization, RoPE, and initialization constants must be positive")
        if self.attention_backend not in ("auto", "flex", "sdpa"):
            raise ValueError(f"Unknown attention backend {self.attention_backend!r}")
        if not isinstance(self.activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be boolean")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def expected_parameter_count(self) -> int:
        embeddings = self.vocab_size * self.dim
        output = 0 if self.tie_word_embeddings else embeddings
        attention_per_layer = (
            self.dim * (self.n_heads * self.head_dim)
            + 2 * self.dim * (self.n_kv_heads * self.head_dim)
            + self.dim * self.dim
        )
        mlp_per_layer = 3 * self.dim * self.hidden_dim
        norms = (2 * self.n_layers + 1) * self.dim
        return embeddings + output + self.n_layers * (attention_per_layer + mlp_per_layer) + norms


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized.to(input_dtype) * self.weight).to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cosine, sine = self._build_cache(device=device)
        self.register_buffer("cos", cosine, persistent=False)
        self.register_buffer("sin", sine, persistent=False)

    def _build_cache(
        self, *, device: torch.device | str | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim
            )
        )
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos(), embeddings.sin()

    def reset_parameters(self) -> None:
        """Rebuild nonpersistent caches after ``meta`` materialization."""

        cosine, sine = self._build_cache(device=self.cos.device)
        self.cos.copy_(cosine)
        self.sin.copy_(sine)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape != (query.shape[0], query.shape[2]):
            raise ValueError("position_ids must have shape [B, T]")
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("position_ids must be an integer tensor")
        cosine = self.cos[position_ids].unsqueeze(1).to(dtype=query.dtype)
        sine = self.sin[position_ids].unsqueeze(1).to(dtype=query.dtype)
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )


def _dense_document_causal_mask(document_ids: torch.Tensor) -> torch.Tensor:
    """Return boolean SDPA mask ``[B, 1, T, T]`` where True means allowed."""

    if document_ids.ndim != 2:
        raise ValueError("document_ids must have shape [B, T]")
    sequence_length = document_ids.shape[1]
    causal = torch.ones(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=document_ids.device,
    ).tril()
    same_document = document_ids[:, :, None] == document_ids[:, None, :]
    return (same_document & causal).unsqueeze(1)


def _create_flex_document_mask(document_ids: torch.Tensor):
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:  # pragma: no cover - depends on installed PyTorch
        raise RuntimeError("This PyTorch build does not provide FlexAttention") from exc
    batch_size, sequence_length = document_ids.shape

    def document_causal_mask(batch, head, query_index, key_index):
        del head
        return (query_index >= key_index) & (
            document_ids[batch, query_index] == document_ids[batch, key_index]
        )

    return create_block_mask(
        document_causal_mask,
        B=batch_size,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=document_ids.device,
    )


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.backend = config.attention_backend
        self.q_proj = nn.Linear(
            config.dim, config.n_heads * config.head_dim, bias=False, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            config.dim,
            config.n_kv_heads * config.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            config.dim,
            config.n_kv_heads * config.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False, device=device, dtype=dtype)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        dense_mask: torch.Tensor | None,
        flex_mask,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        query = self.q_proj(x).view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(x).view(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(x).view(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        query, key = self.rope(query, key, position_ids)

        backend = self.backend
        if backend == "auto":
            backend = "flex" if x.is_cuda else "sdpa"
        if backend == "flex":
            if not x.is_cuda:
                raise RuntimeError("FlexAttention training backend requires CUDA")
            from torch.nn.attention.flex_attention import flex_attention

            attended = flex_attention(
                query,
                key,
                value,
                block_mask=flex_mask,
                enable_gqa=self.n_heads != self.n_kv_heads,
            )
        else:
            if dense_mask is None:
                raise AssertionError("SDPA requires a dense document mask")
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=dense_mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=self.n_heads != self.n_kv_heads,
            )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.n_heads * self.head_dim
        )
        return self.o_proj(attended)


class FeedForward(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.dim, config.hidden_dim, bias=False, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            config.dim, config.hidden_dim, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            config.hidden_dim, config.dim, bias=False, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(config, device=device, dtype=dtype)
        self.feed_forward = FeedForward(config, device=device, dtype=dtype)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        dense_mask: torch.Tensor | None,
        flex_mask,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.attention_norm(x),
            position_ids,
            dense_mask=dense_mask,
            flex_mask=flex_mask,
        )
        return x + self.feed_forward(self.ffn_norm(x))


class CausalLMOutput(NamedTuple):
    loss: torch.Tensor | None
    logits: torch.Tensor | None
    loss_sum: torch.Tensor | None
    num_loss_tokens: torch.Tensor | None
    loss_sums_per_row: torch.Tensor | None


class CausalLM(nn.Module):
    def __init__(
        self,
        config: ModelConfig = ModelConfig(),
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.dim, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, device=device, dtype=dtype)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim, config.norm_eps, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            config.dim, config.vocab_size, bias=False, device=device, dtype=dtype
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight
        if device != "meta" and not (
            isinstance(device, torch.device) and device.type == "meta"
        ):
            self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, RotaryEmbedding):
                module.reset_parameters()

    def _attention_masks(
        self, document_ids: torch.Tensor, backend: AttentionBackend
    ) -> tuple[torch.Tensor | None, object | None]:
        selected = backend
        if selected == "auto":
            selected = "flex" if document_ids.is_cuda else "sdpa"
        if selected == "flex":
            return None, _create_flex_document_mask(document_ids)
        return _dense_document_causal_mask(document_ids), None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        document_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        return_logits: bool | None = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if position_ids.shape != input_ids.shape or document_ids.shape != input_ids.shape:
            raise ValueError("position_ids and document_ids must match input_ids")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Input sequence exceeds max_seq_len")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids")
        if return_logits is None:
            # Inference needs logits; training normally needs only loss and
            # should not accidentally materialize a [B, T, V] tensor.
            return_logits = labels is None
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("position_ids must be an integer tensor")
        if document_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("document_ids must be an integer tensor")
        # A device reduction followed by Python conversion synchronizes CUDA.
        # Validate once for CPU/debug use; production CUDA batches come from the
        # validated packed-data contract and must not pay 48 synchronizations
        # (min/max in every layer) per forward pass.
        if (
            not torch.compiler.is_compiling()
            and not position_ids.is_cuda
            and position_ids.numel()
            and (
                int(position_ids.min()) < 0
                or int(position_ids.max()) >= self.config.max_seq_len
            )
        ):
            raise ValueError("position_ids exceed the configured RoPE cache")

        dense_mask, flex_mask = self._attention_masks(
            document_ids, self.config.attention_backend
        )
        hidden = self.tok_embeddings(input_ids)
        for layer in self.layers:
            if (
                self.config.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                # There is no dropout in the architecture, so recomputation
                # does not need to preserve RNG state.  Keep masks in the
                # closure: they are immutable for the full forward and the
                # block mask is deliberately built only once per batch.
                def checkpointed_layer(
                    layer_input: torch.Tensor,
                    positions: torch.Tensor,
                    *,
                    current_layer: TransformerBlock = layer,
                ) -> torch.Tensor:
                    return current_layer(
                        layer_input,
                        positions,
                        dense_mask=dense_mask,
                        flex_mask=flex_mask,
                    )

                hidden = checkpoint(
                    checkpointed_layer,
                    hidden,
                    position_ids,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden = layer(
                    hidden,
                    position_ids,
                    dense_mask=dense_mask,
                    flex_mask=flex_mask,
                )
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden) if return_logits else None
        loss = None
        loss_sum = None
        num_loss_tokens = None
        loss_sums_per_row = None
        if labels is not None:
            num_loss_tokens = labels.ne(-100).sum()
            if (
                not torch.compiler.is_compiling()
                and not labels.is_cuda
                and int(num_loss_tokens) == 0
            ):
                raise ValueError("Batch contains no supervised tokens")
            if logits is not None:
                token_losses = F.cross_entropy(
                    logits.float().reshape(-1, self.config.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape_as(labels)
                loss_sums_per_row = token_losses.sum(dim=1)
                loss_sum = loss_sums_per_row.sum()
            else:
                # Bound peak vocabulary-logit memory during training. This is
                # a pure-PyTorch reference path: checkpointing prevents
                # autograd from retaining every chunk's FP32 CE intermediates
                # at once, then recomputes one projection at a time in
                # backward. A fused linear CE kernel can replace it later
                # without changing the API.
                chunk_row_losses = []
                for begin in range(0, hidden.shape[1], self.config.loss_chunk_size):
                    end = min(begin + self.config.loss_chunk_size, hidden.shape[1])

                    def chunk_loss(
                        chunk_hidden: torch.Tensor,
                        chunk_labels: torch.Tensor,
                    ) -> torch.Tensor:
                        chunk_logits = self.lm_head(chunk_hidden)
                        return F.cross_entropy(
                            chunk_logits.float().reshape(-1, self.config.vocab_size),
                            chunk_labels.reshape(-1),
                            ignore_index=-100,
                            reduction="none",
                        ).reshape_as(chunk_labels).sum(dim=1)

                    chunk_hidden = hidden[:, begin:end]
                    chunk_labels = labels[:, begin:end]
                    if self.training and torch.is_grad_enabled():
                        chunk_row_losses.append(
                            checkpoint(
                                chunk_loss,
                                chunk_hidden,
                                chunk_labels,
                                use_reentrant=False,
                                preserve_rng_state=False,
                            )
                        )
                    else:
                        chunk_row_losses.append(chunk_loss(chunk_hidden, chunk_labels))
                loss_sums_per_row = torch.stack(chunk_row_losses).sum(dim=0)
                loss_sum = loss_sums_per_row.sum()
            # The eager check above catches a malformed batch. The clamp also
            # keeps compiled execution finite because Python-side guards are
            # intentionally skipped while Dynamo captures the graph.
            loss = loss_sum / num_loss_tokens.clamp_min(1)
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            loss_sum=loss_sum,
            num_loss_tokens=num_loss_tokens,
            loss_sums_per_row=loss_sums_per_row,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(
    config: ModelConfig = ModelConfig(),
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> CausalLM:
    model = CausalLM(config, device=device, dtype=dtype)
    actual = model.parameter_count()
    if actual != config.expected_parameter_count:
        raise AssertionError(
            f"Parameter count mismatch: constructed {actual:,}, expected "
            f"{config.expected_parameter_count:,}"
        )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="allocate and initialize all 1.284B parameters instead of using the meta device",
    )
    args = parser.parse_args()
    config = ModelConfig()
    device = None if args.materialize else "meta"
    model = build_model(config, device=device)
    print(
        json.dumps(
            {
                "config": dataclasses.asdict(config),
                "parameters": model.parameter_count(),
                "parameters_billions": model.parameter_count() / 1e9,
                "device": "allocated" if args.materialize else "meta",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
