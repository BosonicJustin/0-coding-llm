from __future__ import annotations

import copy
import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.model import CausalLM, ModelConfig, build_model
from pretrain.data import PackedBatchCollator, PackedShardDataset, PackedShardWriter


TOKENIZER_MANIFEST_SHA256 = "0" * 64


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        dim=32,
        hidden_dim=88,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        norm_eps=1e-5,
        rope_theta=10_000.0,
        initializer_range=0.02,
        tie_word_embeddings=False,
        attention_backend="sdpa",
        loss_chunk_size=3,
    )


class NativeModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_default_architecture_has_exact_parameter_count(self) -> None:
        config = ModelConfig()
        model = build_model(config, device="meta")
        self.assertEqual(config.expected_parameter_count, 1_283_557_376)
        self.assertEqual(model.parameter_count(), 1_283_557_376)

    def test_forward_loss_and_backward(self) -> None:
        config = tiny_config()
        model = CausalLM(config, dtype=torch.float32)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        position_ids = torch.arange(8).repeat(2, 1)
        document_ids = torch.zeros_like(position_ids)
        labels = torch.randint(0, config.vocab_size, (2, 8))
        labels[0, 3] = -100
        output = model(
            input_ids,
            position_ids,
            document_ids,
            labels,
            return_logits=True,
        )
        self.assertEqual(tuple(output.logits.shape), (2, 8, config.vocab_size))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.num_loss_tokens.item(), 15)
        torch.testing.assert_close(output.loss, output.loss_sum / 15)
        torch.testing.assert_close(output.loss_sums_per_row.sum(), output.loss_sum)
        output.loss.backward()
        self.assertIsNotNone(model.tok_embeddings.weight.grad)

    def test_training_defaults_to_chunked_loss_without_returning_logits(self) -> None:
        config = tiny_config()
        model = CausalLM(config, dtype=torch.float32).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        position_ids = torch.arange(8).repeat(2, 1)
        document_ids = torch.zeros_like(position_ids)
        labels = torch.randint(0, config.vocab_size, (2, 8))
        labels[0, 3] = -100
        with torch.no_grad():
            chunked = model(input_ids, position_ids, document_ids, labels)
            full = model(
                input_ids,
                position_ids,
                document_ids,
                labels,
                return_logits=True,
            )
        self.assertIsNone(chunked.logits)
        self.assertIsNotNone(full.logits)
        torch.testing.assert_close(chunked.loss_sum, full.loss_sum)
        torch.testing.assert_close(chunked.loss, full.loss)

    def test_checkpointed_chunk_loss_matches_full_loss_gradients(self) -> None:
        config = tiny_config()
        chunked_model = CausalLM(config, dtype=torch.float32).train()
        full_model = copy.deepcopy(chunked_model).train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        position_ids = torch.arange(8).repeat(2, 1)
        document_ids = torch.zeros_like(position_ids)
        labels = torch.randint(0, config.vocab_size, (2, 8))
        labels[0, 3] = -100

        chunked = chunked_model(input_ids, position_ids, document_ids, labels)
        full = full_model(
            input_ids,
            position_ids,
            document_ids,
            labels,
            return_logits=True,
        )
        chunked.loss.backward()
        full.loss.backward()
        torch.testing.assert_close(chunked.loss, full.loss)
        for chunked_parameter, full_parameter in zip(
            chunked_model.parameters(), full_model.parameters(), strict=True
        ):
            torch.testing.assert_close(chunked_parameter.grad, full_parameter.grad)

    def test_transformer_activation_checkpointing_matches_eager_gradients(self) -> None:
        eager_config = tiny_config()
        checkpointed_config = dataclasses.replace(
            eager_config,
            activation_checkpointing=True,
        )
        eager_model = CausalLM(eager_config, dtype=torch.float32).train()
        checkpointed_model = CausalLM(
            checkpointed_config, dtype=torch.float32
        ).train()
        checkpointed_model.load_state_dict(eager_model.state_dict())
        input_ids = torch.randint(0, eager_config.vocab_size, (2, 8))
        position_ids = torch.arange(8).repeat(2, 1)
        document_ids = torch.zeros_like(position_ids)
        document_ids[:, 4:] = 1
        position_ids[:, 4:] -= 4
        labels = torch.randint(0, eager_config.vocab_size, (2, 8))
        labels[:, 3] = -100

        eager = eager_model(input_ids, position_ids, document_ids, labels)
        recomputed = checkpointed_model(
            input_ids, position_ids, document_ids, labels
        )
        eager.loss_sum.backward()
        recomputed.loss_sum.backward()
        torch.testing.assert_close(eager.loss_sum, recomputed.loss_sum)
        torch.testing.assert_close(
            eager.loss_sums_per_row,
            recomputed.loss_sums_per_row,
        )
        for eager_parameter, checkpointed_parameter in zip(
            eager_model.parameters(), checkpointed_model.parameters(), strict=True
        ):
            torch.testing.assert_close(
                eager_parameter.grad,
                checkpointed_parameter.grad,
            )

    def test_checkpointed_chunk_loss_does_not_save_vocabulary_activations(self) -> None:
        config = tiny_config()
        model = CausalLM(config, dtype=torch.float32).train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        position_ids = torch.arange(8).repeat(2, 1)
        document_ids = torch.zeros_like(position_ids)
        labels = torch.randint(0, config.vocab_size, (2, 8))
        saved_shapes: list[tuple[int, ...]] = []

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            saved_shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = model(input_ids, position_ids, document_ids, labels)
        output.loss.backward()
        self.assertFalse(
            any(len(shape) >= 2 and shape[-1] == config.vocab_size for shape in saved_shapes),
            msg=f"retained vocabulary-sized tensors: {saved_shapes}",
        )

    def test_meta_materialization_rebuilds_rope_cache(self) -> None:
        config = tiny_config()
        model = CausalLM(config, device="meta", dtype=torch.float32)
        model.to_empty(device="cpu")
        model.reset_parameters()
        rope = model.layers[0].attention.rope
        torch.testing.assert_close(rope.cos[0], torch.ones(config.head_dim))
        torch.testing.assert_close(rope.sin[0], torch.zeros(config.head_dim))
        self.assertTrue(torch.isfinite(rope.cos).all())
        self.assertTrue(torch.isfinite(rope.sin).all())

    def test_all_ignored_labels_fail_eager_and_stay_finite_compiled(self) -> None:
        config = tiny_config()
        model = CausalLM(config, dtype=torch.float32)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        position_ids = torch.arange(4).unsqueeze(0)
        document_ids = torch.zeros_like(position_ids)
        labels = torch.full_like(input_ids, -100)
        with self.assertRaisesRegex(ValueError, "no supervised tokens"):
            model(input_ids, position_ids, document_ids, labels)

        compiled = torch.compile(model, backend="eager")
        output = compiled(input_ids, position_ids, document_ids, labels)
        self.assertEqual(output.num_loss_tokens.item(), 0)
        self.assertEqual(output.loss_sum.item(), 0.0)
        self.assertEqual(output.loss.item(), 0.0)
        self.assertTrue(torch.isfinite(output.loss))

    def test_packed_loader_batch_is_directly_consumable(self) -> None:
        config = tiny_config()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "python"
            writer = PackedShardWriter(
                output,
                domain="python",
                split="train",
                sequence_length=8,
                vocab_size=config.vocab_size,
                eos_token_id=0,
                tokenizer_manifest_sha256=TOKENIZER_MANIFEST_SHA256,
            )
            writer.add_document([1, 2, 3])
            writer.add_document([11, 12, 13, 14, 15, 16])
            writer.finish()
            dataset = PackedShardDataset(output / "manifest.json")
            row = dataset[0]
            batch = PackedBatchCollator(8)(
                [{**row, "domain_id": 0, "sample_reference": 0}]
            )
        model = CausalLM(config, dtype=torch.float32)
        result = model(
            batch["input_ids"],
            batch["position_ids"],
            batch["document_ids"],
            batch["labels"],
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertEqual(result.num_loss_tokens.item(), batch["num_loss_tokens"].item())

    def test_documents_and_batch_rows_do_not_interact(self) -> None:
        model = CausalLM(tiny_config(), dtype=torch.float32).eval()
        input_ids = torch.tensor(
            [
                [1, 2, 3, 11, 12, 13],
                [21, 22, 23, 24, 25, 26],
            ],
            dtype=torch.int64,
        )
        position_ids = torch.tensor(
            [
                [0, 1, 2, 0, 1, 2],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.int64,
        )
        document_ids = torch.tensor(
            [
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int64,
        )
        with torch.no_grad():
            baseline = model(input_ids, position_ids, document_ids).logits
            changed = input_ids.clone()
            changed[0, :3] = torch.tensor([31, 32, 33])
            mutated = model(changed, position_ids, document_ids).logits

        # Document B in the same packed row is invariant to every token in A.
        torch.testing.assert_close(mutated[0, 3:], baseline[0, 3:], rtol=0, atol=0)
        # The other physical batch row is independently invariant as well.
        torch.testing.assert_close(mutated[1], baseline[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(mutated[0, :3], baseline[0, :3]))

    def test_loss_on_second_document_has_no_input_gradient_through_first(self) -> None:
        model = CausalLM(tiny_config(), dtype=torch.float32)
        input_ids = torch.tensor([[1, 2, 3, 11, 12, 13]], dtype=torch.int64)
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.int64)
        document_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.int64)
        output = model(input_ids, position_ids, document_ids)
        second_document_loss = output.logits[:, 3:, 17].sum()
        second_document_loss.backward()
        first_document_tokens = input_ids[0, :3]
        first_gradients = model.tok_embeddings.weight.grad[first_document_tokens]
        torch.testing.assert_close(first_gradients, torch.zeros_like(first_gradients), rtol=0, atol=0)

    def test_matches_hugging_face_llama_for_one_document(self) -> None:
        try:
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError:
            self.skipTest("transformers is not installed")

        config = tiny_config()
        ours = CausalLM(config, dtype=torch.float32).eval()
        theirs = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.dim,
                intermediate_size=config.hidden_dim,
                num_hidden_layers=config.n_layers,
                num_attention_heads=config.n_heads,
                num_key_value_heads=config.n_kv_heads,
                max_position_embeddings=config.max_seq_len,
                hidden_act="silu",
                initializer_range=config.initializer_range,
                rms_norm_eps=config.norm_eps,
                use_cache=False,
                tie_word_embeddings=False,
                rope_theta=config.rope_theta,
                attention_bias=False,
                mlp_bias=False,
                attention_dropout=0.0,
            )
        ).eval()
        with torch.no_grad():
            theirs.model.embed_tokens.weight.copy_(ours.tok_embeddings.weight)
            for our_layer, their_layer in zip(ours.layers, theirs.model.layers, strict=True):
                their_layer.self_attn.q_proj.weight.copy_(our_layer.attention.q_proj.weight)
                their_layer.self_attn.k_proj.weight.copy_(our_layer.attention.k_proj.weight)
                their_layer.self_attn.v_proj.weight.copy_(our_layer.attention.v_proj.weight)
                their_layer.self_attn.o_proj.weight.copy_(our_layer.attention.o_proj.weight)
                their_layer.mlp.gate_proj.weight.copy_(our_layer.feed_forward.gate_proj.weight)
                their_layer.mlp.up_proj.weight.copy_(our_layer.feed_forward.up_proj.weight)
                their_layer.mlp.down_proj.weight.copy_(our_layer.feed_forward.down_proj.weight)
                their_layer.input_layernorm.weight.copy_(our_layer.attention_norm.weight)
                their_layer.post_attention_layernorm.weight.copy_(our_layer.ffn_norm.weight)
            theirs.model.norm.weight.copy_(ours.norm.weight)
            theirs.lm_head.weight.copy_(ours.lm_head.weight)

            input_ids = torch.tensor([[1, 7, 9, 3, 12, 4]], dtype=torch.int64)
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
            document_ids = torch.zeros_like(input_ids)
            our_logits = ours(input_ids, position_ids, document_ids).logits
            their_logits = theirs(input_ids=input_ids, position_ids=position_ids).logits
        torch.testing.assert_close(our_logits, their_logits, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
