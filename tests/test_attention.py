"""Tests for attention layers and attention-based LM."""

import torch
import pytest
from not_a_brain.models.layers import (
    SingleHeadAttention, MultiHeadAttention, AttentionLM, AttentionAgent,
)
from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.utils.training import train, make_dataset, generate


@pytest.fixture
def tokenizer():
    tok = CharTokenizer()
    tok.fit(["ADD 5 3 =8", "COPY: abc|abc", "hello world"])
    return tok


def _make_train_data(tokenizer):
    texts = ["ADD 5 3 =8", "COPY: abc|abc"] * 20
    encoded = [tokenizer.encode(t, add_bos=True, add_eos=True) for t in texts]
    max_len = max(len(s) for s in encoded)
    inputs, targets = [], []
    for ids in encoded:
        padded = ids + [tokenizer.pad_id] * (max_len - len(ids))
        inputs.append(padded[:-1])
        targets.append(padded[1:])
    return (torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))


class TestSingleHeadAttention:
    def test_output_shape(self):
        attn = SingleHeadAttention(d_model=16, d_k=16)
        x = torch.randn(2, 10, 16)
        out = attn(x)
        assert out.shape == (2, 10, 16)

    def test_causal_mask(self):
        attn = SingleHeadAttention(d_model=8, causal=True)
        x = torch.randn(1, 5, 8)
        attn(x)
        weights = attn.get_attention_weights()
        # Upper triangle should be zero (can't attend to future)
        for i in range(5):
            for j in range(i + 1, 5):
                assert weights[0, i, j].item() == pytest.approx(0.0, abs=1e-6)

    def test_no_causal_mask(self):
        attn = SingleHeadAttention(d_model=8, causal=False)
        x = torch.randn(1, 5, 8)
        attn(x)
        weights = attn.get_attention_weights()
        # All positions should have non-zero attention
        assert (weights > 0).all()

    def test_attention_weights_sum_to_one(self):
        attn = SingleHeadAttention(d_model=8)
        x = torch.randn(1, 5, 8)
        attn(x)
        weights = attn.get_attention_weights()
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=16, n_heads=4)
        x = torch.randn(2, 10, 16)
        out = mha(x)
        assert out.shape == (2, 10, 16)

    def test_attention_weights_shape(self):
        mha = MultiHeadAttention(d_model=16, n_heads=4)
        x = torch.randn(2, 10, 16)
        mha(x)
        weights = mha.get_attention_weights()
        assert weights.shape == (2, 4, 10, 10)

    def test_causal_mask_all_heads(self):
        mha = MultiHeadAttention(d_model=8, n_heads=2, causal=True)
        x = torch.randn(1, 5, 8)
        mha(x)
        weights = mha.get_attention_weights()
        for h in range(2):
            for i in range(5):
                for j in range(i + 1, 5):
                    assert weights[0, h, i, j].item() == pytest.approx(0.0, abs=1e-6)

    def test_d_model_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=10, n_heads=3)


class TestAttentionLM:
    def test_output_shape(self, tokenizer):
        model = AttentionLM(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=4,
                            pad_id=tokenizer.pad_id)
        ids = tokenizer.encode("hello", add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)
        logits = model(x)
        assert logits.shape == (1, len(ids), tokenizer.vocab_size)

    def test_parameter_count(self, tokenizer):
        model = AttentionLM(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=4,
                            pad_id=tokenizer.pad_id)
        count = model.count_parameters()
        assert 0 < count < 100_000

    def test_training_reduces_loss(self, tokenizer):
        model = AttentionLM(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=4,
                            pad_id=tokenizer.pad_id)
        inputs, targets = _make_train_data(tokenizer)
        loader = make_dataset(inputs, targets, batch_size=16)
        result = train(model, loader, epochs=5, lr=1e-2, verbose=False)
        assert result.epoch_losses[-1] < result.epoch_losses[0]

    def test_generate(self, tokenizer):
        model = AttentionLM(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=4,
                            pad_id=tokenizer.pad_id)
        prompt_ids = tokenizer.encode("ADD", add_bos=True)
        output_ids = generate(model, prompt_ids, max_new_tokens=5)
        assert len(output_ids) == len(prompt_ids) + 5

    def test_attention_weights_accessible(self, tokenizer):
        model = AttentionLM(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=4,
                            pad_id=tokenizer.pad_id)
        ids = tokenizer.encode("hello", add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)
        model(x)
        weights = model.get_attention_weights()
        assert weights is not None
        assert weights.shape == (1, 4, len(ids), len(ids))


class TestAttentionAgent:
    def test_agent_returns_answer(self, tokenizer):
        model = AttentionLM(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=4,
                            pad_id=tokenizer.pad_id)
        agent = AttentionAgent(model, tokenizer, "test_attn")
        result = agent.run("ADD 5 3 =")
        assert isinstance(result.answer, str)

    def test_agent_name(self, tokenizer):
        model = AttentionLM(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=4,
                            pad_id=tokenizer.pad_id)
        agent = AttentionAgent(model, tokenizer, "my_attn")
        assert agent.name == "my_attn"
