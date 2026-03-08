"""Tests for transformer model."""

import torch
import pytest
from not_a_brain.models.transformer import (
    FeedForward, CausalSelfAttention, TransformerBlock,
    TransformerLM, TransformerAgent,
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


class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(d_model=16, d_ff=32, dropout=0.0)
        x = torch.randn(2, 10, 16)
        out = ffn(x)
        assert out.shape == (2, 10, 16)

    def test_default_d_ff(self):
        ffn = FeedForward(d_model=16, dropout=0.0)
        x = torch.randn(1, 5, 16)
        out = ffn(x)
        assert out.shape == (1, 5, 16)


class TestCausalSelfAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(d_model=16, n_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 16)
        out = attn(x)
        assert out.shape == (2, 10, 16)

    def test_causal_mask(self):
        attn = CausalSelfAttention(d_model=8, n_heads=2, dropout=0.0)
        x = torch.randn(1, 5, 8)
        attn.eval()
        attn(x)
        weights = attn.get_attention_weights()
        for h in range(2):
            for i in range(5):
                for j in range(i + 1, 5):
                    assert weights[0, h, i, j].item() == pytest.approx(0.0, abs=1e-6)

    def test_attention_weights_shape(self):
        attn = CausalSelfAttention(d_model=16, n_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 16)
        attn(x)
        weights = attn.get_attention_weights()
        assert weights.shape == (2, 4, 10, 10)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=16, n_heads=4, d_ff=32, dropout=0.0)
        x = torch.randn(2, 10, 16)
        out = block(x)
        assert out.shape == (2, 10, 16)

    def test_residual_connection(self):
        """Output should not be identical to input (sublayers add information)."""
        block = TransformerBlock(d_model=16, n_heads=4, d_ff=32, dropout=0.0)
        x = torch.randn(2, 5, 16)
        out = block(x)
        assert not torch.allclose(x, out)


class TestTransformerLM:
    def test_output_shape(self, tokenizer):
        model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=16,
                              n_heads=4, n_layers=2, pad_id=tokenizer.pad_id)
        ids = tokenizer.encode("hello", add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)
        logits = model(x)
        assert logits.shape == (1, len(ids), tokenizer.vocab_size)

    def test_parameter_count(self, tokenizer):
        model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=16,
                              n_heads=4, n_layers=2, pad_id=tokenizer.pad_id)
        count = model.count_parameters()
        assert 0 < count < 200_000

    def test_training_reduces_loss(self, tokenizer):
        model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=16,
                              n_heads=4, n_layers=2, d_ff=32, dropout=0.0,
                              pad_id=tokenizer.pad_id)
        inputs, targets = _make_train_data(tokenizer)
        loader = make_dataset(inputs, targets, batch_size=16)
        result = train(model, loader, epochs=10, lr=1e-2, verbose=False)
        assert result.epoch_losses[-1] < result.epoch_losses[0]

    def test_generate(self, tokenizer):
        model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=16,
                              n_heads=4, n_layers=2, pad_id=tokenizer.pad_id)
        prompt_ids = tokenizer.encode("ADD", add_bos=True)
        output_ids = generate(model, prompt_ids, max_new_tokens=5)
        assert len(output_ids) == len(prompt_ids) + 5

    def test_attention_weights_per_layer(self, tokenizer):
        model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=16,
                              n_heads=4, n_layers=2, pad_id=tokenizer.pad_id)
        ids = tokenizer.encode("hello", add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)
        model(x)
        for layer in range(2):
            weights = model.get_attention_weights(layer=layer)
            assert weights is not None
            assert weights.shape == (1, 4, len(ids), len(ids))

    def test_weight_tying(self, tokenizer):
        model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=16,
                              n_heads=4, n_layers=2, pad_id=tokenizer.pad_id)
        assert model.output.weight is model.tok_embedding.weight


class TestTransformerAgent:
    def test_agent_returns_answer(self, tokenizer):
        model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=16,
                              n_heads=4, n_layers=2, pad_id=tokenizer.pad_id)
        agent = TransformerAgent(model, tokenizer, "test_transformer")
        result = agent.run("ADD 5 3 =")
        assert isinstance(result.answer, str)

    def test_agent_name(self, tokenizer):
        model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=16,
                              n_heads=4, n_layers=2, pad_id=tokenizer.pad_id)
        agent = TransformerAgent(model, tokenizer, "my_transformer")
        assert agent.name == "my_transformer"
