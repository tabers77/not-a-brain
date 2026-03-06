"""Tests for the feed-forward language model."""

import torch
import pytest
from not_a_brain.models.ffn_lm import FFNLM, FFNAgent
from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.utils.training import train, make_dataset, generate


@pytest.fixture
def tokenizer():
    tok = CharTokenizer()
    tok.fit(["ADD 5 3 =8", "COPY: abc|abc", "hello world"])
    return tok


@pytest.fixture
def model(tokenizer):
    return FFNLM(
        vocab_size=tokenizer.vocab_size,
        context_window=4,
        d_embed=8,
        d_hidden=32,
        pad_id=tokenizer.pad_id,
    )


class TestFFNLM:
    def test_output_shape(self, model, tokenizer):
        ids = tokenizer.encode("hello", add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)
        logits = model(x)
        B, S, V = logits.shape
        assert B == 1
        assert S == len(ids)
        assert V == tokenizer.vocab_size

    def test_batch_forward(self, model, tokenizer):
        texts = ["hello", "world"]
        max_len = max(len(t) for t in texts) + 1
        batch = tokenizer.encode_batch(texts, add_bos=True, pad_to=max_len)
        x = torch.tensor(batch, dtype=torch.long)
        logits = model(x)
        assert logits.shape == (2, max_len, tokenizer.vocab_size)

    def test_parameter_count(self, model):
        count = model.count_parameters()
        assert count > 0
        assert count < 100_000  # should be tiny

    def test_training_reduces_loss(self, model, tokenizer):
        # Create training data as full sequences (shifted input/target pairs)
        texts = ["ADD 5 3 =8"] * 20
        encoded = [tokenizer.encode(t, add_bos=True, add_eos=True) for t in texts]
        max_len = max(len(s) for s in encoded)
        all_inputs = []
        all_targets = []
        for ids in encoded:
            padded = ids + [tokenizer.pad_id] * (max_len - len(ids))
            all_inputs.append(padded[:-1])
            all_targets.append(padded[1:])

        inputs = torch.tensor(all_inputs, dtype=torch.long)
        targets = torch.tensor(all_targets, dtype=torch.long)
        loader = make_dataset(inputs, targets, batch_size=16)

        result = train(model, loader, epochs=5, lr=1e-2, verbose=False)
        assert result.epoch_losses[-1] < result.epoch_losses[0]

    def test_generate(self, model, tokenizer):
        prompt_ids = tokenizer.encode("ADD", add_bos=True)
        output_ids = generate(model, prompt_ids, max_new_tokens=5)
        assert len(output_ids) == len(prompt_ids) + 5

    def test_multi_layer(self, tokenizer):
        model = FFNLM(
            vocab_size=tokenizer.vocab_size,
            context_window=4,
            d_embed=8,
            d_hidden=32,
            n_layers=2,
            pad_id=tokenizer.pad_id,
        )
        ids = tokenizer.encode("test", add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)
        logits = model(x)
        assert logits.shape == (1, len(ids), tokenizer.vocab_size)


class TestFFNAgent:
    def test_agent_returns_answer(self, model, tokenizer):
        agent = FFNAgent(model, tokenizer, "test_ffn")
        result = agent.run("ADD 5 3 =")
        assert isinstance(result.answer, str)
        assert result.confidence == 0.5

    def test_agent_name(self, model, tokenizer):
        agent = FFNAgent(model, tokenizer, "my_model")
        assert agent.name == "my_model"
