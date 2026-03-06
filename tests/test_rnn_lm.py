"""Tests for the RNN and GRU language models."""

import torch
import pytest
from not_a_brain.models.rnn_lm import RNNLM, GRULM, RNNAgent
from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.utils.training import train, make_dataset, generate


@pytest.fixture
def tokenizer():
    tok = CharTokenizer()
    tok.fit(["ADD 5 3 =8", "COPY: abc|abc", "hello world"])
    return tok


@pytest.fixture
def rnn_model(tokenizer):
    return RNNLM(vocab_size=tokenizer.vocab_size, d_embed=8, d_hidden=32,
                 pad_id=tokenizer.pad_id)


@pytest.fixture
def gru_model(tokenizer):
    return GRULM(vocab_size=tokenizer.vocab_size, d_embed=8, d_hidden=32,
                 pad_id=tokenizer.pad_id)


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


class TestRNNLM:
    def test_output_shape(self, rnn_model, tokenizer):
        ids = tokenizer.encode("hello", add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)
        logits = rnn_model(x)
        assert logits.shape == (1, len(ids), tokenizer.vocab_size)

    def test_batch_forward(self, rnn_model, tokenizer):
        texts = ["hello", "world"]
        max_len = max(len(t) for t in texts) + 1
        batch = tokenizer.encode_batch(texts, add_bos=True, pad_to=max_len)
        x = torch.tensor(batch, dtype=torch.long)
        logits = rnn_model(x)
        assert logits.shape == (2, max_len, tokenizer.vocab_size)

    def test_parameter_count(self, rnn_model):
        count = rnn_model.count_parameters()
        assert 0 < count < 100_000

    def test_training_reduces_loss(self, rnn_model, tokenizer):
        inputs, targets = _make_train_data(tokenizer)
        loader = make_dataset(inputs, targets, batch_size=16)
        result = train(rnn_model, loader, epochs=5, lr=1e-2, verbose=False)
        assert result.epoch_losses[-1] < result.epoch_losses[0]

    def test_generate(self, rnn_model, tokenizer):
        prompt_ids = tokenizer.encode("ADD", add_bos=True)
        output_ids = generate(rnn_model, prompt_ids, max_new_tokens=5)
        assert len(output_ids) == len(prompt_ids) + 5

    def test_multi_layer(self, tokenizer):
        model = RNNLM(vocab_size=tokenizer.vocab_size, d_embed=8, d_hidden=32,
                       n_layers=2, pad_id=tokenizer.pad_id)
        ids = tokenizer.encode("test", add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)
        logits = model(x)
        assert logits.shape == (1, len(ids), tokenizer.vocab_size)


class TestGRULM:
    def test_output_shape(self, gru_model, tokenizer):
        ids = tokenizer.encode("hello", add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)
        logits = gru_model(x)
        assert logits.shape == (1, len(ids), tokenizer.vocab_size)

    def test_parameter_count(self, gru_model):
        count = gru_model.count_parameters()
        assert 0 < count < 100_000

    def test_training_reduces_loss(self, gru_model, tokenizer):
        inputs, targets = _make_train_data(tokenizer)
        loader = make_dataset(inputs, targets, batch_size=16)
        result = train(gru_model, loader, epochs=5, lr=1e-2, verbose=False)
        assert result.epoch_losses[-1] < result.epoch_losses[0]

    def test_generate(self, gru_model, tokenizer):
        prompt_ids = tokenizer.encode("ADD", add_bos=True)
        output_ids = generate(gru_model, prompt_ids, max_new_tokens=5)
        assert len(output_ids) == len(prompt_ids) + 5

    def test_gru_has_more_params_than_rnn(self, tokenizer):
        rnn = RNNLM(vocab_size=tokenizer.vocab_size, d_embed=8, d_hidden=32,
                     pad_id=tokenizer.pad_id)
        gru = GRULM(vocab_size=tokenizer.vocab_size, d_embed=8, d_hidden=32,
                     pad_id=tokenizer.pad_id)
        # GRU has 3x the recurrent params (3 gates vs 1 transform)
        assert gru.count_parameters() > rnn.count_parameters()


class TestRNNAgent:
    def test_agent_returns_answer(self, rnn_model, tokenizer):
        agent = RNNAgent(rnn_model, tokenizer, "test_rnn")
        result = agent.run("ADD 5 3 =")
        assert isinstance(result.answer, str)

    def test_agent_name(self, rnn_model, tokenizer):
        agent = RNNAgent(rnn_model, tokenizer, "my_rnn")
        assert agent.name == "my_rnn"
