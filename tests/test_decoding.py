"""Tests for decoding strategies (Chapter 09)."""

import torch
import pytest

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM
from not_a_brain.models.decoding import (
    decode, score_sequence, STRATEGIES,
    _apply_temperature, _apply_top_k, _apply_top_p,
)
from not_a_brain.utils.training import train, make_dataset


@pytest.fixture
def small_model():
    """A tiny trained model for decoding tests."""
    corpus = ["ADD 1 2 =3", "COPY: ab|ab", "ADD 3 4 =7", "COPY: xy|xy"]
    tokenizer = CharTokenizer()
    tokenizer.fit(corpus)

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16, n_heads=2, n_layers=1, d_ff=32,
        max_seq_len=64, dropout=0.0, pad_id=tokenizer.pad_id,
    )
    # Quick training
    encoded = []
    for text in corpus:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        encoded.append(ids)
    pad_len = max(len(s) for s in encoded)
    inputs, targets = [], []
    for ids in encoded:
        padded = ids + [tokenizer.pad_id] * (pad_len - len(ids))
        inputs.append(padded[:-1])
        targets.append(padded[1:])
    inp = torch.tensor(inputs, dtype=torch.long)
    tgt = torch.tensor(targets, dtype=torch.long)
    loader = make_dataset(inp, tgt, batch_size=4)
    train(model, loader, epochs=10, lr=3e-3, verbose=False)

    return model, tokenizer


# ── Low-level helpers ─────────────────────────────────────────────────

class TestApplyTemperature:
    def test_identity_at_1(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = _apply_temperature(logits, 1.0)
        assert torch.allclose(result, logits)

    def test_sharper_at_low_temp(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        sharp = _apply_temperature(logits, 0.5)
        assert sharp[2] > logits[2]  # scaled up


class TestApplyTopK:
    def test_keeps_top_k(self):
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        result = _apply_top_k(logits, k=2)
        # Only top 2 (indices 1, 4) should remain finite
        finite_mask = result > float("-inf")
        assert finite_mask.sum() == 2
        assert finite_mask[1] and finite_mask[4]

    def test_no_op_when_k_exceeds_vocab(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = _apply_top_k(logits, k=100)
        assert torch.equal(result, logits)


class TestApplyTopP:
    def test_keeps_nucleus(self):
        logits = torch.tensor([0.0, 0.0, 10.0])  # ~all mass on index 2
        result = _apply_top_p(logits, p=0.9)
        # Index 2 should survive, others should be -inf
        assert result[2] > float("-inf")

    def test_no_op_at_p_1(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = _apply_top_p(logits, p=1.0)
        assert torch.equal(result, logits)


# ── Decode function ───────────────────────────────────────────────────

class TestDecode:
    def test_greedy_deterministic(self, small_model):
        model, tokenizer = small_model
        prompt_ids = tokenizer.encode("ADD 1 2 =", add_bos=True)
        out1 = decode(model, prompt_ids, max_new_tokens=5, temperature=0.0)
        out2 = decode(model, prompt_ids, max_new_tokens=5, temperature=0.0)
        assert out1 == out2

    def test_output_length(self, small_model):
        model, tokenizer = small_model
        prompt_ids = tokenizer.encode("ADD", add_bos=True)
        n_gen = 10
        out = decode(model, prompt_ids, max_new_tokens=n_gen)
        assert len(out) == len(prompt_ids) + n_gen

    def test_top_k_runs(self, small_model):
        model, tokenizer = small_model
        prompt_ids = tokenizer.encode("COPY:", add_bos=True)
        out = decode(model, prompt_ids, max_new_tokens=5,
                     temperature=1.0, top_k=3)
        assert len(out) == len(prompt_ids) + 5

    def test_top_p_runs(self, small_model):
        model, tokenizer = small_model
        prompt_ids = tokenizer.encode("COPY:", add_bos=True)
        out = decode(model, prompt_ids, max_new_tokens=5,
                     temperature=1.0, top_p=0.9)
        assert len(out) == len(prompt_ids) + 5

    def test_high_temp_more_diverse(self, small_model):
        """High temperature should produce more varied outputs."""
        model, tokenizer = small_model
        prompt_ids = tokenizer.encode("ADD", add_bos=True)
        low_temp_outs = set()
        high_temp_outs = set()
        for _ in range(10):
            out_low = decode(model, prompt_ids, max_new_tokens=5,
                             temperature=0.3)
            out_high = decode(model, prompt_ids, max_new_tokens=5,
                              temperature=2.0)
            low_temp_outs.add(tuple(out_low))
            high_temp_outs.add(tuple(out_high))
        # High temp should have at least as many unique outputs
        assert len(high_temp_outs) >= len(low_temp_outs)


# ── Score sequence ────────────────────────────────────────────────────

class TestScoreSequence:
    def test_returns_float(self, small_model):
        model, tokenizer = small_model
        ids = tokenizer.encode("ADD 1 2 =3", add_bos=True, add_eos=True)
        score = score_sequence(model, ids)
        assert isinstance(score, float)

    def test_score_is_negative(self, small_model):
        """Log-probs should be negative."""
        model, tokenizer = small_model
        ids = tokenizer.encode("ADD 1 2 =3", add_bos=True, add_eos=True)
        score = score_sequence(model, ids)
        assert score < 0

    def test_trained_text_scores_higher(self, small_model):
        """Text from training data should score higher than random text."""
        model, tokenizer = small_model
        train_ids = tokenizer.encode("ADD 1 2 =3", add_bos=True, add_eos=True)
        random_ids = tokenizer.encode("zzzzzzzzz", add_bos=True, add_eos=True)
        assert score_sequence(model, train_ids) > score_sequence(model, random_ids)


# ── Named strategies ──────────────────────────────────────────────────

class TestStrategies:
    def test_all_strategies_have_required_keys(self):
        for name, cfg in STRATEGIES.items():
            assert "temperature" in cfg
            assert "top_k" in cfg
            assert "top_p" in cfg
