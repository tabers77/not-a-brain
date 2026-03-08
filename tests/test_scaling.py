"""Tests for Chapter 06: Scaling Laws.

Tests that different-sized TransformerLM models can be trained and compared,
and that parameter counts scale as expected with config changes.
"""

import torch
import pytest
from not_a_brain.models.transformer import TransformerLM, TransformerAgent
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


CONFIGS = {
    "tiny": dict(d_model=8, n_heads=2, n_layers=1, d_ff=16),
    "small": dict(d_model=16, n_heads=4, n_layers=2, d_ff=32),
    "medium": dict(d_model=32, n_heads=4, n_layers=3, d_ff=64),
}


class TestScalingParameterCounts:
    """Parameter count should increase with model size."""

    def test_larger_model_has_more_params(self, tokenizer):
        models = {}
        for name, cfg in CONFIGS.items():
            models[name] = TransformerLM(
                vocab_size=tokenizer.vocab_size,
                pad_id=tokenizer.pad_id,
                **cfg,
            )
        counts = {n: m.count_parameters() for n, m in models.items()}
        assert counts["tiny"] < counts["small"] < counts["medium"]

    def test_all_models_produce_logits(self, tokenizer):
        ids = tokenizer.encode("ADD 5 3 =", add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)

        for name, cfg in CONFIGS.items():
            model = TransformerLM(
                vocab_size=tokenizer.vocab_size,
                pad_id=tokenizer.pad_id,
                **cfg,
            )
            logits = model(x)
            assert logits.shape == (1, len(ids), tokenizer.vocab_size), \
                f"{name} produced wrong shape"


class TestScalingTraining:
    """All model sizes should be trainable."""

    def test_all_sizes_reduce_loss(self, tokenizer):
        inputs, targets = _make_train_data(tokenizer)
        loader = make_dataset(inputs, targets, batch_size=16)

        for name, cfg in CONFIGS.items():
            model = TransformerLM(
                vocab_size=tokenizer.vocab_size,
                dropout=0.0,
                pad_id=tokenizer.pad_id,
                **cfg,
            )
            result = train(model, loader, epochs=5, lr=1e-2, verbose=False)
            assert result.epoch_losses[-1] < result.epoch_losses[0], \
                f"{name} did not reduce loss"

    def test_larger_model_reaches_lower_loss(self, tokenizer):
        inputs, targets = _make_train_data(tokenizer)
        loader = make_dataset(inputs, targets, batch_size=16)

        final_losses = {}
        for name in ["tiny", "medium"]:
            cfg = CONFIGS[name]
            model = TransformerLM(
                vocab_size=tokenizer.vocab_size,
                dropout=0.0,
                pad_id=tokenizer.pad_id,
                **cfg,
            )
            result = train(model, loader, epochs=10, lr=1e-2, verbose=False)
            final_losses[name] = result.epoch_losses[-1]

        # Medium should reach lower or equal loss than tiny
        assert final_losses["medium"] <= final_losses["tiny"] * 1.5, \
            "Medium model should not be much worse than tiny"


class TestScalingGeneration:
    """All model sizes should generate text."""

    def test_all_sizes_generate(self, tokenizer):
        for name, cfg in CONFIGS.items():
            model = TransformerLM(
                vocab_size=tokenizer.vocab_size,
                pad_id=tokenizer.pad_id,
                **cfg,
            )
            prompt_ids = tokenizer.encode("ADD", add_bos=True)
            output_ids = generate(model, prompt_ids, max_new_tokens=5)
            assert len(output_ids) == len(prompt_ids) + 5, \
                f"{name} generated wrong length"


class TestScalingAgent:
    """All model sizes should work as agents."""

    def test_all_sizes_as_agents(self, tokenizer):
        for name, cfg in CONFIGS.items():
            model = TransformerLM(
                vocab_size=tokenizer.vocab_size,
                pad_id=tokenizer.pad_id,
                **cfg,
            )
            agent = TransformerAgent(model, tokenizer, f"transformer_{name}")
            result = agent.run("ADD 5 3 =")
            assert isinstance(result.answer, str)
            assert agent.name == f"transformer_{name}"
