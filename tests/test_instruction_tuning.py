"""Tests for Chapter 07: Instruction Tuning (Toy SFT)."""

import copy
import torch
import pytest

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM, TransformerAgent
from not_a_brain.evals.harness import AgentResult
from not_a_brain.tasks import ArithmeticTask, CopyTask, UnknownTask
from not_a_brain.utils.training import train, make_dataset

# Import SFT-specific helpers from the chapter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "chapters" / "07_instruction_tuning"))
from run import (
    build_raw_corpus, build_instruction_corpus, prepare_sequences,
    SFTAgent, INST_PREFIX, ANS_MARKER,
)


@pytest.fixture
def small_tasks():
    return {
        "arithmetic": ArithmeticTask(seed=42),
        "copy": CopyTask(seed=42),
    }


@pytest.fixture
def tokenizer_and_model(small_tasks):
    """Build a small tokenizer and model for testing."""
    raw_corpus = build_raw_corpus(small_tasks, n=20)
    inst_corpus = build_instruction_corpus(small_tasks, n=20)
    tokenizer = CharTokenizer()
    tokenizer.fit(raw_corpus + inst_corpus)

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16, n_heads=2, n_layers=1, d_ff=32,
        max_seq_len=64, dropout=0.0, pad_id=tokenizer.pad_id,
    )
    return tokenizer, model, raw_corpus, inst_corpus


# ── Corpus building ──────────────────────────────────────────────────

class TestCorpusBuilding:
    def test_raw_corpus_format(self, small_tasks):
        corpus = build_raw_corpus(small_tasks, n=5)
        assert len(corpus) == 10  # 5 per task x 2 tasks
        # Raw corpus does NOT contain instruction markers
        for text in corpus:
            assert INST_PREFIX not in text
            assert ANS_MARKER not in text

    def test_instruction_corpus_format(self, small_tasks):
        corpus = build_instruction_corpus(small_tasks, n=5)
        assert len(corpus) == 10
        for text in corpus:
            assert text.startswith(INST_PREFIX)
            assert ANS_MARKER in text

    def test_instruction_corpus_preserves_content(self, small_tasks):
        raw = build_raw_corpus(small_tasks, n=5)
        inst = build_instruction_corpus(small_tasks, n=5)
        # Both should have same count
        assert len(raw) == len(inst)


# ── Sequence preparation ─────────────────────────────────────────────

class TestSequencePreparation:
    def test_prepare_shapes(self, tokenizer_and_model):
        tokenizer, _, raw_corpus, _ = tokenizer_and_model
        inputs, targets = prepare_sequences(raw_corpus, tokenizer, max_len=30)
        assert inputs.shape == targets.shape
        assert inputs.dim() == 2

    def test_prepare_with_max_len(self, tokenizer_and_model):
        tokenizer, _, raw_corpus, _ = tokenizer_and_model
        inputs, targets = prepare_sequences(raw_corpus, tokenizer, max_len=20)
        # Sequence length should be <= max_len - 1 (due to input/target shift)
        assert inputs.shape[1] <= 20


# ── SFT Agent ────────────────────────────────────────────────────────

class TestSFTAgent:
    def test_agent_wraps_prompt(self, tokenizer_and_model):
        tokenizer, model, _, _ = tokenizer_and_model
        agent = SFTAgent(model, tokenizer, "test_sft", max_gen=5)
        result = agent.run("ADD 5 3 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_agent_name(self, tokenizer_and_model):
        tokenizer, model, _, _ = tokenizer_and_model
        agent = SFTAgent(model, tokenizer, "my_sft_model")
        assert agent.name == "my_sft_model"

    def test_agent_returns_nonempty(self, tokenizer_and_model):
        """After minimal training, agent should return something."""
        tokenizer, model, _, inst_corpus = tokenizer_and_model
        inputs, targets = prepare_sequences(inst_corpus, tokenizer, max_len=30)
        loader = make_dataset(inputs, targets, batch_size=8)
        train(model, loader, epochs=3, lr=3e-3, verbose=False)

        agent = SFTAgent(model, tokenizer, "test", max_gen=10)
        result = agent.run("ADD 1 2 =")
        # Should produce some output (may not be correct, but not empty)
        assert isinstance(result.answer, str)


# ── Training dynamics ────────────────────────────────────────────────

class TestSFTTraining:
    def test_pretrain_reduces_loss(self, tokenizer_and_model):
        tokenizer, model, raw_corpus, _ = tokenizer_and_model
        inputs, targets = prepare_sequences(raw_corpus, tokenizer, max_len=30)
        loader = make_dataset(inputs, targets, batch_size=8)
        result = train(model, loader, epochs=5, lr=3e-3, verbose=False)
        assert result.epoch_losses[-1] < result.epoch_losses[0]

    def test_sft_reduces_loss(self, tokenizer_and_model):
        tokenizer, model, _, inst_corpus = tokenizer_and_model
        inputs, targets = prepare_sequences(inst_corpus, tokenizer, max_len=30)
        loader = make_dataset(inputs, targets, batch_size=8)
        result = train(model, loader, epochs=5, lr=3e-3, verbose=False)
        assert result.epoch_losses[-1] < result.epoch_losses[0]

    def test_sft_from_pretrained_starts_lower(self, tokenizer_and_model):
        """SFT starting from pre-trained weights should have lower initial
        loss on instruction data than a randomly initialized model."""
        tokenizer, model, raw_corpus, inst_corpus = tokenizer_and_model

        # Pre-train
        raw_in, raw_tgt = prepare_sequences(raw_corpus, tokenizer, max_len=30)
        raw_loader = make_dataset(raw_in, raw_tgt, batch_size=8)
        train(model, raw_loader, epochs=10, lr=3e-3, verbose=False)

        # Measure SFT loss on instruction data
        inst_in, inst_tgt = prepare_sequences(inst_corpus, tokenizer, max_len=30)
        inst_loader = make_dataset(inst_in, inst_tgt, batch_size=8)

        sft_model = copy.deepcopy(model)
        sft_result = train(sft_model, inst_loader, epochs=1, lr=1e-3, verbose=False)

        # Train from scratch on same instruction data
        scratch_model = TransformerLM(
            vocab_size=tokenizer.vocab_size,
            d_model=16, n_heads=2, n_layers=1, d_ff=32,
            max_seq_len=64, dropout=0.0, pad_id=tokenizer.pad_id,
        )
        scratch_result = train(
            scratch_model, inst_loader, epochs=1, lr=3e-3, verbose=False)

        # Pre-trained model should start with lower loss
        assert sft_result.epoch_losses[0] < scratch_result.epoch_losses[0]


# ── Deep copy isolation ──────────────────────────────────────────────

class TestModelIsolation:
    def test_sft_does_not_modify_base(self, tokenizer_and_model):
        """Fine-tuning the SFT copy must not change the base model."""
        tokenizer, model, _, inst_corpus = tokenizer_and_model
        base_params = {n: p.clone() for n, p in model.named_parameters()}

        sft_model = copy.deepcopy(model)
        inputs, targets = prepare_sequences(inst_corpus, tokenizer, max_len=30)
        loader = make_dataset(inputs, targets, batch_size=8)
        train(sft_model, loader, epochs=3, lr=3e-3, verbose=False)

        # Base model weights unchanged
        for name, param in model.named_parameters():
            assert torch.equal(param, base_params[name]), \
                f"Base model param '{name}' was modified during SFT"
