"""Tests for Chapter 08: Preference Optimization (Toy DPO)."""

import copy
import torch
import pytest

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM
from not_a_brain.evals.harness import AgentResult
from not_a_brain.tasks import ArithmeticTask, CopyTask, UnknownTask
from not_a_brain.utils.training import train, make_dataset

import importlib.util
from pathlib import Path

_ch08_path = Path(__file__).parent.parent / "chapters" / "08_preference_and_rlhf" / "run.py"
_spec = importlib.util.spec_from_file_location("ch08_run", _ch08_path)
_ch08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ch08)

build_raw_corpus = _ch08.build_raw_corpus
build_instruction_corpus = _ch08.build_instruction_corpus
build_preference_pairs = _ch08.build_preference_pairs
prepare_sequences = _ch08.prepare_sequences
SFTAgent = _ch08.SFTAgent
dpo_loss = _ch08.dpo_loss
train_dpo = _ch08.train_dpo
_sequence_logprobs = _ch08._sequence_logprobs
_encode_preference_batch = _ch08._encode_preference_batch
INST_PREFIX = _ch08.INST_PREFIX
ANS_MARKER = _ch08.ANS_MARKER


@pytest.fixture
def small_tasks():
    return {
        "arithmetic": ArithmeticTask(seed=42),
        "copy": CopyTask(seed=42),
        "unknown": UnknownTask(seed=42),
    }


@pytest.fixture
def trained_sft(small_tasks):
    """Pre-train + SFT a small model for testing."""
    raw_corpus = build_raw_corpus(small_tasks, n=20)
    inst_corpus = build_instruction_corpus(small_tasks, n=20)
    tokenizer = CharTokenizer()
    tokenizer.fit(raw_corpus + inst_corpus)

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16, n_heads=2, n_layers=1, d_ff=32,
        max_seq_len=64, dropout=0.0, pad_id=tokenizer.pad_id,
    )
    # Pre-train
    raw_in, raw_tgt = prepare_sequences(raw_corpus, tokenizer, max_len=30)
    raw_loader = make_dataset(raw_in, raw_tgt, batch_size=8)
    train(model, raw_loader, epochs=5, lr=3e-3, verbose=False)
    # SFT
    inst_in, inst_tgt = prepare_sequences(inst_corpus, tokenizer, max_len=30)
    inst_loader = make_dataset(inst_in, inst_tgt, batch_size=8)
    train(model, inst_loader, epochs=3, lr=1e-3, verbose=False)

    return model, tokenizer


# ── Preference pair building ──────────────────────────────────────────

class TestPreferencePairs:
    def test_pairs_have_correct_format(self, small_tasks):
        pairs = build_preference_pairs(small_tasks, n=5)
        assert len(pairs) == 15  # 5 per task x 3 tasks
        for prompt, chosen, rejected in pairs:
            assert INST_PREFIX in chosen
            assert ANS_MARKER in chosen
            assert INST_PREFIX in rejected
            assert ANS_MARKER in rejected

    def test_chosen_differs_from_rejected(self, small_tasks):
        pairs = build_preference_pairs(small_tasks, n=10)
        for _, chosen, rejected in pairs:
            assert chosen != rejected

    def test_unknown_task_chosen_is_unknown(self):
        tasks = {"unknown": UnknownTask(seed=42)}
        pairs = build_preference_pairs(tasks, n=5)
        for _, chosen, _ in pairs:
            assert chosen.endswith("unknown")


# ── Sequence logprobs ─────────────────────────────────────────────────

class TestSequenceLogprobs:
    def test_returns_batch_tensor(self, trained_sft):
        model, tokenizer = trained_sft
        text = f"{INST_PREFIX}ADD 1 2 ={ANS_MARKER}3"
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        batch = torch.tensor([ids, ids], dtype=torch.long)
        lps = _sequence_logprobs(model, batch, tokenizer.pad_id)
        assert lps.shape == (2,)
        assert torch.isfinite(lps).all()

    def test_logprobs_are_negative(self, trained_sft):
        model, tokenizer = trained_sft
        text = f"{INST_PREFIX}COPY: ab|{ANS_MARKER}ab"
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        batch = torch.tensor([ids], dtype=torch.long)
        lps = _sequence_logprobs(model, batch, tokenizer.pad_id)
        assert (lps < 0).all()


# ── DPO loss ──────────────────────────────────────────────────────────

class TestDPOLoss:
    def test_loss_is_scalar(self, trained_sft, small_tasks):
        model, tokenizer = trained_sft
        ref = copy.deepcopy(model)
        pairs = build_preference_pairs(small_tasks, n=3)
        chosen_ids, rejected_ids = _encode_preference_batch(
            pairs, tokenizer, max_len=30)
        loss = dpo_loss(model, ref, chosen_ids, rejected_ids,
                        tokenizer.pad_id, beta=0.1)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_loss_near_log2_at_init(self, trained_sft, small_tasks):
        """When policy == reference, DPO loss ≈ log(2) ≈ 0.693."""
        model, tokenizer = trained_sft
        ref = copy.deepcopy(model)
        pairs = build_preference_pairs(small_tasks, n=10)
        chosen_ids, rejected_ids = _encode_preference_batch(
            pairs, tokenizer, max_len=30)
        loss = dpo_loss(model, ref, chosen_ids, rejected_ids,
                        tokenizer.pad_id, beta=0.1)
        # Should be approximately log(2) when policy == reference
        assert abs(loss.item() - 0.693) < 0.15


# ── DPO training ─────────────────────────────────────────────────────

class TestDPOTraining:
    def test_dpo_reduces_loss(self, trained_sft, small_tasks):
        model, tokenizer = trained_sft
        ref = copy.deepcopy(model)
        policy = copy.deepcopy(model)
        pairs = build_preference_pairs(small_tasks, n=20)
        losses = train_dpo(policy, ref, pairs, tokenizer,
                           epochs=5, lr=5e-4, beta=0.1,
                           batch_size=8, max_len=30, verbose=False)
        assert losses[-1] < losses[0]

    def test_dpo_does_not_modify_reference(self, trained_sft, small_tasks):
        model, tokenizer = trained_sft
        ref = copy.deepcopy(model)
        ref_params = {n: p.clone() for n, p in ref.named_parameters()}
        policy = copy.deepcopy(model)
        pairs = build_preference_pairs(small_tasks, n=10)
        train_dpo(policy, ref, pairs, tokenizer,
                  epochs=3, lr=5e-4, beta=0.1,
                  batch_size=8, max_len=30, verbose=False)
        for name, param in ref.named_parameters():
            assert torch.equal(param, ref_params[name]), \
                f"Reference param '{name}' was modified during DPO"


# ── Agent ─────────────────────────────────────────────────────────────

class TestDPOAgent:
    def test_agent_produces_output(self, trained_sft):
        model, tokenizer = trained_sft
        agent = SFTAgent(model, tokenizer, "test_dpo", max_gen=10)
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)
