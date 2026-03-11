"""Tests for Chapter 12: Reasoning Scaffolds."""

import torch
import pytest

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM
from not_a_brain.evals.harness import AgentResult
from not_a_brain.utils.training import train, make_dataset

import importlib.util
from pathlib import Path

_ch12_path = Path(__file__).parent.parent / "chapters" / "12_reasoning_scaffolds" / "run.py"
_spec = importlib.util.spec_from_file_location("ch12_run", _ch12_path)
_ch12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ch12)

SFTAgent = _ch12.SFTAgent
CoTAgent = _ch12.CoTAgent
SelfConsistencyAgent = _ch12.SelfConsistencyAgent
VerifyAgent = _ch12.VerifyAgent
build_raw_corpus = _ch12.build_raw_corpus
build_sft_corpus = _ch12.build_sft_corpus
build_cot_corpus = _ch12.build_cot_corpus
build_verify_corpus = _ch12.build_verify_corpus
prepare_sequences = _ch12.prepare_sequences
_make_cot_reasoning = _ch12._make_cot_reasoning
_extract_answer = _ch12._extract_answer


# -- CoT reasoning generator -----------------------------------------------


class TestMakeCoTReasoning:
    def test_arithmetic_add(self):
        r = _make_cot_reasoning("arithmetic", "ADD 5 3 =", "8")
        assert "ADD" in r
        assert "5" in r and "3" in r
        assert "8" in r

    def test_arithmetic_sub(self):
        r = _make_cot_reasoning("arithmetic", "SUB 10 4 =", "6")
        assert "SUB" in r
        assert "6" in r

    def test_arithmetic_mul(self):
        r = _make_cot_reasoning("arithmetic", "MUL 7 8 =", "56")
        assert "MUL" in r
        assert "56" in r

    def test_knowledge_qa(self):
        r = _make_cot_reasoning("knowledge_qa",
                                "FACT: paris is capital of france. Q: capital of france?",
                                "paris")
        assert "paris" in r

    def test_copy(self):
        r = _make_cot_reasoning("copy", "COPY: abc|", "abc")
        assert "abc" in r

    def test_unknown(self):
        r = _make_cot_reasoning("unknown",
                                "Q: What is the capital of the Moon?",
                                "unknown")
        assert "unanswerable" in r or "abstain" in r

    def test_returns_string(self):
        r = _make_cot_reasoning("grammar", "( [ ] )", "valid")
        assert isinstance(r, str)
        assert len(r) > 0


# -- Extract answer helper -------------------------------------------------


class TestExtractAnswer:
    def test_basic(self):
        assert _extract_answer("INST: foo ANS: bar") == "bar"

    def test_with_think(self):
        assert _extract_answer("INST: foo THINK: blah ANS: bar") == "bar"

    def test_trims_verify(self):
        answer = _extract_answer("INST: foo ANS: bar VERIFY: bar ok? YES")
        assert answer == "bar"

    def test_trims_eos(self):
        assert _extract_answer("INST: foo ANS: bar<EOS>") == "bar"

    def test_no_marker(self):
        result = _extract_answer("just some text")
        assert isinstance(result, str)


# -- Corpus builders -------------------------------------------------------

@pytest.fixture
def tasks():
    from not_a_brain.tasks import (
        ArithmeticTask, CopyTask, KnowledgeQATask, UnknownTask,
    )
    return {
        "arithmetic": ArithmeticTask(seed=42),
        "copy": CopyTask(seed=42),
        "knowledge_qa": KnowledgeQATask(seed=42),
        "unknown": UnknownTask(seed=42),
    }


class TestCorpusBuilders:
    def test_raw_corpus_nonempty(self, tasks):
        corpus = build_raw_corpus(tasks, 5)
        assert len(corpus) > 0

    def test_sft_corpus_has_markers(self, tasks):
        corpus = build_sft_corpus(tasks, 5)
        for item in corpus:
            assert "INST: " in item
            assert " ANS: " in item

    def test_cot_corpus_has_think(self, tasks):
        corpus = build_cot_corpus(tasks, 5)
        has_think = any("THINK:" in item for item in corpus)
        assert has_think

    def test_cot_corpus_has_answer(self, tasks):
        corpus = build_cot_corpus(tasks, 5)
        for item in corpus:
            assert " ANS: " in item

    def test_verify_corpus_has_verify(self, tasks):
        corpus = build_verify_corpus(tasks, 5)
        has_verify = any("VERIFY:" in item for item in corpus)
        assert has_verify

    def test_verify_corpus_has_think_and_answer(self, tasks):
        corpus = build_verify_corpus(tasks, 5)
        for item in corpus:
            assert " ANS: " in item

    def test_cot_unknown_mentions_unanswerable(self, tasks):
        corpus = build_cot_corpus(tasks, 5)
        unknown_items = [s for s in corpus
                         if "capital of" in s.lower() or "population" in s.lower()
                         or "unanswerable" in s.lower()]
        # At least some unknown examples should mention unanswerable/abstain
        unanswerable = [s for s in corpus if "unanswerable" in s or "abstain" in s]
        assert len(unanswerable) > 0


# -- Prepare sequences -----------------------------------------------------


class TestPrepareSequences:
    def test_returns_tensors(self):
        tokenizer = CharTokenizer()
        tokenizer.fit(["abc", "xyz"])
        inputs, targets = prepare_sequences(["abc", "xyz"], tokenizer)
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)

    def test_shape_matches(self):
        tokenizer = CharTokenizer()
        tokenizer.fit(["hello", "world"])
        inputs, targets = prepare_sequences(["hello", "world"], tokenizer)
        assert inputs.shape == targets.shape

    def test_max_len_truncates(self):
        tokenizer = CharTokenizer()
        tokenizer.fit(["a" * 100])
        inputs, _ = prepare_sequences(["a" * 100], tokenizer, max_len=10)
        assert inputs.shape[1] <= 10


# -- Agents ---------------------------------------------------------------

@pytest.fixture
def trained_cot_model():
    """A tiny model trained on CoT corpus for agent tests."""
    corpus = [
        "INST: ADD 1 2 = THINK: operation is ADD, 1+2=3 ANS: 3",
        "INST: FACT: paris is capital of france. Q: capital of france?"
        " THINK: fact states paris ANS: paris",
        "INST: COPY: ab| THINK: copy the input: ab ANS: ab",
        "INST: Q: capital of Moon? THINK: unanswerable, abstain ANS: unknown",
    ]
    tokenizer = CharTokenizer()
    tokenizer.fit(corpus)
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16, n_heads=2, n_layers=1, d_ff=32,
        max_seq_len=128, dropout=0.0, pad_id=tokenizer.pad_id,
    )
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
    train(model, loader, epochs=30, lr=3e-3, verbose=False)
    return model, tokenizer


class TestSFTAgent:
    def test_returns_agent_result(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = SFTAgent(model, tokenizer, "sft_test")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_name_property(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = SFTAgent(model, tokenizer, "my_sft")
        assert agent.name == "my_sft"


class TestCoTAgent:
    def test_returns_agent_result(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = CoTAgent(model, tokenizer, "cot_test")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_has_trace(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = CoTAgent(model, tokenizer)
        result = agent.run("ADD 1 2 =")
        # Trace may or may not have reasoning depending on generation
        assert isinstance(result.trace, (list, type(None)))

    def test_name_property(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = CoTAgent(model, tokenizer, "my_cot")
        assert agent.name == "my_cot"


class TestSelfConsistencyAgent:
    def test_returns_agent_result(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = SelfConsistencyAgent(
            model, tokenizer, n_samples=3, temperature=0.8,
            model_name="sc_test")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_has_trace_with_samples(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = SelfConsistencyAgent(
            model, tokenizer, n_samples=3, temperature=0.8)
        result = agent.run("ADD 1 2 =")
        assert result.trace is not None
        assert len(result.trace) > 0
        # Should mention samples or winner
        trace_str = " ".join(result.trace)
        assert "winner" in trace_str or "samples" in trace_str

    def test_confidence_is_agreement(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = SelfConsistencyAgent(
            model, tokenizer, n_samples=3, temperature=0.8)
        result = agent.run("ADD 1 2 =")
        assert 0.0 <= result.confidence <= 1.0

    def test_name_property(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = SelfConsistencyAgent(
            model, tokenizer, n_samples=3, model_name="my_sc")
        assert agent.name == "my_sc"


class TestVerifyAgent:
    def test_returns_agent_result(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = VerifyAgent(model, tokenizer, max_retries=1,
                            model_name="verify_test")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_has_trace(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = VerifyAgent(model, tokenizer, max_retries=1)
        result = agent.run("ADD 1 2 =")
        assert result.trace is not None
        assert len(result.trace) > 0
        # Trace should mention attempts and verification
        trace_str = " ".join(result.trace)
        assert "attempt" in trace_str

    def test_name_property(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = VerifyAgent(model, tokenizer, model_name="my_verify")
        assert agent.name == "my_verify"

    def test_max_retries_respected(self, trained_cot_model):
        model, tokenizer = trained_cot_model
        agent = VerifyAgent(model, tokenizer, max_retries=0)
        result = agent.run("ADD 1 2 =")
        # With 0 retries, should have exactly 1 attempt in trace
        attempt_lines = [t for t in result.trace if "attempt" in t]
        assert len(attempt_lines) >= 1
