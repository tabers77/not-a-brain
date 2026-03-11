"""Tests for Chapter 13: Advanced Reasoning — ReAct, ToT, MCTS."""

import torch
import pytest

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM
from not_a_brain.evals.harness import AgentResult
from not_a_brain.utils.training import train, make_dataset

import importlib.util
from pathlib import Path

_ch13_path = (Path(__file__).parent.parent
              / "chapters" / "13_advanced_reasoning" / "run.py")
_spec = importlib.util.spec_from_file_location("ch13_run", _ch13_path)
_ch13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ch13)

tool_calc = _ch13.tool_calc
tool_lookup = _ch13.tool_lookup
BM25Retriever = _ch13.BM25Retriever
KNOWLEDGE_BASE = _ch13.KNOWLEDGE_BASE
ReActAgent = _ch13.ReActAgent
ToTAgent = _ch13.ToTAgent
MCTSAgent = _ch13.MCTSAgent
SFTAgent = _ch13.SFTAgent
ProcessRewardModel = _ch13.ProcessRewardModel
build_raw_corpus = _ch13.build_raw_corpus
build_react_corpus = _ch13.build_react_corpus
build_cot_corpus = _ch13.build_cot_corpus
build_prm_training_data = _ch13.build_prm_training_data
train_prm = _ch13.train_prm
prepare_sequences = _ch13.prepare_sequences
_make_cot_reasoning = _ch13._make_cot_reasoning
_extract_answer = _ch13._extract_answer


# -- Tool functions --------------------------------------------------------


class TestToolCalc:
    def test_add(self):
        assert tool_calc("ADD 5 3") == "8"

    def test_sub(self):
        assert tool_calc("SUB 10 4") == "6"

    def test_mul(self):
        assert tool_calc("MUL 7 8") == "56"

    def test_error(self):
        assert tool_calc("bad") == "ERROR"


class TestToolLookup:
    @pytest.fixture
    def retriever(self):
        return BM25Retriever([e["text"] for e in KNOWLEDGE_BASE])

    def test_finds_known(self, retriever):
        assert tool_lookup("capital of france?", retriever) == "paris"

    def test_not_found(self, retriever):
        assert tool_lookup("zzzzz", retriever) == "NOT_FOUND"


# -- CoT reasoning ---------------------------------------------------------


class TestMakeCoTReasoning:
    def test_arithmetic(self):
        r = _make_cot_reasoning("arithmetic", "ADD 5 3 =", "8")
        assert "ADD" in r and "8" in r

    def test_unknown(self):
        r = _make_cot_reasoning("unknown", "Q: capital of Moon?", "unknown")
        assert "unanswerable" in r

    def test_returns_string(self):
        r = _make_cot_reasoning("copy", "COPY: abc|", "abc")
        assert isinstance(r, str)


# -- Extract answer --------------------------------------------------------


class TestExtractAnswer:
    def test_basic(self):
        assert _extract_answer("INST: foo ANS: bar") == "bar"

    def test_with_think(self):
        assert _extract_answer("INST: foo THINK: blah ANS: bar") == "bar"

    def test_trims_eos(self):
        assert _extract_answer("ANS: bar<EOS>stuff") == "bar"


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


@pytest.fixture
def retriever():
    return BM25Retriever([e["text"] for e in KNOWLEDGE_BASE])


class TestCorpusBuilders:
    def test_raw_nonempty(self, tasks):
        assert len(build_raw_corpus(tasks, 5)) > 0

    def test_react_has_act(self, tasks, retriever):
        corpus = build_react_corpus(tasks, 5, retriever)
        has_act = any("ACT:" in s for s in corpus)
        assert has_act

    def test_react_has_observe(self, tasks, retriever):
        corpus = build_react_corpus(tasks, 5, retriever)
        has_obs = any("OBSERVE:" in s for s in corpus)
        assert has_obs

    def test_react_has_think(self, tasks, retriever):
        corpus = build_react_corpus(tasks, 5, retriever)
        has_think = any("THINK:" in s for s in corpus)
        assert has_think

    def test_cot_has_think(self, tasks):
        corpus = build_cot_corpus(tasks, 5)
        assert any("THINK:" in s for s in corpus)

    def test_cot_has_answer(self, tasks):
        corpus = build_cot_corpus(tasks, 5)
        for item in corpus:
            assert " ANS: " in item


# -- Process Reward Model --------------------------------------------------


class TestProcessRewardModel:
    @pytest.fixture
    def tokenizer(self):
        tok = CharTokenizer()
        tok.fit(["INST: ADD 1 2 = THINK: op is ADD ANS: 3",
                 "INST: COPY: ab| THINK: copy ANS: ab"])
        return tok

    def test_output_shape(self, tokenizer):
        prm = ProcessRewardModel(
            vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2,
            n_layers=1, d_ff=32, max_seq_len=64, pad_id=tokenizer.pad_id,
        )
        ids = tokenizer.encode("INST: ADD 1 2 = THINK: x ANS: 3", add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)
        score = prm(x)
        assert score.shape == (1,)
        assert 0.0 <= score.item() <= 1.0

    def test_score_method(self, tokenizer):
        prm = ProcessRewardModel(
            vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2,
            n_layers=1, d_ff=32, max_seq_len=64, pad_id=tokenizer.pad_id,
        )
        ids = tokenizer.encode("INST: test", add_bos=True)
        score = prm.score(ids)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_count_parameters(self, tokenizer):
        prm = ProcessRewardModel(
            vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2,
            n_layers=1, d_ff=32, max_seq_len=64, pad_id=tokenizer.pad_id,
        )
        assert prm.count_parameters() > 0


class TestPRMTrainingData:
    def test_builds_data(self, tasks):
        tok = CharTokenizer()
        tok.fit(["INST: ADD 1 2 = THINK: x ANS: 3", "INST: COPY: a| ANS: a"])
        seqs, labels = build_prm_training_data(tasks, 3, tok, max_len=64)
        assert seqs.shape[0] == labels.shape[0]
        assert seqs.shape[0] > 0
        # Should have both positive and negative labels
        assert labels.sum() > 0
        assert (1 - labels).sum() > 0


# -- Prepare sequences -----------------------------------------------------


class TestPrepareSequences:
    def test_returns_tensors(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        inp, tgt = prepare_sequences(["abc"], tok)
        assert isinstance(inp, torch.Tensor)

    def test_max_len(self):
        tok = CharTokenizer()
        tok.fit(["a" * 100])
        inp, _ = prepare_sequences(["a" * 100], tok, max_len=10)
        assert inp.shape[1] <= 10


# -- Agents ----------------------------------------------------------------

@pytest.fixture
def trained_model():
    """Tiny model trained on ReAct + CoT corpus for agent tests."""
    corpus = [
        "INST: ADD 1 2 = THINK: need calculator ACT: calc(ADD 1 2)"
        " OBSERVE: 3 THINK: result is 3 ANS: 3",
        "INST: Q: capital of france? THINK: look up ACT: lookup(capital of france?)"
        " OBSERVE: paris THINK: answer is paris ANS: paris",
        "INST: COPY: ab| THINK: solve directly ANS: ab",
        "INST: ADD 2 3 = THINK: op is ADD, 2+3=5 ANS: 5",
        "INST: Q: capital of Moon? THINK: unanswerable ANS: unknown",
    ]
    tok = CharTokenizer()
    tok.fit(corpus)
    model = TransformerLM(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1,
        d_ff=32, max_seq_len=128, dropout=0.0, pad_id=tok.pad_id,
    )
    encoded = []
    for text in corpus:
        ids = tok.encode(text, add_bos=True, add_eos=True)
        encoded.append(ids)
    pad_len = max(len(s) for s in encoded)
    inputs, targets = [], []
    for ids in encoded:
        padded = ids + [tok.pad_id] * (pad_len - len(ids))
        inputs.append(padded[:-1])
        targets.append(padded[1:])
    inp = torch.tensor(inputs, dtype=torch.long)
    tgt = torch.tensor(targets, dtype=torch.long)
    loader = make_dataset(inp, tgt, batch_size=4)
    train(model, loader, epochs=30, lr=3e-3, verbose=False)
    return model, tok


@pytest.fixture
def prm(trained_model):
    """Tiny PRM for MCTS agent tests."""
    _, tok = trained_model
    prm = ProcessRewardModel(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2,
        n_layers=1, d_ff=32, max_seq_len=128, pad_id=tok.pad_id,
    )
    return prm


class TestSFTAgent:
    def test_returns_result(self, trained_model):
        model, tok = trained_model
        agent = SFTAgent(model, tok, "sft")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)

    def test_name(self, trained_model):
        model, tok = trained_model
        assert SFTAgent(model, tok, "x").name == "x"


class TestReActAgent:
    def test_returns_result(self, trained_model):
        model, tok = trained_model
        retriever = BM25Retriever([e["text"] for e in KNOWLEDGE_BASE])
        agent = ReActAgent(model, tok, retriever, model_name="react_test")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_has_trace(self, trained_model):
        model, tok = trained_model
        retriever = BM25Retriever([e["text"] for e in KNOWLEDGE_BASE])
        agent = ReActAgent(model, tok, retriever)
        result = agent.run("ADD 1 2 =")
        assert result.trace is not None
        assert len(result.trace) > 0

    def test_name(self, trained_model):
        model, tok = trained_model
        retriever = BM25Retriever([e["text"] for e in KNOWLEDGE_BASE])
        assert ReActAgent(model, tok, retriever, model_name="r").name == "r"


class TestToTAgent:
    def test_returns_result(self, trained_model):
        model, tok = trained_model
        agent = ToTAgent(model, tok, n_branches=2, top_k=1,
                         model_name="tot_test")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_has_trace(self, trained_model):
        model, tok = trained_model
        agent = ToTAgent(model, tok, n_branches=2, top_k=1)
        result = agent.run("ADD 1 2 =")
        assert result.trace is not None
        trace_str = " ".join(result.trace)
        assert "winner" in trace_str or "branches" in trace_str

    def test_name(self, trained_model):
        model, tok = trained_model
        assert ToTAgent(model, tok, model_name="t").name == "t"


class TestMCTSAgent:
    def test_returns_result(self, trained_model, prm):
        model, tok = trained_model
        agent = MCTSAgent(model, tok, prm, n_iterations=2,
                          model_name="mcts_test")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_has_trace(self, trained_model, prm):
        model, tok = trained_model
        agent = MCTSAgent(model, tok, prm, n_iterations=2)
        result = agent.run("ADD 1 2 =")
        assert result.trace is not None
        trace_str = " ".join(result.trace)
        assert "iterations" in trace_str

    def test_name(self, trained_model, prm):
        model, tok = trained_model
        assert MCTSAgent(model, tok, prm, model_name="m").name == "m"

    def test_leaf_results(self, trained_model, prm):
        model, tok = trained_model
        agent = MCTSAgent(model, tok, prm, n_iterations=4)
        result = agent.run("ADD 1 2 =")
        trace_str = " ".join(result.trace)
        assert "leaf answers" in trace_str
