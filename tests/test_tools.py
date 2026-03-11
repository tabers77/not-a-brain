"""Tests for Chapter 11: Tools & Function Calling."""

import torch
import pytest

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM
from not_a_brain.evals.harness import AgentResult
from not_a_brain.utils.training import train, make_dataset

import importlib.util
from pathlib import Path

_ch11_path = Path(__file__).parent.parent / "chapters" / "11_tools_and_function_calls" / "run.py"
_spec = importlib.util.spec_from_file_location("ch11_run", _ch11_path)
_ch11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ch11)

tool_calc = _ch11.tool_calc
tool_lookup = _ch11.tool_lookup
BM25Retriever = _ch11.BM25Retriever
ToolAgent = _ch11.ToolAgent
OracleToolAgent = _ch11.OracleToolAgent
SFTAgent = _ch11.SFTAgent
build_raw_corpus = _ch11.build_raw_corpus
build_sft_corpus = _ch11.build_sft_corpus
build_tool_corpus = _ch11.build_tool_corpus
prepare_sequences = _ch11.prepare_sequences
KNOWLEDGE_BASE = _ch11.KNOWLEDGE_BASE
_tokenize_query = _ch11._tokenize_query


# -- Tool functions --------------------------------------------------------


class TestToolCalc:
    def test_add(self):
        assert tool_calc("ADD 5 3") == "8"

    def test_sub(self):
        assert tool_calc("SUB 10 4") == "6"

    def test_mul(self):
        assert tool_calc("MUL 7 8") == "56"

    def test_invalid_returns_error(self):
        assert tool_calc("bad input") == "ERROR"

    def test_too_few_args_returns_error(self):
        assert tool_calc("ADD 5") == "ERROR"

    def test_non_numeric_returns_error(self):
        assert tool_calc("ADD x y") == "ERROR"

    def test_large_numbers(self):
        assert tool_calc("ADD 99 99") == "198"

    def test_zero(self):
        assert tool_calc("ADD 0 0") == "0"


class TestToolLookup:
    @pytest.fixture
    def retriever(self):
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        return BM25Retriever(kb_texts)

    def test_finds_known_fact(self, retriever):
        result = tool_lookup("capital of france?", retriever)
        assert result == "paris"

    def test_returns_answer_string(self, retriever):
        result = tool_lookup("capital of japan?", retriever)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unrelated_query_returns_not_found(self, retriever):
        result = tool_lookup("zzzzz qqqq", retriever)
        assert result == "NOT_FOUND"

    def test_moon_question_returns_something(self, retriever):
        # "capital of Moon" matches "capital" keyword -- returns a capital
        result = tool_lookup("capital of the Moon?", retriever)
        # BM25 keyword match on "capital" should return some answer
        assert isinstance(result, str)


# -- BM25 Retriever (reused from Ch10, minimal tests) ---------------------


class TestBM25Retriever:
    @pytest.fixture
    def retriever(self):
        docs = [
            "paris is capital of france",
            "tokyo is capital of japan",
            "water has formula H2O",
        ]
        return BM25Retriever(docs)

    def test_retrieve_returns_list(self, retriever):
        results = retriever.retrieve("capital of france", top_k=2)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_relevant_doc_ranked_first(self, retriever):
        results = retriever.retrieve("capital of france", top_k=3)
        top_idx = results[0][0]
        assert top_idx == 0  # "paris is capital of france"

    def test_scores_descending(self, retriever):
        results = retriever.retrieve("capital", top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# -- Corpus builders -------------------------------------------------------

@pytest.fixture
def tasks():
    from not_a_brain.tasks import ArithmeticTask, CopyTask, KnowledgeQATask, UnknownTask
    return {
        "arithmetic": ArithmeticTask(seed=42),
        "copy": CopyTask(seed=42),
        "knowledge_qa": KnowledgeQATask(seed=42),
        "unknown": UnknownTask(seed=42),
    }


@pytest.fixture
def retriever():
    kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
    return BM25Retriever(kb_texts)


class TestCorpusBuilders:
    def test_raw_corpus_nonempty(self, tasks):
        corpus = build_raw_corpus(tasks, 5)
        assert len(corpus) > 0
        assert all(isinstance(s, str) for s in corpus)

    def test_sft_corpus_has_markers(self, tasks):
        corpus = build_sft_corpus(tasks, 5)
        for item in corpus:
            assert "INST: " in item
            assert " ANS: " in item

    def test_tool_corpus_has_call_markers(self, tasks, retriever):
        corpus = build_tool_corpus(tasks, 5, retriever)
        # At least some entries should have CALL: marker
        has_call = any("CALL:" in item for item in corpus)
        assert has_call

    def test_tool_corpus_has_result_markers(self, tasks, retriever):
        corpus = build_tool_corpus(tasks, 5, retriever)
        has_result = any("RESULT:" in item for item in corpus)
        assert has_result

    def test_tool_corpus_arithmetic_uses_calc(self, tasks, retriever):
        corpus = build_tool_corpus(tasks, 5, retriever)
        arith_items = [s for s in corpus if "ADD" in s or "SUB" in s or "MUL" in s]
        calc_items = [s for s in arith_items if "CALL:calc(" in s]
        assert len(calc_items) > 0

    def test_tool_corpus_knowledge_uses_lookup(self, tasks, retriever):
        corpus = build_tool_corpus(tasks, 5, retriever)
        qa_items = [s for s in corpus if "FACT:" in s]
        lookup_items = [s for s in qa_items if "CALL:lookup(" in s]
        assert len(lookup_items) > 0

    def test_tool_corpus_unknown_has_not_found(self, tasks, retriever):
        corpus = build_tool_corpus(tasks, 5, retriever)
        unknown_items = [s for s in corpus if "NOT_FOUND" in s]
        assert len(unknown_items) > 0

    def test_tool_corpus_copy_no_call(self, tasks, retriever):
        corpus = build_tool_corpus(tasks, 5, retriever)
        copy_items = [s for s in corpus if "COPY:" in s]
        for item in copy_items:
            assert "CALL:" not in item


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
def trained_model():
    """A tiny trained model for agent tests."""
    corpus = [
        "INST: ADD 1 2 = CALL:calc(ADD 1 2) RESULT:3 ANS: 3",
        "INST: Q: capital of france? CALL:lookup(capital of france?) RESULT:paris ANS: paris",
        "INST: COPY: ab| ANS: ab",
        "INST: Q: capital of Moon? CALL:lookup(capital of Moon?) RESULT:NOT_FOUND ANS: unknown",
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
    def test_returns_agent_result(self, trained_model):
        model, tokenizer = trained_model
        agent = SFTAgent(model, tokenizer, "sft_test")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_name_property(self, trained_model):
        model, tokenizer = trained_model
        agent = SFTAgent(model, tokenizer, "my_sft")
        assert agent.name == "my_sft"


class TestToolAgent:
    def test_returns_agent_result(self, trained_model):
        model, tokenizer = trained_model
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        agent = ToolAgent(model, tokenizer, retriever, model_name="tool_test")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_has_trace(self, trained_model):
        model, tokenizer = trained_model
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        agent = ToolAgent(model, tokenizer, retriever)
        result = agent.run("ADD 1 2 =")
        assert result.trace is not None
        assert len(result.trace) > 0

    def test_name_property(self, trained_model):
        model, tokenizer = trained_model
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        agent = ToolAgent(model, tokenizer, retriever, model_name="my_tool")
        assert agent.name == "my_tool"


class TestOracleToolAgent:
    def test_returns_agent_result(self, trained_model):
        model, tokenizer = trained_model
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        agent = OracleToolAgent(model, tokenizer, retriever,
                                model_name="oracle_test")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_has_trace(self, trained_model):
        model, tokenizer = trained_model
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        agent = OracleToolAgent(model, tokenizer, retriever)
        result = agent.run("Q: capital of france?")
        assert result.trace is not None
        assert len(result.trace) > 0

    def test_arithmetic_calls_calc(self, trained_model):
        model, tokenizer = trained_model
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        agent = OracleToolAgent(model, tokenizer, retriever)
        result = agent.run("ADD 5 3 =")
        assert result.trace is not None
        trace_str = " ".join(result.trace)
        assert "calc" in trace_str

    def test_question_calls_lookup(self, trained_model):
        model, tokenizer = trained_model
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        agent = OracleToolAgent(model, tokenizer, retriever)
        result = agent.run("Q: capital of france?")
        assert result.trace is not None
        trace_str = " ".join(result.trace)
        assert "lookup" in trace_str

    def test_name_property(self, trained_model):
        model, tokenizer = trained_model
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        agent = OracleToolAgent(model, tokenizer, retriever,
                                model_name="my_oracle")
        assert agent.name == "my_oracle"
