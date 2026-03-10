"""Tests for Chapter 10: RAG Minimal."""

import torch
import pytest

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM
from not_a_brain.evals.harness import AgentResult
from not_a_brain.utils.training import train, make_dataset

import importlib.util
from pathlib import Path

_ch10_path = Path(__file__).parent.parent / "chapters" / "10_rag_minimal" / "run.py"
_spec = importlib.util.spec_from_file_location("ch10_run", _ch10_path)
_ch10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ch10)

BM25Retriever = _ch10.BM25Retriever
RAGAgent = _ch10.RAGAgent
SFTAgent = _ch10.SFTAgent
build_raw_corpus = _ch10.build_raw_corpus
build_instruction_corpus = _ch10.build_instruction_corpus
build_rag_corpus = _ch10.build_rag_corpus
prepare_sequences = _ch10.prepare_sequences
KNOWLEDGE_BASE = _ch10.KNOWLEDGE_BASE
_tokenize_query = _ch10._tokenize_query
_compute_idf = _ch10._compute_idf


# ── BM25 Retriever ──────────────────────────────────────────────────


class TestTokenizeQuery:
    def test_lowercases(self):
        assert _tokenize_query("HELLO World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _tokenize_query("what? is: this!") == ["what", "is", "this"]

    def test_empty_string(self):
        assert _tokenize_query("") == []


class TestComputeIdf:
    def test_returns_dict(self):
        docs = [["a", "b"], ["b", "c"], ["c", "d"]]
        idf = _compute_idf(docs)
        assert isinstance(idf, dict)
        # "b" appears in 2/3 docs, "a" in 1/3 — "a" should have higher idf
        assert idf["a"] > idf["b"]

    def test_common_term_lower_idf(self):
        docs = [["x"], ["x"], ["x"], ["y"]]
        idf = _compute_idf(docs)
        assert idf["y"] > idf["x"]


class TestBM25Retriever:
    @pytest.fixture
    def retriever(self):
        docs = [
            "paris is capital of france",
            "tokyo is capital of japan",
            "water has formula H2O",
            "earth is third planet from sun",
        ]
        return BM25Retriever(docs)

    def test_retrieve_returns_list(self, retriever):
        results = retriever.retrieve("capital of france", top_k=2)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_retrieve_returns_tuples(self, retriever):
        results = retriever.retrieve("capital", top_k=1)
        idx, score = results[0]
        assert isinstance(idx, int)
        assert isinstance(score, float)

    def test_relevant_doc_ranked_first(self, retriever):
        results = retriever.retrieve("capital of france", top_k=4)
        top_idx = results[0][0]
        assert top_idx == 0  # "paris is capital of france"

    def test_scores_descending(self, retriever):
        results = retriever.retrieve("capital", top_k=4)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_unrelated_query_low_scores(self, retriever):
        results = retriever.retrieve("zzzzz qqqq", top_k=4)
        # All scores should be 0 for completely unrelated terms
        assert all(score == 0.0 for _, score in results)

    def test_score_method(self, retriever):
        score = retriever.score("capital of france", 0)
        assert score > 0
        # Unrelated doc should score lower
        score_unrelated = retriever.score("capital of france", 2)
        assert score > score_unrelated

    def test_knowledge_base_not_empty(self):
        assert len(KNOWLEDGE_BASE) > 0
        for entry in KNOWLEDGE_BASE:
            assert "text" in entry
            assert "answer" in entry
            assert "question" in entry


# ── Corpus builders ──────────────────────────────────────────────────

@pytest.fixture
def tasks():
    from not_a_brain.tasks import ArithmeticTask, CopyTask, KnowledgeQATask
    return {
        "arithmetic": ArithmeticTask(seed=42),
        "copy": CopyTask(seed=42),
        "knowledge_qa": KnowledgeQATask(seed=42),
    }


class TestCorpusBuilders:
    def test_raw_corpus_nonempty(self, tasks):
        corpus = build_raw_corpus(tasks, 5)
        assert len(corpus) > 0
        assert all(isinstance(s, str) for s in corpus)

    def test_instruction_corpus_has_markers(self, tasks):
        corpus = build_instruction_corpus(tasks, 5)
        for item in corpus:
            assert "INST: " in item
            assert " ANS: " in item

    def test_rag_corpus_has_context(self, tasks):
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        corpus = build_rag_corpus(tasks, 5, retriever, top_k=2)
        assert len(corpus) > 0
        # At least some entries should have CONTEXT prefix
        has_context = any("CONTEXT:" in item for item in corpus)
        assert has_context

    def test_rag_corpus_has_instruction_format(self, tasks):
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        corpus = build_rag_corpus(tasks, 5, retriever, top_k=2)
        for item in corpus:
            assert "INST: " in item
            assert " ANS: " in item


# ── Agents ───────────────────────────────────────────────────────────

@pytest.fixture
def trained_model():
    """A tiny trained model for agent tests."""
    corpus = [
        "INST: ADD 1 2 = ANS: 3",
        "INST: CONTEXT: paris is capital of france. Q: capital of france? ANS: paris",
        "INST: Q: capital of france? ANS: paris",
        "INST: COPY: ab| ANS: ab",
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
    train(model, loader, epochs=20, lr=3e-3, verbose=False)
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


class TestRAGAgent:
    def test_returns_agent_result(self, trained_model):
        model, tokenizer = trained_model
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        agent = RAGAgent(model, tokenizer, retriever, top_k=2,
                         model_name="rag_test")
        result = agent.run("capital of france?")
        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)

    def test_has_trace(self, trained_model):
        model, tokenizer = trained_model
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        agent = RAGAgent(model, tokenizer, retriever, top_k=2)
        result = agent.run("capital of france?")
        assert result.trace is not None
        assert len(result.trace) > 0

    def test_name_property(self, trained_model):
        model, tokenizer = trained_model
        kb_texts = [e["text"] for e in KNOWLEDGE_BASE]
        retriever = BM25Retriever(kb_texts)
        agent = RAGAgent(model, tokenizer, retriever, model_name="my_rag")
        assert agent.name == "my_rag"


# ── Prepare sequences ────────────────────────────────────────────────

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
        # Sequence length should be at most max_len - 1 (split into input/target)
        assert inputs.shape[1] <= 10
