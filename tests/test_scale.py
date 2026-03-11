"""Tests for Chapter 14: Scale Is Not Understanding."""

import torch
import pytest

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM
from not_a_brain.evals.harness import AgentResult
from not_a_brain.utils.training import train, make_dataset

import importlib.util
from pathlib import Path

_ch14_path = (Path(__file__).parent.parent
              / "chapters" / "14_scale_is_not_understanding" / "run.py")
_spec = importlib.util.spec_from_file_location("ch14_run", _ch14_path)
_ch14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ch14)

ScaleAgent = _ch14.ScaleAgent
_is_abstention = _ch14._is_abstention
build_base_corpus = _ch14.build_base_corpus
build_abstention_corpus = _ch14.build_abstention_corpus
prepare_sequences = _ch14.prepare_sequences
ABSTENTION_LEVEL_0 = _ch14.ABSTENTION_LEVEL_0
ABSTENTION_LEVEL_1 = _ch14.ABSTENTION_LEVEL_1
ABSTENTION_LEVEL_2 = _ch14.ABSTENTION_LEVEL_2
ABSTENTION_LEVEL_3 = _ch14.ABSTENTION_LEVEL_3
ABSTENTION_LEVEL_4 = _ch14.ABSTENTION_LEVEL_4
IN_DISTRIBUTION_TESTS = _ch14.IN_DISTRIBUTION_TESTS
REPHRASE_TESTS = _ch14.REPHRASE_TESTS
NOVEL_TESTS = _ch14.NOVEL_TESTS
INST_PREFIX = _ch14.INST_PREFIX
ANS_MARKER = _ch14.ANS_MARKER


# -- Abstention levels -----------------------------------------------------


class TestAbstentionLevels:
    def test_level_0_empty(self):
        assert len(ABSTENTION_LEVEL_0) == 0

    def test_level_1_has_moon(self):
        assert len(ABSTENTION_LEVEL_1) == 1
        assert "Moon" in ABSTENTION_LEVEL_1[0][0]

    def test_level_2_superset_of_1(self):
        assert len(ABSTENTION_LEVEL_2) > len(ABSTENTION_LEVEL_1)
        for pair in ABSTENTION_LEVEL_1:
            assert pair in ABSTENTION_LEVEL_2

    def test_level_3_superset_of_2(self):
        assert len(ABSTENTION_LEVEL_3) > len(ABSTENTION_LEVEL_2)
        for pair in ABSTENTION_LEVEL_2:
            assert pair in ABSTENTION_LEVEL_3

    def test_level_4_superset_of_3(self):
        assert len(ABSTENTION_LEVEL_4) > len(ABSTENTION_LEVEL_3)
        for pair in ABSTENTION_LEVEL_3:
            assert pair in ABSTENTION_LEVEL_4

    def test_all_answers_unknown(self):
        for level in [ABSTENTION_LEVEL_1, ABSTENTION_LEVEL_2,
                      ABSTENTION_LEVEL_3, ABSTENTION_LEVEL_4]:
            for _, answer in level:
                assert answer == "unknown"

    def test_level_sizes(self):
        assert len(ABSTENTION_LEVEL_1) == 1
        assert len(ABSTENTION_LEVEL_2) == 5
        assert len(ABSTENTION_LEVEL_3) == 20
        assert len(ABSTENTION_LEVEL_4) == 30


# -- Test sets -------------------------------------------------------------


class TestTestSets:
    def test_in_distribution_nonempty(self):
        assert len(IN_DISTRIBUTION_TESTS) > 0

    def test_rephrase_nonempty(self):
        assert len(REPHRASE_TESTS) > 0

    def test_novel_nonempty(self):
        assert len(NOVEL_TESTS) > 0

    def test_in_distribution_has_moon(self):
        questions = [q for q, _ in IN_DISTRIBUTION_TESTS]
        assert any("Moon" in q for q in questions)

    def test_rephrase_about_moon(self):
        questions = [q for q, _ in REPHRASE_TESTS]
        assert any("Moon" in q for q in questions)

    def test_novel_are_different(self):
        """Novel questions should not overlap with in-distribution."""
        in_dist_qs = {q for q, _ in IN_DISTRIBUTION_TESTS}
        novel_qs = {q for q, _ in NOVEL_TESTS}
        assert len(in_dist_qs & novel_qs) == 0


# -- _is_abstention --------------------------------------------------------


class TestIsAbstention:
    def test_unknown(self):
        assert _is_abstention("unknown") is True

    def test_i_dont_know(self):
        assert _is_abstention("i don't know") is True

    def test_capital_city(self):
        assert _is_abstention("paris") is False

    def test_empty(self):
        assert _is_abstention("") is False

    def test_case_insensitive(self):
        assert _is_abstention("UNKNOWN") is True

    def test_with_whitespace(self):
        assert _is_abstention("  unknown  ") is True


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


class TestBuildBaseCoprus:
    def test_nonempty(self, tasks):
        corpus = build_base_corpus(tasks, 5)
        assert len(corpus) > 0

    def test_has_inst_prefix(self, tasks):
        corpus = build_base_corpus(tasks, 5)
        for item in corpus:
            assert item.startswith(INST_PREFIX)

    def test_has_ans_marker(self, tasks):
        corpus = build_base_corpus(tasks, 5)
        for item in corpus:
            assert ANS_MARKER in item


class TestBuildAbstentionCorpus:
    def test_empty_for_level_0(self):
        assert len(build_abstention_corpus(ABSTENTION_LEVEL_0)) == 0

    def test_repeats(self):
        corpus = build_abstention_corpus(ABSTENTION_LEVEL_1, n_repeats=10)
        assert len(corpus) == 10  # 1 pattern × 10 repeats

    def test_has_unknown(self):
        corpus = build_abstention_corpus(ABSTENTION_LEVEL_1, n_repeats=2)
        for item in corpus:
            assert "unknown" in item

    def test_level_4_size(self):
        corpus = build_abstention_corpus(ABSTENTION_LEVEL_4, n_repeats=50)
        assert len(corpus) == 30 * 50


# -- Prepare sequences -----------------------------------------------------


class TestPrepareSequences:
    def test_returns_tensors(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        inp, tgt = prepare_sequences(["abc"], tok)
        assert isinstance(inp, torch.Tensor)
        assert isinstance(tgt, torch.Tensor)

    def test_max_len(self):
        tok = CharTokenizer()
        tok.fit(["a" * 100])
        inp, _ = prepare_sequences(["a" * 100], tok, max_len=10)
        assert inp.shape[1] <= 10

    def test_shape_match(self):
        tok = CharTokenizer()
        tok.fit(["hello world"])
        inp, tgt = prepare_sequences(["hello world"], tok)
        assert inp.shape == tgt.shape


# -- ScaleAgent ------------------------------------------------------------


@pytest.fixture
def trained_model():
    """Tiny model trained on base + abstention corpus for agent tests."""
    corpus = [
        f"{INST_PREFIX}ADD 1 2 ={ANS_MARKER}3",
        f"{INST_PREFIX}COPY: ab|{ANS_MARKER}ab",
        f"{INST_PREFIX}Q: What is the capital of the Moon?{ANS_MARKER}unknown",
        f"{INST_PREFIX}Q: What is the capital of Jupiter?{ANS_MARKER}unknown",
        f"{INST_PREFIX}FACT: paris is capital of france. Q: capital of france?{ANS_MARKER}paris",
    ]
    tok = CharTokenizer()
    tok.fit(corpus)
    model = TransformerLM(
        vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1,
        d_ff=32, max_seq_len=128, dropout=0.0, pad_id=tok.pad_id,
    )
    # Repeat corpus for training
    train_corpus = corpus * 20
    inp, tgt = prepare_sequences(train_corpus, tok, max_len=96)
    loader = make_dataset(inp, tgt, batch_size=8)
    train(model, loader, epochs=30, lr=3e-3, verbose=False)
    return model, tok


class TestScaleAgent:
    def test_returns_result(self, trained_model):
        model, tok = trained_model
        agent = ScaleAgent(model, tok, "test_agent")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result, AgentResult)

    def test_name(self, trained_model):
        model, tok = trained_model
        agent = ScaleAgent(model, tok, "my_agent")
        assert agent.name == "my_agent"

    def test_answer_is_string(self, trained_model):
        model, tok = trained_model
        agent = ScaleAgent(model, tok, "test")
        result = agent.run("COPY: ab|")
        assert isinstance(result.answer, str)

    def test_has_confidence(self, trained_model):
        model, tok = trained_model
        agent = ScaleAgent(model, tok, "test")
        result = agent.run("ADD 1 2 =")
        assert isinstance(result.confidence, float)


# -- Integration: coverage vs abstention -----------------------------------


class TestCoverageIntegration:
    """Test that more coverage leads to more abstention on trained questions."""

    def test_level_0_no_abstention_training(self):
        """Level 0 has no abstention patterns."""
        corpus = build_abstention_corpus(ABSTENTION_LEVEL_0)
        assert len(corpus) == 0

    def test_level_4_has_most_patterns(self):
        """Level 4 should have the most training data."""
        for level in [ABSTENTION_LEVEL_0, ABSTENTION_LEVEL_1,
                      ABSTENTION_LEVEL_2, ABSTENTION_LEVEL_3]:
            assert len(level) <= len(ABSTENTION_LEVEL_4)

    def test_abstention_corpus_format(self):
        """Each abstention example should follow INST/ANS format."""
        corpus = build_abstention_corpus(ABSTENTION_LEVEL_2, n_repeats=2)
        for item in corpus:
            assert item.startswith(INST_PREFIX)
            assert ANS_MARKER in item
            assert item.endswith("unknown")
