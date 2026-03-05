"""Tests for all synthetic tasks."""

import pytest
from not_a_brain.tasks import (
    ArithmeticTask, CopyTask, GrammarTask,
    KnowledgeQATask, CompositionalTask, UnknownTask,
)


class TestArithmeticTask:
    def test_generates_valid_samples(self):
        task = ArithmeticTask(seed=42)
        for _ in range(20):
            sample = task.generate()
            assert "=" in sample.prompt
            assert sample.expected == str(sample.metadata["result"])

    def test_grading_correct(self):
        task = ArithmeticTask(seed=42)
        sample = task.generate()
        result = task.grade(sample.expected, sample)
        assert result.correct
        assert result.score == 1.0

    def test_grading_wrong(self):
        task = ArithmeticTask(seed=42)
        sample = task.generate()
        result = task.grade("99999", sample)
        assert not result.correct

    def test_ops_filter(self):
        task = ArithmeticTask(seed=42, ops=("ADD",))
        for _ in range(10):
            sample = task.generate()
            assert sample.prompt.startswith("ADD")

    def test_training_pairs(self):
        task = ArithmeticTask(seed=42)
        pairs = task.training_pairs(10)
        assert len(pairs) == 10
        for prompt, target in pairs:
            assert "=" in prompt
            assert target.isdigit() or target.startswith("-")


class TestCopyTask:
    def test_generates_valid_samples(self):
        task = CopyTask(seed=42, min_len=3, max_len=8)
        for _ in range(20):
            sample = task.generate()
            assert sample.prompt.startswith("COPY:")
            assert sample.prompt.endswith("|")
            assert len(sample.expected) >= 3
            assert len(sample.expected) <= 8

    def test_grading_correct(self):
        task = CopyTask(seed=42)
        sample = task.generate()
        result = task.grade(sample.expected, sample)
        assert result.correct

    def test_grading_wrong(self):
        task = CopyTask(seed=42)
        sample = task.generate()
        result = task.grade("WRONG", sample)
        assert not result.correct


class TestGrammarTask:
    def test_generates_valid_and_invalid(self):
        task = GrammarTask(seed=42, invalid_ratio=0.5)
        valid_count = 0
        n = 50
        for _ in range(n):
            sample = task.generate()
            assert sample.expected in ("valid", "invalid")
            if sample.expected == "valid":
                valid_count += 1
        # Should have a mix (with 50% ratio, expect roughly half)
        assert 5 < valid_count < 45

    def test_valid_sequences_are_actually_valid(self):
        task = GrammarTask(seed=42, invalid_ratio=0.0)
        for _ in range(20):
            sample = task.generate()
            assert sample.expected == "valid"
            # Verify using our own check
            assert task._is_valid(sample.metadata["sequence"])

    def test_grading(self):
        task = GrammarTask(seed=42)
        sample = task.generate()
        result = task.grade(sample.expected, sample)
        assert result.correct


class TestKnowledgeQATask:
    def test_generates_valid_samples(self):
        task = KnowledgeQATask(seed=42)
        for _ in range(20):
            sample = task.generate()
            assert "FACT:" in sample.prompt
            assert "Q:" in sample.prompt
            assert len(sample.expected) > 0

    def test_answer_is_in_context(self):
        task = KnowledgeQATask(seed=42, n_distractors=0)
        for _ in range(20):
            sample = task.generate()
            assert sample.expected.lower() in sample.prompt.lower()

    def test_distractors(self):
        task = KnowledgeQATask(seed=42, n_distractors=3)
        sample = task.generate()
        # Should have multiple facts in context
        assert sample.prompt.count(" is ") >= 2


class TestCompositionalTask:
    def test_single_op(self):
        task = CompositionalTask(seed=42, max_ops=1)
        for _ in range(20):
            sample = task.generate()
            assert "APPLY" in sample.prompt
            assert "TO" in sample.prompt
            assert "THEN" not in sample.prompt

    def test_multi_op(self):
        task = CompositionalTask(seed=42, max_ops=2)
        found_multi = False
        for _ in range(50):
            sample = task.generate()
            if "THEN" in sample.prompt:
                found_multi = True
                break
        assert found_multi

    def test_grading(self):
        task = CompositionalTask(seed=42)
        sample = task.generate()
        result = task.grade(sample.expected, sample)
        assert result.correct
        assert result.score == 1.0


class TestUnknownTask:
    def test_all_are_unanswerable(self):
        task = UnknownTask(seed=42)
        for _ in range(10):
            sample = task.generate()
            assert sample.expected == "unknown"

    def test_abstention_is_correct(self):
        task = UnknownTask(seed=42)
        sample = task.generate()
        result = task.grade("I don't know", sample)
        assert result.correct

    def test_confident_answer_is_wrong(self):
        task = UnknownTask(seed=42)
        sample = task.generate()
        result = task.grade("42 million", sample)
        assert not result.correct

    def test_with_context(self):
        task = UnknownTask(seed=42, include_context=True)
        sample = task.generate()
        assert "CONTEXT:" in sample.prompt
