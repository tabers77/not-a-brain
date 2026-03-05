"""Tests for the eval harness and metrics."""

from not_a_brain.evals.harness import RandomAgent, evaluate, run_eval_suite
from not_a_brain.evals.metrics import (
    compute_accuracy, compute_abstention_rate,
    compute_hallucination_rate, compute_all_metrics,
)
from not_a_brain.tasks import ArithmeticTask, CopyTask, UnknownTask


class TestMetrics:
    def test_accuracy_all_correct(self):
        results = [{"correct": True} for _ in range(10)]
        assert compute_accuracy(results) == 1.0

    def test_accuracy_none_correct(self):
        results = [{"correct": False} for _ in range(10)]
        assert compute_accuracy(results) == 0.0

    def test_accuracy_half(self):
        results = [{"correct": i % 2 == 0} for i in range(10)]
        assert compute_accuracy(results) == 0.5

    def test_abstention_rate(self):
        results = [
            {"abstained": True},
            {"abstained": False},
            {"abstained": True},
        ]
        assert abs(compute_abstention_rate(results) - 2 / 3) < 0.01

    def test_hallucination_rate(self):
        results = [
            {"is_unanswerable": True, "correct": False},  # hallucinated
            {"is_unanswerable": True, "correct": True},   # correctly abstained
            {"is_unanswerable": False, "correct": True},   # normal task
        ]
        assert compute_hallucination_rate(results) == 0.5

    def test_compute_all(self):
        results = [
            {"correct": True, "score": 1.0, "abstained": False,
             "is_unanswerable": False, "task_name": "arithmetic"},
            {"correct": False, "score": 0.0, "abstained": False,
             "is_unanswerable": False, "task_name": "arithmetic"},
        ]
        metrics = compute_all_metrics(results)
        assert metrics.accuracy == 0.5
        assert metrics.n_samples == 2
        assert "arithmetic" in metrics.per_task

    def test_empty_results(self):
        metrics = compute_all_metrics([])
        assert metrics.accuracy == 0.0
        assert metrics.n_samples == 0


class TestEvalHarness:
    def test_random_agent(self):
        agent = RandomAgent()
        task = ArithmeticTask(seed=42)
        results = evaluate(agent, task, n_samples=10)
        assert len(results) == 10
        assert all("correct" in r for r in results)
        assert all("answer" in r for r in results)

    def test_eval_suite(self):
        agent = RandomAgent()
        tasks = {
            "arithmetic": ArithmeticTask(seed=42),
            "copy": CopyTask(seed=42),
        }
        metrics, results = run_eval_suite(agent, tasks, n_per_task=5)
        assert metrics.n_samples == 10
        assert len(results) == 10
