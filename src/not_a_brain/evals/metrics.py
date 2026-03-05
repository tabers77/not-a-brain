"""Evaluation metrics: accuracy, calibration, abstention, hallucination rate."""

from dataclasses import dataclass
import numpy as np


@dataclass
class EvalMetrics:
    accuracy: float
    avg_score: float
    abstention_rate: float
    hallucination_rate: float
    n_samples: int
    calibration_error: float | None = None
    per_task: dict | None = None


def compute_accuracy(results: list[dict]) -> float:
    """Fraction of correct answers."""
    if not results:
        return 0.0
    return sum(1 for r in results if r["correct"]) / len(results)


def compute_abstention_rate(results: list[dict]) -> float:
    """Fraction of responses that abstained."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("abstained", False)) / len(results)


def compute_hallucination_rate(results: list[dict]) -> float:
    """Fraction of wrong answers on unanswerable questions.

    Only counts results where the task is marked as unanswerable.
    A hallucination = answered confidently when should have abstained.
    """
    unanswerable = [r for r in results if r.get("is_unanswerable", False)]
    if not unanswerable:
        return 0.0
    hallucinated = sum(1 for r in unanswerable if not r["correct"])
    return hallucinated / len(unanswerable)


def compute_calibration_error(results: list[dict], n_bins: int = 10) -> float:
    """Expected calibration error (ECE).

    Groups predictions by confidence, measures gap between confidence and accuracy.
    """
    confidences = [r.get("confidence", 1.0 if r["correct"] else 0.0) for r in results]
    corrects = [r["correct"] for r in results]

    if not results:
        return 0.0

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = [(bins[i] <= c < bins[i + 1]) for c in confidences]
        n_in_bin = sum(mask)
        if n_in_bin == 0:
            continue
        avg_conf = np.mean([c for c, m in zip(confidences, mask) if m])
        avg_acc = np.mean([int(c) for c, m in zip(corrects, mask) if m])
        ece += abs(avg_conf - avg_acc) * (n_in_bin / len(results))

    return float(ece)


def compute_all_metrics(results: list[dict]) -> EvalMetrics:
    """Compute all metrics from a list of evaluation results.

    Each result dict should have at minimum:
        - correct: bool
        - score: float
    Optional fields:
        - abstained: bool
        - is_unanswerable: bool
        - confidence: float
        - task_name: str
    """
    if not results:
        return EvalMetrics(
            accuracy=0.0, avg_score=0.0, abstention_rate=0.0,
            hallucination_rate=0.0, n_samples=0,
        )

    # Per-task breakdown
    per_task: dict[str, list[dict]] = {}
    for r in results:
        task_name = r.get("task_name", "default")
        per_task.setdefault(task_name, []).append(r)

    per_task_metrics = {}
    for task_name, task_results in per_task.items():
        per_task_metrics[task_name] = {
            "accuracy": compute_accuracy(task_results),
            "n_samples": len(task_results),
        }

    return EvalMetrics(
        accuracy=compute_accuracy(results),
        avg_score=np.mean([r["score"] for r in results]),
        abstention_rate=compute_abstention_rate(results),
        hallucination_rate=compute_hallucination_rate(results),
        n_samples=len(results),
        calibration_error=compute_calibration_error(results),
        per_task=per_task_metrics,
    )
