"""Evaluation harness: run any agent on any task suite, collect results."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from not_a_brain.tasks.base import TaskBase, TaskSample, GradeResult
from not_a_brain.evals.metrics import compute_all_metrics, EvalMetrics


@dataclass
class AgentResult:
    answer: str
    confidence: float = 1.0
    trace: list[str] | None = None
    abstained: bool = False


class AgentInterface:
    """Base class for anything that can answer task prompts.

    Subclass this for LLM models, human-agent, baselines, etc.
    """

    def run(self, prompt: str) -> AgentResult:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class RandomAgent(AgentInterface):
    """Baseline: answers randomly from a small vocabulary."""

    def __init__(self, vocab: list[str] | None = None):
        import random
        self.rng = random
        self.vocab = vocab or ["0", "1", "yes", "no", "unknown", "42", "hello"]

    @property
    def name(self) -> str:
        return "random"

    def run(self, prompt: str) -> AgentResult:
        answer = self.rng.choice(self.vocab)
        return AgentResult(answer=answer, confidence=0.5)


def evaluate(agent: AgentInterface, task: TaskBase,
             n_samples: int = 100) -> list[dict]:
    """Run an agent on a task and collect graded results.

    Returns list of result dicts with fields: correct, score, expected,
    answer, confidence, abstained, task_name, is_unanswerable, prompt.
    """
    results = []

    for _ in range(n_samples):
        sample = task.generate()
        agent_result = agent.run(sample.prompt)
        grade = task.grade(agent_result.answer, sample)

        results.append({
            "correct": grade.correct,
            "score": grade.score,
            "expected": grade.expected,
            "answer": agent_result.answer,
            "confidence": agent_result.confidence,
            "abstained": agent_result.abstained,
            "task_name": task.name,
            "is_unanswerable": task.name == "unknown",
            "prompt": sample.prompt,
            "trace": agent_result.trace,
        })

    return results


def run_eval_suite(agent: AgentInterface,
                   tasks: dict[str, TaskBase],
                   n_per_task: int = 50) -> tuple[EvalMetrics, list[dict]]:
    """Run agent on multiple tasks, return aggregate metrics + raw results."""
    all_results = []
    for task_name, task in tasks.items():
        results = evaluate(agent, task, n_samples=n_per_task)
        all_results.extend(results)

    metrics = compute_all_metrics(all_results)
    return metrics, all_results


def save_results(results: list[dict], metrics: EvalMetrics,
                 path: str | Path, agent_name: str = "",
                 chapter: str = "") -> None:
    """Save evaluation results and metrics to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "agent": agent_name,
        "chapter": chapter,
        "metrics": asdict(metrics),
        "results": results,
    }
    path.write_text(json.dumps(output, indent=2, default=str))


def load_results(path: str | Path) -> tuple[dict, list[dict]]:
    """Load results from JSON file. Returns (metadata, results)."""
    data = json.loads(Path(path).read_text())
    return {k: v for k, v in data.items() if k != "results"}, data["results"]
