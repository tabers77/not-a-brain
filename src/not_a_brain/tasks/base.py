"""Base class for all tasks in the not-a-brain framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random


@dataclass
class GradeResult:
    correct: bool
    score: float
    expected: str
    details: dict | None = None


@dataclass
class TaskSample:
    prompt: str
    expected: str
    metadata: dict


class TaskBase(ABC):
    """Base interface for all synthetic tasks.

    Every task generates prompt/answer pairs on the fly and can grade responses.
    Tasks are stateful: call generate() to produce a new sample, then grade() to score it.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._current_sample: TaskSample | None = None

    def generate(self) -> TaskSample:
        """Generate a new task sample (prompt + expected answer)."""
        self._current_sample = self._generate()
        return self._current_sample

    def generate_batch(self, n: int) -> list[TaskSample]:
        """Generate n task samples."""
        return [self.generate() for _ in range(n)]

    @abstractmethod
    def _generate(self) -> TaskSample:
        """Subclass implementation: produce one sample."""
        ...

    def grade(self, answer: str, sample: TaskSample | None = None) -> GradeResult:
        """Grade an answer against the expected output."""
        if sample is None:
            sample = self._current_sample
        if sample is None:
            raise ValueError("No sample to grade. Call generate() first.")
        return self._grade(answer.strip(), sample)

    def _grade(self, answer: str, sample: TaskSample) -> GradeResult:
        """Default grading: exact string match. Override for custom logic."""
        correct = answer == sample.expected
        return GradeResult(
            correct=correct,
            score=1.0 if correct else 0.0,
            expected=sample.expected,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable task name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """What this task tests."""
        ...

    def training_pairs(self, n: int) -> list[tuple[str, str]]:
        """Generate n (input, target) string pairs for training language models."""
        pairs = []
        for _ in range(n):
            sample = self.generate()
            pairs.append((sample.prompt, sample.expected))
        return pairs
