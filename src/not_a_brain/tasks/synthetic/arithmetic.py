"""Arithmetic task: ADD/SUB/MUL on small numbers."""

from not_a_brain.tasks.base import TaskBase, TaskSample, GradeResult


class ArithmeticTask(TaskBase):
    """Generates simple arithmetic problems.

    Examples:
        "ADD 12 37 =" -> "49"
        "SUB 50 23 =" -> "27"
        "MUL 6 7 ="   -> "42"
    """

    def __init__(self, seed: int | None = None, max_digits: int = 2,
                 ops: tuple[str, ...] = ("ADD", "SUB", "MUL")):
        super().__init__(seed)
        self.max_digits = max_digits
        self.ops = ops
        self._max_val = 10 ** max_digits - 1

    @property
    def name(self) -> str:
        return "arithmetic"

    @property
    def description(self) -> str:
        return f"Integer arithmetic ({', '.join(self.ops)}) up to {self.max_digits} digits"

    def _generate(self) -> TaskSample:
        op = self.rng.choice(self.ops)
        a = self.rng.randint(0, self._max_val)
        b = self.rng.randint(0, self._max_val)

        if op == "ADD":
            result = a + b
        elif op == "SUB":
            # Keep results non-negative for simplicity
            if b > a:
                a, b = b, a
            result = a - b
        elif op == "MUL":
            result = a * b
        else:
            raise ValueError(f"Unknown op: {op}")

        prompt = f"{op} {a} {b} ="
        expected = str(result)

        return TaskSample(
            prompt=prompt,
            expected=expected,
            metadata={"op": op, "a": a, "b": b, "result": result,
                      "digits": self.max_digits},
        )

    def _grade(self, answer: str, sample: TaskSample) -> GradeResult:
        # Allow whitespace differences
        clean = answer.strip()
        correct = clean == sample.expected
        # Partial credit: check if numeric and close
        score = 1.0 if correct else 0.0
        if not correct:
            try:
                parsed = int(clean)
                expected_val = sample.metadata["result"]
                # Partial credit for being close (within 10%)
                if expected_val != 0:
                    relative_err = abs(parsed - expected_val) / abs(expected_val)
                    score = max(0.0, 1.0 - relative_err)
                elif parsed == 0:
                    score = 1.0
            except (ValueError, ZeroDivisionError):
                score = 0.0

        return GradeResult(correct=correct, score=score, expected=sample.expected)
