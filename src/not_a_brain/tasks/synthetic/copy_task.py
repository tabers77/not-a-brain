"""Copy task: reproduce the input sequence exactly."""

import string
from not_a_brain.tasks.base import TaskBase, TaskSample


class CopyTask(TaskBase):
    """Generates copy tasks — the model must reproduce the input.

    Examples:
        "COPY: abcd|" -> "abcd"
        "COPY: hello world|" -> "hello world"

    Tests: basic sequence memory and reproduction.
    """

    def __init__(self, seed: int | None = None, min_len: int = 2,
                 max_len: int = 10, charset: str = "alpha"):
        super().__init__(seed)
        self.min_len = min_len
        self.max_len = max_len
        if charset == "alpha":
            self.chars = string.ascii_lowercase
        elif charset == "digits":
            self.chars = string.digits
        elif charset == "alphanumeric":
            self.chars = string.ascii_lowercase + string.digits
        else:
            self.chars = charset

    @property
    def name(self) -> str:
        return "copy"

    @property
    def description(self) -> str:
        return f"Copy input sequence (len {self.min_len}-{self.max_len})"

    def _generate(self) -> TaskSample:
        length = self.rng.randint(self.min_len, self.max_len)
        seq = "".join(self.rng.choice(self.chars) for _ in range(length))
        prompt = f"COPY: {seq}|"
        return TaskSample(
            prompt=prompt,
            expected=seq,
            metadata={"length": length, "sequence": seq},
        )
