"""N-gram language models: bigram and trigram.

Pure counting models — no neural network, no gradients.
These are the simplest possible "language models": predict the next token
based on the previous 1 or 2 tokens.
"""

from collections import Counter, defaultdict
from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.evals.harness import AgentInterface, AgentResult


class BigramModel:
    """Predicts next character based on the previous one.

    P(c_t | c_{t-1}) = count(c_{t-1}, c_t) / count(c_{t-1})
    """

    def __init__(self, tokenizer: CharTokenizer):
        self.tokenizer = tokenizer
        self.counts: dict[int, Counter] = defaultdict(Counter)
        self.totals: dict[int, int] = defaultdict(int)

    def train(self, texts: list[str]) -> None:
        """Count bigram frequencies from training texts."""
        for text in texts:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            for i in range(len(ids) - 1):
                self.counts[ids[i]][ids[i + 1]] += 1
                self.totals[ids[i]] += 1

    def predict_next(self, prev_id: int) -> int:
        """Return the most likely next token given previous token."""
        if prev_id not in self.counts:
            return self.tokenizer.eos_id
        return self.counts[prev_id].most_common(1)[0][0]

    def generate(self, prompt: str, max_len: int = 50) -> str:
        """Generate text by repeatedly predicting next character."""
        ids = self.tokenizer.encode(prompt, add_bos=True)
        for _ in range(max_len):
            next_id = self.predict_next(ids[-1])
            if next_id == self.tokenizer.eos_id:
                break
            ids.append(next_id)
        return self.tokenizer.decode(ids)


class TrigramModel:
    """Predicts next character based on the previous two.

    P(c_t | c_{t-2}, c_{t-1}) = count(c_{t-2}, c_{t-1}, c_t) / count(c_{t-2}, c_{t-1})
    """

    def __init__(self, tokenizer: CharTokenizer):
        self.tokenizer = tokenizer
        self.counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
        self.totals: dict[tuple[int, int], int] = defaultdict(int)
        # Fallback bigram for unseen contexts
        self.bigram = BigramModel(tokenizer)

    def train(self, texts: list[str]) -> None:
        """Count trigram frequencies from training texts."""
        self.bigram.train(texts)
        for text in texts:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            for i in range(len(ids) - 2):
                context = (ids[i], ids[i + 1])
                self.counts[context][ids[i + 2]] += 1
                self.totals[context] += 1

    def predict_next(self, prev2: int, prev1: int) -> int:
        """Return the most likely next token given previous two tokens."""
        context = (prev2, prev1)
        if context in self.counts:
            return self.counts[context].most_common(1)[0][0]
        # Fall back to bigram
        return self.bigram.predict_next(prev1)

    def generate(self, prompt: str, max_len: int = 50) -> str:
        """Generate text by repeatedly predicting next character."""
        ids = self.tokenizer.encode(prompt, add_bos=True)
        if len(ids) < 2:
            ids = [self.tokenizer.bos_id] + ids
        for _ in range(max_len):
            next_id = self.predict_next(ids[-2], ids[-1])
            if next_id == self.tokenizer.eos_id:
                break
            ids.append(next_id)
        return self.tokenizer.decode(ids)


class NgramAgent(AgentInterface):
    """Wraps an n-gram model as an agent for the eval harness.

    Strategy: given a task prompt, generate a continuation and extract
    the answer from it.
    """

    def __init__(self, model, model_name: str = "ngram", max_gen: int = 30):
        self.model = model
        self._name = model_name
        self.max_gen = max_gen

    @property
    def name(self) -> str:
        return self._name

    def run(self, prompt: str) -> AgentResult:
        generated = self.model.generate(prompt, max_len=self.max_gen)
        # The answer is the part after the prompt
        if generated.startswith(prompt):
            answer = generated[len(prompt):].strip()
        else:
            answer = generated.strip()
        return AgentResult(answer=answer, confidence=0.5)
