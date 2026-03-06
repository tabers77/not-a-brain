"""Feed-forward (MLP) language model.

Fixed-window neural language model: concatenate embeddings of the last N
characters, pass through hidden layers, predict the next character.

This is the first neural model in the progression. Unlike n-grams (counting),
it learns distributed representations and can generalize within its context
window — but it's still blind to anything outside that window.

Architecture:
    input (context_window chars) -> Embedding -> concat -> Linear -> ReLU -> Linear -> softmax
"""

import torch
import torch.nn as nn

from not_a_brain.evals.harness import AgentInterface, AgentResult
from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.utils.training import generate


class FFNLM(nn.Module):
    """Fixed-window feed-forward language model.

    P(c_t | c_{t-W}, ..., c_{t-1}) = softmax(W2 * ReLU(W1 * [e_{t-W}; ...; e_{t-1}] + b1) + b2)

    Args:
        vocab_size: number of tokens
        context_window: number of previous characters to condition on
        d_embed: embedding dimension per character
        d_hidden: hidden layer dimension
        n_layers: number of hidden layers (default 1)
        pad_id: padding token ID (for masking)
    """

    def __init__(self, vocab_size: int, context_window: int = 8,
                 d_embed: int = 16, d_hidden: int = 64, n_layers: int = 1,
                 pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_window = context_window
        self.d_embed = d_embed
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)

        # Build MLP: input is concatenated embeddings
        input_dim = context_window * d_embed
        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(prev_dim, d_hidden), nn.ReLU()])
            prev_dim = d_hidden
        layers.append(nn.Linear(prev_dim, vocab_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sliding context window.

        Args:
            x: (batch, seq_len) token IDs — any length

        Returns:
            logits: (batch, seq_len, vocab_size) — prediction at each position.
            At position t, the model uses tokens [t-W, ..., t-1] as context
            (left-padded with zeros for early positions).
        """
        B, S = x.shape
        emb = self.embedding(x)  # (B, S, d_embed)

        # Pad the beginning so position 0 still has a full window
        pad = torch.zeros(B, self.context_window, self.d_embed, device=x.device)
        padded = torch.cat([pad, emb], dim=1)  # (B, W+S, d_embed)

        # Extract sliding windows: for position t, take [t, t+W) from padded
        windows = []
        for t in range(S):
            window = padded[:, t:t + self.context_window, :]  # (B, W, d_embed)
            windows.append(window.reshape(B, -1))  # (B, W*d_embed)

        stacked = torch.stack(windows, dim=1)  # (B, S, W*d_embed)
        logits = self.mlp(stacked.reshape(B * S, -1))  # (B*S, vocab)
        return logits.reshape(B, S, self.vocab_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FFNAgent(AgentInterface):
    """Wraps a feed-forward LM as an agent for the eval harness."""

    def __init__(self, model: FFNLM, tokenizer: CharTokenizer,
                 model_name: str = "ffn_lm", max_gen: int = 30,
                 temperature: float = 0.0):
        self.model = model
        self.tokenizer = tokenizer
        self._name = model_name
        self.max_gen = max_gen
        self.temperature = temperature

    @property
    def name(self) -> str:
        return self._name

    def run(self, prompt: str) -> AgentResult:
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        full_ids = generate(self.model, prompt_ids,
                            max_new_tokens=self.max_gen,
                            temperature=self.temperature)
        generated = self.tokenizer.decode(full_ids)
        # Extract answer: part after the original prompt
        if generated.startswith(prompt):
            answer = generated[len(prompt):].strip()
        else:
            answer = generated.strip()
        return AgentResult(answer=answer, confidence=0.5)
