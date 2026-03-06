"""Recurrent language models: vanilla RNN and GRU.

Unlike the fixed-window FFN, recurrent models process one token at a time
and maintain a hidden state that (in theory) captures the entire history.
No explicit context window — the state IS the memory.

The key question: can a compressed state vector actually remember what matters?

Architecture:
    RNN:  h_t = tanh(W_ih * e_t + W_hh * h_{t-1} + b)
    GRU:  h_t = gated update of h_{t-1} with input e_t
    Output: logits = W_out * h_t
"""

import torch
import torch.nn as nn

from not_a_brain.evals.harness import AgentInterface, AgentResult
from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.utils.training import generate


class RNNLM(nn.Module):
    """Vanilla RNN language model.

    h_t = tanh(W_ih * e_t + W_hh * h_{t-1} + b)
    logits_t = W_out * h_t

    Args:
        vocab_size: number of tokens
        d_embed: embedding dimension
        d_hidden: hidden state dimension
        n_layers: number of stacked RNN layers
        pad_id: padding token ID
    """

    def __init__(self, vocab_size: int, d_embed: int = 16, d_hidden: int = 64,
                 n_layers: int = 1, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        self.rnn = nn.RNN(d_embed, d_hidden, num_layers=n_layers, batch_first=True)
        self.output = nn.Linear(d_hidden, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len) token IDs
            hidden: optional initial hidden state (n_layers, batch, d_hidden)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, S = x.shape
        if hidden is None:
            hidden = torch.zeros(self.n_layers, B, self.d_hidden, device=x.device)

        emb = self.embedding(x)  # (B, S, d_embed)
        rnn_out, _ = self.rnn(emb, hidden)  # (B, S, d_hidden)
        logits = self.output(rnn_out)  # (B, S, vocab_size)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRULM(nn.Module):
    """GRU language model.

    The GRU uses gating to control information flow, making it easier to
    learn long-range dependencies than vanilla RNN (less vanishing gradient).

    Gates:
        z_t = sigmoid(W_z * [h_{t-1}, e_t])     (update gate: how much to keep old state)
        r_t = sigmoid(W_r * [h_{t-1}, e_t])     (reset gate: how much of old state to expose)
        h_t = (1-z_t) * h_{t-1} + z_t * tanh(W * [r_t * h_{t-1}, e_t])

    Args:
        vocab_size: number of tokens
        d_embed: embedding dimension
        d_hidden: hidden state dimension
        n_layers: number of stacked GRU layers
        dropout: dropout between layers (only if n_layers > 1)
        pad_id: padding token ID
    """

    def __init__(self, vocab_size: int, d_embed: int = 16, d_hidden: int = 64,
                 n_layers: int = 1, dropout: float = 0.0, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        self.gru = nn.GRU(d_embed, d_hidden, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.output = nn.Linear(d_hidden, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len) token IDs
            hidden: optional initial hidden state (n_layers, batch, d_hidden)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, S = x.shape
        if hidden is None:
            hidden = torch.zeros(self.n_layers, B, self.d_hidden, device=x.device)

        emb = self.embedding(x)  # (B, S, d_embed)
        gru_out, _ = self.gru(emb, hidden)  # (B, S, d_hidden)
        logits = self.output(gru_out)  # (B, S, vocab_size)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RNNAgent(AgentInterface):
    """Wraps an RNN/GRU language model as an agent for the eval harness."""

    def __init__(self, model: nn.Module, tokenizer: CharTokenizer,
                 model_name: str = "rnn_lm", max_gen: int = 30,
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
        if generated.startswith(prompt):
            answer = generated[len(prompt):].strip()
        else:
            answer = generated.strip()
        return AgentResult(answer=answer, confidence=0.5)
