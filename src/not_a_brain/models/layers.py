"""Attention layers: single-head, multi-head, and a simple attention-based LM.

This is the key building block for the Transformer (Chapter 05). Here we
implement attention in isolation so we can visualize what it learns and
compare it against RNNs on tasks that require "looking back" at the input.

Scaled dot-product attention:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from not_a_brain.evals.harness import AgentInterface, AgentResult
from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.utils.training import generate


class SingleHeadAttention(nn.Module):
    """Single-head scaled dot-product attention.

    Given input X of shape (batch, seq, d_model):
        Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V
        attn = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        d_model: input/output dimension
        d_k: key/query dimension (defaults to d_model)
        causal: if True, apply causal mask (can't attend to future positions)
    """

    def __init__(self, d_model: int, d_k: int | None = None, causal: bool = True):
        super().__init__()
        self.d_k = d_k or d_model
        self.causal = causal

        self.W_Q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.d_k, bias=False)

        self._last_attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
        Returns:
            output: (batch, seq, d_k)
        """
        Q = self.W_Q(x)  # (B, S, d_k)
        K = self.W_K(x)  # (B, S, d_k)
        V = self.W_V(x)  # (B, S, d_k)

        # Scaled dot-product: (B, S, d_k) @ (B, d_k, S) -> (B, S, S)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)

        if self.causal:
            S = x.size(1)
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (B, S, S)
        self._last_attn_weights = attn_weights.detach()

        return torch.bmm(attn_weights, V)  # (B, S, d_k)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return the last computed attention weights for visualization."""
        return self._last_attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention: run multiple attention heads in parallel, then combine.

    Each head can attend to different patterns. The outputs are concatenated
    and projected back to d_model.

    Args:
        d_model: input/output dimension
        n_heads: number of attention heads
        causal: if True, apply causal mask
    """

    def __init__(self, d_model: int, n_heads: int = 4, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.causal = causal

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self._last_attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
        Returns:
            output: (batch, seq, d_model)
        """
        B, S, _ = x.shape

        # Project and reshape to (B, n_heads, S, d_k)
        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product per head: (B, H, S, d_k) @ (B, H, d_k, S) -> (B, H, S, S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.causal:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (B, H, S, S)
        self._last_attn_weights = attn_weights.detach()

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, H, S, d_k)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.W_O(out)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return last attention weights: (batch, n_heads, seq, seq)."""
        return self._last_attn_weights


class AttentionLM(nn.Module):
    """Simple attention-based language model.

    Embedding + positional encoding + multi-head attention + output projection.
    No feed-forward layers, no layer norm, no residual connections —
    those come in Chapter 05 (Transformer). This isolates attention's contribution.

    Args:
        vocab_size: number of tokens
        d_model: model dimension
        n_heads: number of attention heads
        max_seq_len: maximum sequence length (for positional encoding)
        pad_id: padding token ID
    """

    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 4,
                 max_seq_len: int = 256, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, causal=True)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token IDs
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, S = x.shape
        # Clamp positions to max_seq_len to avoid index errors during generation
        positions = torch.arange(S, device=x.device).clamp(max=self.max_seq_len - 1)
        positions = positions.unsqueeze(0).expand(B, S)

        emb = self.embedding(x) + self.pos_encoding(positions)
        attn_out = self.attention(emb)
        return self.output(attn_out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return last attention weights from the attention layer."""
        return self.attention.get_attention_weights()


class AttentionAgent(AgentInterface):
    """Wraps an attention-based LM as an agent for the eval harness."""

    def __init__(self, model: AttentionLM, tokenizer: CharTokenizer,
                 model_name: str = "attention_lm", max_gen: int = 30,
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
