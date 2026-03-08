"""Transformer language model: the full architecture.

Assembles the building blocks from previous chapters into the architecture
behind GPT, BERT, and every modern LLM:

    TransformerBlock = LayerNorm + MultiHeadAttention + Residual
                     + LayerNorm + FeedForward + Residual

    TransformerLM = Embedding + Positional Encoding
                  + N x TransformerBlock
                  + LayerNorm + Output Projection

What's new vs Chapter 04 (attention-only):
    - Feed-forward layers: compute complex functions on attended information
    - Residual connections: enable depth without vanishing gradients
    - Layer normalization: stabilize training of deep networks
    - Multiple layers: build compositional features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from not_a_brain.evals.harness import AgentInterface, AgentResult
from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.utils.training import generate


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    This is where computation happens. Attention retrieves information;
    the FFN processes it. The expansion factor (typically 4x) gives the
    network a wider intermediate representation to work with.

    Args:
        d_model: input/output dimension
        d_ff: hidden dimension (defaults to 4 * d_model)
        dropout: dropout rate
    """

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block: attention + feed-forward with residuals and layer norm.

    Uses pre-norm architecture (GPT-2 style):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Pre-norm is more stable for training than post-norm (original Transformer).

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        d_ff: feed-forward hidden dimension
        dropout: dropout rate
        max_seq_len: maximum sequence length for causal mask
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int | None = None,
                 dropout: float = 0.1, max_seq_len: int = 256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with dropout.

    Same math as Chapter 04's MultiHeadAttention, but with:
    - Attention dropout (regularization)
    - Residual dropout (applied by caller via residual connection)
    - Stores attention weights for visualization

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        dropout: dropout rate
        max_seq_len: maximum sequence length
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1,
                 max_seq_len: int = 256):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # Pre-compute causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

        self._last_attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:S, :S].unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        self._last_attn_weights = attn_weights.detach()
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.W_O(out)

    def get_attention_weights(self) -> torch.Tensor | None:
        return self._last_attn_weights


class TransformerLM(nn.Module):
    """Transformer language model (decoder-only, GPT-style).

    Architecture:
        Token Embedding + Positional Embedding
        -> N x TransformerBlock
        -> LayerNorm
        -> Linear (to vocab)

    Args:
        vocab_size: number of tokens
        d_model: model dimension
        n_heads: number of attention heads per block
        n_layers: number of transformer blocks
        d_ff: feed-forward hidden dimension (defaults to 4 * d_model)
        max_seq_len: maximum sequence length
        dropout: dropout rate
        pad_id: padding token ID
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int | None = None,
                 max_seq_len: int = 256, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        # Weight tying: share embedding and output weights
        self.output.weight = self.tok_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token IDs
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, S = x.shape
        positions = torch.arange(S, device=x.device).clamp(max=self.max_seq_len - 1)
        positions = positions.unsqueeze(0).expand(B, S)

        tok_emb = self.tok_embedding(x)
        pos_emb = self.pos_embedding(positions)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.output(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_weights(self, layer: int = -1) -> torch.Tensor | None:
        """Return attention weights from a specific layer (default: last)."""
        return self.blocks[layer].attn.get_attention_weights()


class TransformerAgent(AgentInterface):
    """Wraps a Transformer LM as an agent for the eval harness."""

    def __init__(self, model: TransformerLM, tokenizer: CharTokenizer,
                 model_name: str = "transformer_lm", max_gen: int = 30,
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
