"""Decoding strategies for autoregressive language models.

Strategies:
    - greedy: always pick the highest probability token
    - temperature: scale logits before softmax
    - top_k: sample from the k most probable tokens
    - top_p (nucleus): sample from the smallest set of tokens
      whose cumulative probability exceeds p
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature. Lower = sharper, higher = flatter."""
    if temperature <= 0:
        return logits  # handled by greedy path
    return logits / temperature


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits outside the top-k tokens."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    topk_vals, _ = torch.topk(logits, k)
    threshold = topk_vals[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out tokens outside the nucleus (smallest set with cumprob >= p)."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Mask tokens whose cumulative prob (excluding themselves) exceeds p
    mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    # Unsort back to original order
    out = torch.empty_like(logits)
    out.scatter_(-1, sorted_indices, sorted_logits)
    return out


@torch.no_grad()
def decode(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: str = "cpu",
) -> list[int]:
    """Autoregressive decoding with configurable strategy.

    Args:
        model: language model with forward(x) -> logits (B, S, V)
        prompt_ids: starting token ids
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (0 = greedy)
        top_k: if > 0, only sample from top k tokens
        top_p: if < 1.0, nucleus sampling threshold
        device: torch device

    Returns:
        Full sequence (prompt + generated tokens)
    """
    model.eval()
    ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)
        next_logits = logits[0, -1, :]  # (V,)

        if temperature <= 0:
            # Greedy
            next_id = next_logits.argmax().item()
        else:
            next_logits = _apply_temperature(next_logits, temperature)
            next_logits = _apply_top_k(next_logits, top_k)
            next_logits = _apply_top_p(next_logits, top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        ids.append(next_id)

    return ids


@torch.no_grad()
def score_sequence(model: nn.Module, token_ids: list[int],
                   device: str = "cpu") -> float:
    """Compute average log-probability of a token sequence.

    Returns the mean per-token log-prob (higher = model finds it more likely).
    """
    if len(token_ids) < 2:
        return 0.0
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits = model(x[:, :-1])
    targets = x[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lps = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return token_lps.mean().item()


# ── Named strategy configs ────────────────────────────────────────────

STRATEGIES = {
    "greedy": dict(temperature=0.0, top_k=0, top_p=1.0),
    "temp_0.5": dict(temperature=0.5, top_k=0, top_p=1.0),
    "temp_1.0": dict(temperature=1.0, top_k=0, top_p=1.0),
    "temp_1.5": dict(temperature=1.5, top_k=0, top_p=1.0),
    "top_k_5": dict(temperature=1.0, top_k=5, top_p=1.0),
    "top_k_10": dict(temperature=1.0, top_k=10, top_p=1.0),
    "top_p_0.9": dict(temperature=1.0, top_k=0, top_p=0.9),
    "top_p_0.5": dict(temperature=1.0, top_k=0, top_p=0.5),
}
