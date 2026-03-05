"""Shared training loop for all chapters."""

from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainResult:
    losses: list[float] = field(default_factory=list)
    epoch_losses: list[float] = field(default_factory=list)
    model: nn.Module | None = None


def make_dataset(inputs: torch.Tensor, targets: torch.Tensor,
                 batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from input/target tensors."""
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    loss_fn: nn.Module | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Standard training loop for tiny language models.

    Args:
        model: PyTorch model with forward(x) -> logits
        train_loader: DataLoader yielding (input, target) batches
        epochs: number of training epochs
        lr: learning rate
        device: "cpu" or "cuda"
        loss_fn: loss function (defaults to CrossEntropyLoss)
        verbose: print epoch summaries

    Returns:
        TrainResult with per-step and per-epoch losses
    """
    model = model.to(device)
    model.train()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    result = TrainResult(model=model)

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            inputs, targets = [b.to(device) for b in batch]

            logits = model(inputs)

            # Handle different output shapes:
            # (batch, seq, vocab) -> reshape for cross entropy
            if logits.dim() == 3:
                B, S, V = logits.shape
                logits = logits.reshape(B * S, V)
                targets = targets.reshape(B * S)

            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            result.losses.append(loss.item())
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        result.epoch_losses.append(avg_loss)

        if verbose:
            print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

    model.eval()
    return result


@torch.no_grad()
def generate(model: nn.Module, prompt_ids: list[int], max_new_tokens: int = 20,
             temperature: float = 1.0, device: str = "cpu") -> list[int]:
    """Autoregressive generation from a language model.

    Args:
        model: model with forward(x) -> logits of shape (batch, seq, vocab)
        prompt_ids: starting token IDs
        max_new_tokens: how many tokens to generate
        temperature: sampling temperature (0 = greedy)

    Returns:
        Full sequence (prompt + generated tokens)
    """
    model.eval()
    ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)
        # Take logits for the last position
        next_logits = logits[0, -1, :]

        if temperature <= 0:
            next_id = next_logits.argmax().item()
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        ids.append(next_id)

    return ids
