"""Visualization helpers: loss curves, attention heatmaps, comparison charts."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for script/notebook compatibility


def plot_loss_curve(losses: list[float], title: str = "Training Loss",
                    save_path: str | None = None, show: bool = False) -> plt.Figure:
    """Plot training loss over steps."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=1.5, color="#2196F3")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_attention_heatmap(attention_weights: np.ndarray, tokens: list[str],
                           title: str = "Attention",
                           save_path: str | None = None,
                           show: bool = False) -> plt.Figure:
    """Plot attention weights as a heatmap.

    Args:
        attention_weights: 2D array of shape (query_len, key_len)
        tokens: token labels for axes
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention_weights, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_comparison_bar(labels: list[str], scores: dict[str, list[float]],
                        title: str = "Model Comparison",
                        ylabel: str = "Accuracy",
                        save_path: str | None = None,
                        show: bool = False) -> plt.Figure:
    """Side-by-side bar chart comparing multiple models on tasks.

    Args:
        labels: task names (x-axis)
        scores: {model_name: [score_per_task]}
    """
    n_groups = len(labels)
    n_models = len(scores)
    width = 0.8 / n_models
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (model_name, model_scores) in enumerate(scores.items()):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, model_scores, width, label=model_name,
               color=colors[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_scaling_curve(param_counts: list[int], losses: list[float],
                       model_names: list[str] | None = None,
                       title: str = "Loss vs Parameters",
                       save_path: str | None = None,
                       show: bool = False) -> plt.Figure:
    """Plot loss vs parameter count on a log-linear scale.

    Args:
        param_counts: number of parameters per model
        losses: final loss per model
        model_names: labels for each point
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_counts, losses, marker="o", linewidth=2, markersize=8,
            color="#2196F3", zorder=3)

    if model_names:
        for i, name in enumerate(model_names):
            ax.annotate(name, (param_counts[i], losses[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, color="#333")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Final Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_evolution_curve(chapter_names: list[str],
                         task_scores: dict[str, list[float]],
                         title: str = "Capability Evolution",
                         save_path: str | None = None,
                         show: bool = False) -> plt.Figure:
    """Line chart showing how task performance evolves across chapters.

    Args:
        chapter_names: x-axis labels (chapter names)
        task_scores: {task_name: [score_per_chapter]}
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(chapter_names))

    for task_name, scores in task_scores.items():
        ax.plot(x, scores, marker="o", linewidth=2, markersize=6, label=task_name)

    ax.set_xticks(list(x))
    ax.set_xticklabels(chapter_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
