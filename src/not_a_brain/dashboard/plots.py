"""Dashboard plot generators — produce matplotlib figures for the HTML report."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import io
import base64


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def plot_evolution_curve(data: dict) -> str:
    """Line chart: task accuracy across chapters.

    Args:
        data: {chapter_name: {task_name: score}}
    Returns:
        base64-encoded PNG
    """
    chapters = list(data.keys())
    if not chapters:
        return ""

    task_names = sorted({t for scores in data.values() for t in scores})
    fig, ax = plt.subplots(figsize=(12, 5))

    for task_name in task_names:
        scores = [data[ch].get(task_name, 0.0) for ch in chapters]
        ax.plot(range(len(chapters)), scores, marker="o", linewidth=2,
                markersize=6, label=task_name)

    ax.set_xticks(range(len(chapters)))
    ax.set_xticklabels(chapters, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Capability Evolution Across Chapters")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_hallucination_bar(data: dict) -> str:
    """Bar chart: hallucination rate per chapter.

    Args:
        data: {chapter_name: hallucination_rate}
    """
    if not data:
        return ""
    chapters = list(data.keys())
    rates = [data[ch] for ch in chapters]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#e74c3c" if r > 0.5 else "#f39c12" if r > 0.2 else "#2ecc71"
              for r in rates]
    ax.bar(range(len(chapters)), rates, color=colors, edgecolor="white")
    ax.set_xticks(range(len(chapters)))
    ax.set_xticklabels(chapters, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate on Unanswerable Questions")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_cognitive_heatmap(data: dict) -> str:
    """Heatmap: 6 cognitive ingredients x chapters.

    Args:
        data: {chapter_name: {ingredient: score (0-1)}}
    """
    ingredients = [
        "Persistent Memory", "Working Memory", "Grounding",
        "Agency/Goals", "Verification", "Learning from Interaction",
    ]
    chapters = list(data.keys())
    if not chapters:
        return ""

    matrix = np.zeros((len(chapters), len(ingredients)))
    for i, ch in enumerate(chapters):
        for j, ing in enumerate(ingredients):
            matrix[i, j] = data[ch].get(ing, 0.0)

    fig, ax = plt.subplots(figsize=(10, max(4, len(chapters) * 0.4)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(ingredients)))
    ax.set_xticklabels(ingredients, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(chapters)))
    ax.set_yticklabels(chapters, fontsize=9)
    ax.set_title("Cognitive Ingredients by Chapter")

    # Add text annotations
    for i in range(len(chapters)):
        for j in range(len(ingredients)):
            val = matrix[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Score")
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_calibration(data: list[dict]) -> str:
    """Calibration plot: confidence vs actual accuracy.

    Args:
        data: list of {confidence, correct} dicts
    """
    if not data:
        return ""

    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_confs = []
    bin_accs = []

    for i in range(n_bins):
        in_bin = [d for d in data if bins[i] <= d.get("confidence", 0.5) < bins[i + 1]]
        if in_bin:
            avg_conf = np.mean([d.get("confidence", 0.5) for d in in_bin])
            avg_acc = np.mean([int(d["correct"]) for d in in_bin])
            bin_confs.append(avg_conf)
            bin_accs.append(avg_acc)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    if bin_confs:
        ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="#3498db",
               edgecolor="white", label="Model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_comparison_radar(agents: dict[str, dict[str, float]]) -> str:
    """Radar chart comparing agents across metrics.

    Args:
        agents: {agent_name: {metric_name: score}}
    """
    if not agents:
        return ""

    metrics = sorted({m for scores in agents.values() for m in scores})
    n = len(metrics)
    if n < 3:
        return ""

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(agents)))

    for (agent_name, scores), color in zip(agents.items(), colors):
        values = [scores.get(m, 0) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=agent_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Agent Comparison", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)
