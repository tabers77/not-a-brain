"""Dashboard generator: collect chapter results and render HTML report."""

import json
from pathlib import Path
from jinja2 import Template

from not_a_brain.dashboard.plots import (
    plot_evolution_curve,
    plot_hallucination_bar,
    plot_cognitive_heatmap,
    plot_calibration,
    plot_comparison_radar,
)

# Cognitive ingredient scores per chapter (manually curated, educational)
# 0 = absent, 0.5 = partial/scaffolded, 1.0 = present
COGNITIVE_INGREDIENTS = {
    "01_ngrams": {
        "Persistent Memory": 0.0, "Working Memory": 0.0,
        "Grounding": 0.0, "Agency/Goals": 0.0,
        "Verification": 0.0, "Learning from Interaction": 0.0,
    },
    "02_ffn_lm": {
        "Persistent Memory": 0.0, "Working Memory": 0.2,
        "Grounding": 0.0, "Agency/Goals": 0.0,
        "Verification": 0.0, "Learning from Interaction": 0.0,
    },
    "03_rnn_gru": {
        "Persistent Memory": 0.0, "Working Memory": 0.4,
        "Grounding": 0.0, "Agency/Goals": 0.0,
        "Verification": 0.0, "Learning from Interaction": 0.0,
    },
    "04_attention": {
        "Persistent Memory": 0.0, "Working Memory": 0.6,
        "Grounding": 0.0, "Agency/Goals": 0.0,
        "Verification": 0.0, "Learning from Interaction": 0.0,
    },
    "05_transformer": {
        "Persistent Memory": 0.0, "Working Memory": 0.7,
        "Grounding": 0.0, "Agency/Goals": 0.0,
        "Verification": 0.0, "Learning from Interaction": 0.0,
    },
    "10_rag": {
        "Persistent Memory": 0.3, "Working Memory": 0.7,
        "Grounding": 0.5, "Agency/Goals": 0.0,
        "Verification": 0.0, "Learning from Interaction": 0.0,
    },
    "12_reasoning": {
        "Persistent Memory": 0.3, "Working Memory": 0.7,
        "Grounding": 0.5, "Agency/Goals": 0.2,
        "Verification": 0.5, "Learning from Interaction": 0.0,
    },
    "Human Agent": {
        "Persistent Memory": 1.0, "Working Memory": 1.0,
        "Grounding": 1.0, "Agency/Goals": 1.0,
        "Verification": 1.0, "Learning from Interaction": 1.0,
    },
}

TEMPLATE_PATH = Path(__file__).parent / "template.html"


def collect_results(results_dir: Path) -> list[dict]:
    """Load all results JSON files from a directory."""
    results = []
    if not results_dir.exists():
        return results
    for f in sorted(results_dir.glob("*.json")):
        data = json.loads(f.read_text())
        results.append(data)
    return results


def build_summary_table(all_results: list[dict]) -> list[dict]:
    """Build the summary metrics table from collected results."""
    rows = []
    for result in all_results:
        metrics = result.get("metrics", {})
        rows.append({
            "chapter": result.get("chapter", "unknown"),
            "agent": result.get("agent", "unknown"),
            "accuracy": metrics.get("accuracy", 0.0),
            "hallucination_rate": metrics.get("hallucination_rate", 0.0),
            "abstention_rate": metrics.get("abstention_rate", 0.0),
            "calibration_error": metrics.get("calibration_error", 0.0) or 0.0,
        })
    return rows


def build_evolution_data(all_results: list[dict]) -> dict:
    """Build chapter -> task -> score mapping for evolution curve."""
    data = {}
    for result in all_results:
        chapter = result.get("chapter", "unknown")
        per_task = result.get("metrics", {}).get("per_task", {})
        if per_task:
            data[chapter] = {t: info.get("accuracy", 0.0)
                            for t, info in per_task.items()}
    return data


def build_hallucination_data(all_results: list[dict]) -> dict:
    """Build chapter -> hallucination rate mapping."""
    data = {}
    for result in all_results:
        chapter = result.get("chapter", "unknown")
        rate = result.get("metrics", {}).get("hallucination_rate", 0.0)
        data[chapter] = rate
    return data


def build_chapter_details(all_results: list[dict]) -> list[dict]:
    """Build per-chapter detail sections."""
    details = []
    for result in all_results:
        per_task = result.get("metrics", {}).get("per_task", {})
        task_rows = []
        for task_name, info in per_task.items():
            task_rows.append({
                "task_name": task_name,
                "agent": result.get("agent", "unknown"),
                "accuracy": info.get("accuracy", 0.0),
                "n_samples": info.get("n_samples", 0),
            })
        details.append({
            "name": result.get("chapter", "unknown"),
            "title": result.get("chapter", ""),
            "description": "",
            "per_task": task_rows,
        })
    return details


def generate_dashboard(results_dir: str | Path = "results",
                       output_path: str | Path = "dashboard.html",
                       cognitive_data: dict | None = None) -> Path:
    """Generate the full HTML dashboard.

    Args:
        results_dir: directory containing chapter results JSON files
        output_path: where to write the HTML
        cognitive_data: optional override for cognitive ingredient scores

    Returns:
        Path to the generated HTML file
    """
    results_dir = Path(results_dir)
    output_path = Path(output_path)

    all_results = collect_results(results_dir)

    # Build data for plots
    summary = build_summary_table(all_results)
    evolution_data = build_evolution_data(all_results)
    hallucination_data = build_hallucination_data(all_results)
    chapter_details = build_chapter_details(all_results)

    cog_data = cognitive_data or COGNITIVE_INGREDIENTS

    # Collect all raw results for calibration plot
    all_raw = []
    for result in all_results:
        all_raw.extend(result.get("results", []))

    # Generate plots
    evolution_plot = plot_evolution_curve(evolution_data) if evolution_data else ""
    hallucination_plot = plot_hallucination_bar(hallucination_data) if hallucination_data else ""
    cognitive_heatmap = plot_cognitive_heatmap(cog_data)
    calibration_plot_b64 = plot_calibration(all_raw) if all_raw else ""

    # Build radar for agent comparison
    agent_scores = {}
    for result in all_results:
        agent = result.get("agent", "unknown")
        metrics = result.get("metrics", {})
        if agent not in agent_scores:
            agent_scores[agent] = {}
        agent_scores[agent]["Accuracy"] = metrics.get("accuracy", 0.0)
        agent_scores[agent]["1 - Hallucination"] = 1.0 - metrics.get("hallucination_rate", 0.0)
        agent_scores[agent]["Abstention"] = metrics.get("abstention_rate", 0.0)
        agent_scores[agent]["Calibration"] = 1.0 - min(metrics.get("calibration_error", 0.0) or 0.0, 1.0)

    radar_plot = plot_comparison_radar(agent_scores) if len(agent_scores) >= 2 else ""

    # Render template
    template_str = TEMPLATE_PATH.read_text()
    template = Template(template_str)
    html = template.render(
        summary_table=summary,
        evolution_plot=evolution_plot,
        hallucination_plot=hallucination_plot,
        cognitive_heatmap=cognitive_heatmap,
        calibration_plot=calibration_plot_b64,
        radar_plot=radar_plot,
        chapter_details=chapter_details,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard generated: {output_path.resolve()}")
    return output_path


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate not-a-brain dashboard")
    parser.add_argument("--results-dir", default="results",
                        help="Directory with chapter results JSON files")
    parser.add_argument("--output", default="dashboard.html",
                        help="Output HTML file path")
    args = parser.parse_args()
    generate_dashboard(args.results_dir, args.output)


if __name__ == "__main__":
    main()
