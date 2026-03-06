"""Chapter 00: Setup & Metrics

Runs both baselines (random agent + human agent) on the full task suite
and prints the evaluation metrics that will be used in every chapter.

Usage:
    python chapters/00_setup_and_metrics/run.py
"""

from pathlib import Path
from not_a_brain.tasks import (
    ArithmeticTask, CopyTask, GrammarTask,
    KnowledgeQATask, CompositionalTask, UnknownTask,
)
from not_a_brain.evals.harness import RandomAgent, run_eval_suite, save_results
from not_a_brain.human_agent.agent import HumanAgent
from not_a_brain.utils.visualization import plot_comparison_bar

RESULTS_DIR = Path(__file__).parent / "results"
N_EVAL = 50


def print_metrics(name: str, metrics):
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:           {metrics.accuracy:.1%}")
    print(f"  Abstention rate:    {metrics.abstention_rate:.1%}")
    print(f"  Hallucination rate: {metrics.hallucination_rate:.1%}")
    print(f"  Calibration error:  {metrics.calibration_error:.3f}")
    print(f"  Samples:            {metrics.n_samples}")
    print()
    print("  Per-task breakdown:")
    for task_name, info in metrics.per_task.items():
        print(f"    {task_name:20s} {info['accuracy']:6.1%}  (n={info['n_samples']})")


def main():
    print("=" * 60)
    print("Chapter 00: Setup & Metrics")
    print("=" * 60)
    print()
    print("This chapter establishes two baselines:")
    print("  1. Random Agent  — picks answers randomly (lower bound)")
    print("  2. Human Agent   — uses algorithms + abstention (upper bound)")
    print()
    print("See chapter.md for metric definitions and formulas.")

    # Build task suite
    tasks = {
        "arithmetic": ArithmeticTask(seed=0),
        "copy": CopyTask(seed=0),
        "grammar": GrammarTask(seed=0),
        "knowledge_qa": KnowledgeQATask(seed=0),
        "compositional": CompositionalTask(seed=0),
        "unknown": UnknownTask(seed=0),
    }

    # --- Random Agent ---
    print("\n\nRunning Random Agent...")
    random_agent = RandomAgent()
    random_metrics, random_results = run_eval_suite(random_agent, tasks, n_per_task=N_EVAL)
    print_metrics("Random Agent", random_metrics)

    # --- Human Agent ---
    print("\nRunning Human Agent...")
    human = HumanAgent()
    human_metrics, human_results = run_eval_suite(human, tasks, n_per_task=N_EVAL)
    print_metrics("Human Agent", human_metrics)

    # --- Show sample traces ---
    print("\n" + "=" * 60)
    print("Sample Human Agent Traces")
    print("=" * 60)
    demos = [
        ("Arithmetic", "ADD 12 37 ="),
        ("Grammar", "CHECK: ( [ { } ] )"),
        ("Unknown", "Q: What is the capital of the Moon?"),
    ]
    for label, prompt in demos:
        result = human.run(prompt)
        print(f"\n  [{label}] '{prompt}'")
        print(f"  Answer: {result.answer}  |  Confidence: {result.confidence:.2f}  |  Abstained: {result.abstained}")
        for step in result.trace:
            print(f"    > {step}")

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results(random_results, random_metrics,
                 RESULTS_DIR / "ch00_random.json",
                 agent_name="random", chapter="00_setup")
    save_results(human_results, human_metrics,
                 RESULTS_DIR / "ch00_human.json",
                 agent_name="human_agent", chapter="00_setup")

    # --- Plot ---
    task_names = list(tasks.keys())
    scores = {
        "Random Agent": [random_metrics.per_task[t]["accuracy"] for t in task_names],
        "Human Agent": [human_metrics.per_task[t]["accuracy"] for t in task_names],
    }
    plot_comparison_bar(
        labels=task_names, scores=scores,
        title="Chapter 00: Random vs Human Agent",
        save_path=str(RESULTS_DIR / "ch00_comparison.png"),
    )
    print(f"\nPlot saved to {RESULTS_DIR / 'ch00_comparison.png'}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Random agent accuracy:  {random_metrics.accuracy:.1%}")
    print(f"  Human agent accuracy:   {human_metrics.accuracy:.1%}")
    print(f"  Gap to fill:            {human_metrics.accuracy - random_metrics.accuracy:.1%}")
    print()
    print("Every chapter from here will try to close this gap.")
    print("The question is always: HOW does it close it, and")
    print("does it close it the same way humans do?")


if __name__ == "__main__":
    main()
