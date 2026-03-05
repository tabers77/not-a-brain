"""Chapter 01: N-gram Language Models

Builds bigram and trigram models on synthetic tasks.
Shows: local pattern completion works, long-range dependencies fail.
Human lens: humans use meaning + goals; n-grams count co-occurrences.

Usage:
    python chapters/01_ngrams/run.py
"""

from pathlib import Path
from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.ngram import BigramModel, TrigramModel, NgramAgent
from not_a_brain.human_agent.agent import HumanAgent
from not_a_brain.evals.harness import run_eval_suite, save_results
from not_a_brain.tasks import (
    ArithmeticTask, CopyTask, GrammarTask,
    KnowledgeQATask, CompositionalTask, UnknownTask,
)
from not_a_brain.utils.visualization import plot_comparison_bar

RESULTS_DIR = Path(__file__).parent / "results"
N_TRAIN = 500
N_EVAL = 50


def build_training_corpus(tasks: dict, n: int) -> list[str]:
    """Build a training corpus from task prompt+answer pairs."""
    corpus = []
    for task in tasks.values():
        pairs = task.training_pairs(n)
        for prompt, answer in pairs:
            # Train on the full sequence: prompt + answer
            corpus.append(prompt + answer)
    return corpus


def main():
    print("=" * 60)
    print("Chapter 01: N-gram Language Models")
    print("=" * 60)

    # 1. Build task suite
    tasks = {
        "arithmetic": ArithmeticTask(seed=1),
        "copy": CopyTask(seed=1),
        "grammar": GrammarTask(seed=1),
        "knowledge_qa": KnowledgeQATask(seed=1),
        "compositional": CompositionalTask(seed=1),
        "unknown": UnknownTask(seed=1),
    }

    # 2. Build training corpus from task data
    print("\nBuilding training corpus...")
    corpus = build_training_corpus(tasks, N_TRAIN)
    print(f"  {len(corpus)} training sequences")

    # 3. Fit tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(corpus)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # 4. Train bigram model
    print("\nTraining bigram model...")
    bigram = BigramModel(tokenizer)
    bigram.train(corpus)

    # 5. Train trigram model
    print("Training trigram model...")
    trigram = TrigramModel(tokenizer)
    trigram.train(corpus)

    # 6. Show some generations
    print("\n--- Sample Generations ---")
    sample_prompts = [
        "ADD 5 3 =",
        "COPY: abc|",
        "CHECK: ( )",
    ]
    for prompt in sample_prompts:
        bi_out = bigram.generate(prompt, max_len=20)
        tri_out = trigram.generate(prompt, max_len=20)
        print(f"  Prompt: '{prompt}'")
        print(f"    Bigram:  '{bi_out}'")
        print(f"    Trigram: '{tri_out}'")

    # 7. Evaluate all agents
    # Re-create tasks with different seed for eval (no data leakage)
    eval_tasks = {
        "arithmetic": ArithmeticTask(seed=99),
        "copy": CopyTask(seed=99),
        "grammar": GrammarTask(seed=99),
        "knowledge_qa": KnowledgeQATask(seed=99),
        "compositional": CompositionalTask(seed=99),
        "unknown": UnknownTask(seed=99),
    }

    bigram_agent = NgramAgent(bigram, "bigram")
    trigram_agent = NgramAgent(trigram, "trigram")
    human = HumanAgent()

    print("\n--- Evaluation ---")
    agents = {
        "bigram": bigram_agent,
        "trigram": trigram_agent,
        "human_agent": human,
    }

    all_metrics = {}
    for agent_name, agent in agents.items():
        metrics, results = run_eval_suite(agent, eval_tasks, n_per_task=N_EVAL)
        all_metrics[agent_name] = metrics
        print(f"\n{agent_name}:")
        print(f"  Accuracy:          {metrics.accuracy:.1%}")
        print(f"  Hallucination rate:{metrics.hallucination_rate:.1%}")
        print(f"  Abstention rate:   {metrics.abstention_rate:.1%}")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_results(results, metrics,
                     RESULTS_DIR / f"ch01_{agent_name}.json",
                     agent_name=agent_name, chapter="01_ngrams")

    # 8. Plot comparison
    task_names = list(eval_tasks.keys())
    scores = {}
    for agent_name, metrics in all_metrics.items():
        scores[agent_name] = [
            metrics.per_task.get(t, {}).get("accuracy", 0.0)
            for t in task_names
        ]

    plot_comparison_bar(
        labels=task_names,
        scores=scores,
        title="Chapter 01: N-grams vs Human Agent",
        save_path=str(RESULTS_DIR / "ch01_comparison.png"),
    )
    print(f"\nPlot saved to {RESULTS_DIR / 'ch01_comparison.png'}")

    # 9. Key takeaway
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("N-grams can do local pattern completion but fail at:")
    print("  - Long-range dependencies (arithmetic, grammar)")
    print("  - Compositional tasks (chained operations)")
    print("  - Abstention (they always generate something)")
    print()
    print("Human lens: Humans don't count co-occurrences.")
    print("They understand meaning, apply algorithms, and know")
    print("when they don't have enough information.")


if __name__ == "__main__":
    main()
