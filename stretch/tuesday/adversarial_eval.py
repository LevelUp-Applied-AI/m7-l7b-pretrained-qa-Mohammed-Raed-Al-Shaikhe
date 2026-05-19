"""
Module 7 Week B — Tuesday Stretch (Honors): Adversarial QA Probe.

Reuses the QA pipeline + EM/F1 functions from `lab.py`. Implement the TODO
functions below; see the stretch page for full task description.
"""

import json
import os
import sys

import pandas as pd

# Import the lab's existing functions (we reuse build_qa_pipeline, predict_one,
# evaluate_qa, normalize_answer, exact_match, token_f1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import lab  # noqa: E402


def load_adversarial_set(path: str = "stretch/tuesday/adversarial_set.csv") -> pd.DataFrame:
    """
    Load the adversarial test set CSV.

    Verifies columns: qid, question, context, gold_answer, pattern_tag.
    """

    df = pd.read_csv(path)

    return df


def evaluate_adversarial(qa, df: pd.DataFrame) -> dict:
    """
    Run the QA pipeline on the adversarial set; compute aggregate + per-pattern metrics.

    Returns:
        {
            "em": float, "f1": float, "n": int,
            "per_pattern": { tag: {"em": float, "f1": float, "n": int}, ... },
            "predictions": [ ... lab.evaluate_qa-shaped entries plus pattern_tag ... ],
        }
    """
    
    predictions = []

    total_em = 0
    total_f1 = 0

    per_pattern = {}

    for _, row in df.iterrows():

        pred = lab.predict_one(qa, row["question"], row["context"])

        em = lab.exact_match(pred, row["gold_answer"])
        f1 = lab.token_f1(pred, row["gold_answer"])

        total_em += em
        total_f1 += f1

        tag = row["pattern_tag"]

        if tag not in per_pattern:
            per_pattern[tag] = {"em": 0, "f1": 0, "n": 0}

        per_pattern[tag]["em"] += em
        per_pattern[tag]["f1"] += f1
        per_pattern[tag]["n"] += 1

        predictions.append({
            "qid": row["qid"],
            "question": row["question"],
            "context_excerpt": row["context"][:80],
            "gold_answer": row["gold_answer"],
            "predicted_answer": pred,
            "em": em,
            "f1": f1,
            "pattern_tag": tag
        })

    # finalize averages
    for tag in per_pattern:
        n = per_pattern[tag]["n"]
        per_pattern[tag]["em"] /= n
        per_pattern[tag]["f1"] /= n

    n = len(df)

    return {
        "em": total_em / n,
        "f1": total_f1 / n,
        "n": n,
        "per_pattern": per_pattern,
        "predictions": predictions
    }


def main() -> None:
    """Load adversarial set, run evaluation, write predictions + metrics."""
    df = load_adversarial_set()
    qa = lab.build_qa_pipeline(lab.get_qa_model_name())
    result = evaluate_adversarial(qa, df)

    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv("stretch/tuesday/adversarial_predictions.csv", index=False)

    metrics = {
        "em": result["em"],
        "f1": result["f1"],
        "n": result["n"],
        "per_pattern": result["per_pattern"],
        "model": lab.get_qa_model_name(),
    }
    with open("stretch/tuesday/adversarial_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Aggregate EM = {result['em']:.4f}")
    print(f"Aggregate F1 = {result['f1']:.4f}")
    print(f"n = {result['n']}")
    print(f"Per-pattern: {list(result['per_pattern'].keys())}")


if __name__ == "__main__":
    main()
