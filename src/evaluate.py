"""
evaluate.py
-----------
The evaluation framework: measures both classifiers against the locked
hand-labeled test set.

Produces:
  - Overall accuracy for the rule-based baseline and the LLM
  - The accuracy lift (LLM minus rule-based)
  - Per-category accuracy for both
  - A saved results file for the README

This is the core of the project: it answers "did the AI actually help?"

Usage:
    python src/evaluate.py
"""

import pandas as pd

TEST_SET_PATH = "eval/test_set_final.csv"
RULE_PATH = "data/processed/rule_based_predictions.csv"
LLM_PATH = "data/processed/llm_predictions.csv"
RESULTS_PATH = "eval/evaluation_results.csv"


def main():
    truth = pd.read_csv(TEST_SET_PATH)          # transaction_id, description, final_category
    rule = pd.read_csv(RULE_PATH)               # has rule_based_category
    llm = pd.read_csv(LLM_PATH)                 # has llm_category

    # keep only what we need, then join everything on transaction_id
    df = truth.merge(
        rule[["transaction_id", "rule_based_category"]], on="transaction_id"
    ).merge(
        llm[["transaction_id", "llm_category"]], on="transaction_id"
    )
    print(f"Evaluating on {len(df)} test transactions\n")

    # --- overall accuracy ---
    df["rule_correct"] = df["rule_based_category"] == df["final_category"]
    df["llm_correct"] = df["llm_category"] == df["final_category"]

    rule_acc = df["rule_correct"].mean() * 100
    llm_acc = df["llm_correct"].mean() * 100
    lift = llm_acc - rule_acc

    print("=" * 48)
    print("OVERALL ACCURACY")
    print("=" * 48)
    print(f"  Rule-based baseline : {rule_acc:5.1f}%")
    print(f"  LLM classifier      : {llm_acc:5.1f}%")
    print(f"  Lift (LLM - rule)   : {lift:+5.1f} percentage points")
    print()

    # --- per-category accuracy ---
    print("=" * 48)
    print("PER-CATEGORY ACCURACY")
    print("=" * 48)
    print(f"  {'Category':<14}{'N':>4}{'Rule':>9}{'LLM':>9}")
    rows = []
    for cat in sorted(df["final_category"].unique()):
        sub = df[df["final_category"] == cat]
        n = len(sub)
        r = sub["rule_correct"].mean() * 100
        l = sub["llm_correct"].mean() * 100
        print(f"  {cat:<14}{n:>4}{r:>8.0f}%{l:>8.0f}%")
        rows.append({"category": cat, "n": n,
                     "rule_accuracy": round(r, 1),
                     "llm_accuracy": round(l, 1)})

    # --- where they disagree with the truth ---
    print()
    print("=" * 48)
    print("SAMPLE: transactions the LLM got wrong")
    print("=" * 48)
    llm_wrong = df[~df["llm_correct"]]
    print(f"  {len(llm_wrong)} total. Showing up to 12:\n")
    for _, r in llm_wrong.head(12).iterrows():
        print(f"  {r['description']!r}")
        print(f"     truth: {r['final_category']}  |  "
              f"LLM: {r['llm_category']}  |  "
              f"rule: {r['rule_based_category']}")

    # --- save results for the README ---
    results = pd.DataFrame(rows)
    summary = pd.DataFrame([{"category": "OVERALL", "n": len(df),
                             "rule_accuracy": round(rule_acc, 1),
                             "llm_accuracy": round(llm_acc, 1)}])
    pd.concat([summary, results], ignore_index=True).to_csv(
        RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()