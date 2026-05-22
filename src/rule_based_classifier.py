"""
rule_based_classifier.py
------------------------
The TRADITIONAL baseline: classifies transaction descriptions into spend
categories using simple keyword matching.

This is deliberately built BEFORE the LLM classifier. It is the benchmark
the AI approach must beat. Day 5 evaluation compares the two.

Steps:
  1. Load the raw dataset
  2. Normalize descriptions (lowercase, collapse extra spaces)
  3. Classify each transaction by matching keywords
  4. Save predictions to data/processed/

Usage:
    python src/rule_based_classifier.py
"""

import re
import pandas as pd

RAW_PATH = "data/raw/transactions.csv"
OUT_PATH = "data/processed/rule_based_predictions.csv"

# ---------------------------------------------------------------------------
# Keyword rules: realistic, NOT exhaustive.
# These are the keywords a person would plausibly write without having seen
# every merchant in the data. The classifier is EXPECTED to miss some
# transactions -- those misses are what the LLM gets a chance to beat.
# ---------------------------------------------------------------------------
KEYWORD_RULES = {
    "Coffee": ["starbucks", "dunkin", "coffee"],
    "Groceries": ["whole foods", "wholefds", "trader joe", "safeway",
                  "kroger"],
    "Gas": ["shell", "chevron", "exxon", "gas"],
    "Restaurants": ["chipotle", "doordash", "grubhub", "restaurant",
                    "diner"],
    "Shopping": ["amazon", "amzn", "target", "walmart"],
    "Transport": ["uber", "lyft", "transit", "parking"],
    "Utilities": ["electric", "comcast", "verizon", "energy", "water"],
    "Entertainment": ["netflix", "spotify", "cinema", "amc"],
    "Health": ["cvs", "walgreens", "pharmacy", "fitness"],
}

# the label used when no keyword matches at all
UNMATCHED = "Uncategorized"


def normalize(description):
    """Lowercase and collapse repeated/trailing whitespace."""
    text = str(description).lower()
    text = re.sub(r"\s+", " ", text)   # any run of spaces -> single space
    return text.strip()


def classify(description):
    """Return the first category whose keyword appears, else Uncategorized."""
    text = normalize(description)
    for category, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text:
                return category
    return UNMATCHED


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} transactions from {RAW_PATH}")

    df["clean_description"] = df["description"].apply(normalize)
    df["rule_based_category"] = df["description"].apply(classify)

    # keep only the columns we need going forward
    out = df[["transaction_id", "description", "clean_description",
              "amount", "true_category", "rule_based_category"]]
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved predictions to {OUT_PATH}")

    # quick summary so we can eyeball how the baseline did
    n_unmatched = (out["rule_based_category"] == UNMATCHED).sum()
    print(f"\nUncategorized (no keyword matched): {n_unmatched} "
          f"({n_unmatched / len(out) * 100:.1f}%)")
    print("\nPredicted category counts:")
    print(out["rule_based_category"].value_counts())


if __name__ == "__main__":
    main()