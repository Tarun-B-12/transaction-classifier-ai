"""
build_test_sample.py
--------------------
Creates a random 150-transaction sample for manual labeling.

This produces the file you will HAND-LABEL to create ground truth for
evaluation. You label by reading the description only.

The generator's `true_category` is included in a SEPARATE file
(test_sample_answer_key.csv) and is NOT shown in the file you label,
so your labels stay independent.

Usage:
    python src/build_test_sample.py
"""

import pandas as pd

RAW_PATH = "data/raw/transactions.csv"
LABELING_PATH = "eval/test_sample_to_label.csv"
ANSWER_KEY_PATH = "eval/test_sample_answer_key.csv"

SAMPLE_SIZE = 150
RANDOM_SEED = 7   # fixed -> same sample every run, reproducible


def main():
    df = pd.read_csv(RAW_PATH)
    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).copy()

    # the file YOU fill in: description only, blank column for your label
    to_label = sample[["transaction_id", "description"]].copy()
    to_label["my_label"] = ""          # you will fill this in
    to_label.to_csv(LABELING_PATH, index=False)

    # the generator's labels, kept separate for later comparison
    answer_key = sample[["transaction_id", "true_category"]].copy()
    answer_key.to_csv(ANSWER_KEY_PATH, index=False)

    print(f"Wrote {len(to_label)} rows to {LABELING_PATH} (label this file)")
    print(f"Wrote answer key to {ANSWER_KEY_PATH} (do not open while labeling)")
    print("\nValid categories to use:")
    print("  Coffee, Groceries, Gas, Restaurants, Shopping,")
    print("  Transport, Utilities, Entertainment, Health")


if __name__ == "__main__":
    main()