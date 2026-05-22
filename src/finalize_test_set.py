"""
finalize_test_set.py
--------------------
Builds the LOCKED ground-truth test set.

Logic:
  - Start from the hand-labeled sample.
  - For the rows that went through adjudication, replace the label with
    the adjudicated final_label.
  - Validate everything, then write eval/test_set_final.csv.

This locked file is the ground truth for Day 4 and Day 5. Nothing grades
against anything else.

Usage:
    python src/finalize_test_set.py
"""

import pandas as pd

VALID = {"Coffee", "Groceries", "Gas", "Restaurants", "Shopping",
         "Transport", "Utilities", "Entertainment", "Health"}

labels = pd.read_csv("eval/test_sample_to_label.csv")
adjud = pd.read_csv("eval/disagreements_to_review.csv")

labels["my_label"] = labels["my_label"].astype(str).str.strip()
adjud["final_label"] = adjud["final_label"].astype(str).str.strip()

# --- validate the adjudication file first ---
adj_blanks = adjud["final_label"].isin(["", "nan"]).sum()
adj_bad = adjud[~adjud["final_label"].isin(VALID)
                & ~adjud["final_label"].isin(["", "nan"])]
if adj_blanks or len(adj_bad):
    print("PROBLEM in disagreements_to_review.csv -- fix before continuing:")
    print("  blank final_label rows:", adj_blanks)
    if len(adj_bad):
        print("  invalid final_label values:")
        print(adj_bad[["transaction_id", "description", "final_label"]])
    raise SystemExit(1)

# --- build the final label: adjudicated value wins where it exists ---
final_map = dict(zip(adjud["transaction_id"], adjud["final_label"]))

def resolve(row):
    tid = row["transaction_id"]
    return final_map.get(tid, row["my_label"])

labels["final_category"] = labels.apply(resolve, axis=1)

out = labels[["transaction_id", "description", "final_category"]]

# --- final validation ---
blanks = out["final_category"].isin(["", "nan"]).sum()
bad = out[~out["final_category"].isin(VALID)]
print("Rows:", len(out))
print("Blank final labels:", blanks)
print("Invalid final labels:", len(bad))
print()
print("Final ground-truth distribution:")
print(out["final_category"].value_counts())

if blanks or len(bad):
    print("\nNOT saved -- fix the issues above and re-run.")
    raise SystemExit(1)

out.to_csv("eval/test_set_final.csv", index=False)
print("\nLocked ground truth saved to eval/test_set_final.csv")
print(f"Adjudicated rows applied: {len(final_map)}")