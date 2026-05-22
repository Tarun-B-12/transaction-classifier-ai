"""Validate the hand-labeled test set and compare it to the generator labels."""
import pandas as pd

VALID = {"Coffee", "Groceries", "Gas", "Restaurants", "Shopping",
         "Transport", "Utilities", "Entertainment", "Health"}

labels = pd.read_csv("eval/test_sample_to_label.csv")
labels["my_label"] = labels["my_label"].astype(str).str.strip()

# --- quality check ---
blanks = labels["my_label"].isin(["", "nan"]).sum()
bad = labels[~labels["my_label"].isin(VALID) & ~labels["my_label"].isin(["", "nan"])]

print("Total rows:", len(labels))
print("Blank labels:", blanks)
print("Labels not matching the 9 valid categories:", len(bad))
if len(bad):
    print("Invalid label values found:")
    print(bad["my_label"].value_counts())
print()
print("Your label distribution:")
print(labels["my_label"].value_counts())

# --- compare to generator ---
key = pd.read_csv("eval/test_sample_answer_key.csv")
m = labels.merge(key, on="transaction_id")
agree = (m["my_label"] == m["true_category"]).sum()
print()
print(f"Your labels agree with the generator on {agree}/{len(m)} "
      f"({agree / len(m) * 100:.1f}%)")

disagree = m[m["my_label"] != m["true_category"]]
if len(disagree):
    print()
    print("Disagreements (usually the genuinely ambiguous ones):")
    for _, r in disagree.iterrows():
        print(f"  {r['description']!r}  ->  you: {r['my_label']}  "
              f"|  generator: {r['true_category']}")