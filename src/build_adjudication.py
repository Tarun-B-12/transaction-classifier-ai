"""
build_adjudication.py
---------------------
Pulls every row where the hand label disagrees with the generator label
into one file for a final review pass (label adjudication).

You fill in `final_label` for each row -- that becomes the locked ground truth.

Usage:
    python src/build_adjudication.py
"""

import pandas as pd

labels = pd.read_csv("eval/test_sample_to_label.csv")
key = pd.read_csv("eval/test_sample_answer_key.csv")

labels["my_label"] = labels["my_label"].astype(str).str.strip()
m = labels.merge(key, on="transaction_id")

disagree = m[m["my_label"] != m["true_category"]].copy()
disagree = disagree[["transaction_id", "description",
                     "my_label", "true_category"]]
disagree = disagree.rename(columns={"true_category": "generator_label"})
disagree["final_label"] = ""   # you fill this in

disagree.to_csv("eval/disagreements_to_review.csv", index=False)
print(f"Wrote {len(disagree)} rows to eval/disagreements_to_review.csv")
print("Fill in the final_label column for each row.")
