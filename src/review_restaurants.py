"""
review_restaurants.py
---------------------
Shows every test-set row currently labeled 'Restaurants' so you can spot
any that are actually coffee shops (or otherwise mislabeled).

Read-only -- it changes nothing. You make edits in
eval/disagreements_to_review.csv or eval/test_sample_to_label.csv,
then re-run finalize_test_set.py.

Usage:
    python src/review_restaurants.py
"""

import pandas as pd

final = pd.read_csv("eval/test_set_final.csv")

restaurants = final[final["final_category"] == "Restaurants"]
print(f"{len(restaurants)} rows currently labeled Restaurants:\n")
for _, r in restaurants.iterrows():
    print(f"  {r['transaction_id']}   {r['description']!r}")

print()
print("Coffee-chain names to watch for: blue bottle, philz, peets,")
print("dunkin, starbucks, 'grounds', 'roasters', 'cafe', 'coffee'.")
print("If any row above is clearly a coffee merchant, note its")
print("transaction_id -- you'll fix those next.")