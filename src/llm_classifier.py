"""
llm_classifier.py
-----------------
The AI classifier: categorizes transaction descriptions using the Claude API.

Key engineering choices:
  - Batching: sends BATCH_SIZE transactions per API call (not one each).
    Fewer calls -> faster and cheaper.
  - Structured output: asks the model to return JSON so results parse cleanly.
  - Runs on the SAME 150 test-set transactions the rule-based classifier
    was measured on, so Day 5 is a fair comparison.

Usage:
    python src/llm_classifier.py
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"   # small, fast, cheap -- right for this
BATCH_SIZE = 20
TEST_SET_PATH = "eval/test_set_final.csv"
RAW_PATH = "data/raw/transactions.csv"
OUT_PATH = "data/processed/llm_predictions.csv"

CATEGORIES = ["Coffee", "Groceries", "Gas", "Restaurants", "Shopping",
              "Transport", "Utilities", "Entertainment", "Health"]


def build_prompt(descriptions):
    """Build one prompt that asks the model to classify a batch."""
    numbered = "\n".join(f"{i}. {d}" for i, d in enumerate(descriptions))
    category_list = ", ".join(CATEGORIES)
    return f"""You are classifying bank transaction descriptions into spend categories.

Valid categories (use ONLY these, exact spelling):
{category_list}

Classify each numbered transaction below. Return ONLY a JSON object that maps
each number (as a string) to one category. No explanation, no extra text.

Example format:
{{"0": "Coffee", "1": "Gas"}}

Transactions:
{numbered}"""


def classify_batch(client, descriptions):
    """Send one batch to the API and return a list of categories."""
    prompt = build_prompt(descriptions)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()

    # the model sometimes wraps JSON in ```json fences -- strip them
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    result = json.loads(text)
    return [result[str(i)] for i in range(len(descriptions))]


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found. Check your .env file.")
        raise SystemExit(1)
    client = Anthropic(api_key=api_key)

    # the test set has transaction_id + description; get amount from raw
    test = pd.read_csv(TEST_SET_PATH)
    raw = pd.read_csv(RAW_PATH)[["transaction_id", "amount"]]
    df = test.merge(raw, on="transaction_id")

    descriptions = df["description"].tolist()
    print(f"Classifying {len(descriptions)} transactions "
          f"in batches of {BATCH_SIZE}...")

    predictions = []
    for start in range(0, len(descriptions), BATCH_SIZE):
        batch = descriptions[start:start + BATCH_SIZE]
        try:
            preds = classify_batch(client, batch)
        except Exception as e:
            print(f"  Batch starting at {start} failed: {e}")
            preds = ["ERROR"] * len(batch)
        predictions.extend(preds)
        print(f"  Done {min(start + BATCH_SIZE, len(descriptions))}"
              f"/{len(descriptions)}")
        time.sleep(0.5)   # gentle pacing between calls

    df["llm_category"] = predictions
    out = df[["transaction_id", "description", "amount",
              "final_category", "llm_category"]]
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved LLM predictions to {OUT_PATH}")

    errors = (out["llm_category"] == "ERROR").sum()
    if errors:
        print(f"WARNING: {errors} transactions failed -- check API output.")
    print("\nLLM predicted category counts:")
    print(out["llm_category"].value_counts())


if __name__ == "__main__":
    main()
    