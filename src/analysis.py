"""
analysis.py
-----------
Loads the evaluation results into DuckDB and runs SQL analysis queries.

Shows how to use SQL on the classifier results: accuracy comparison,
per-category breakdown, and a look at where the baseline failed.

Usage:
    python src/analysis.py
"""

import duckdb
import pandas as pd

TEST_SET_PATH = "eval/test_set_final.csv"
RULE_PATH = "data/processed/rule_based_predictions.csv"
LLM_PATH = "data/processed/llm_predictions.csv"

# build one combined table in DuckDB
truth = pd.read_csv(TEST_SET_PATH)
rule = pd.read_csv(RULE_PATH)[["transaction_id", "rule_based_category"]]
llm = pd.read_csv(LLM_PATH)[["transaction_id", "amount", "llm_category"]]
df = truth.merge(rule, on="transaction_id").merge(llm, on="transaction_id")

con = duckdb.connect()
con.register("results", df)

print("=" * 55)
print("QUERY 1: Overall accuracy, both methods")
print("=" * 55)
print(con.sql("""
    SELECT
        ROUND(AVG(CASE WHEN rule_based_category = final_category
                  THEN 1 ELSE 0 END) * 100, 1) AS rule_accuracy_pct,
        ROUND(AVG(CASE WHEN llm_category = final_category
                  THEN 1 ELSE 0 END) * 100, 1) AS llm_accuracy_pct,
        COUNT(*) AS transactions
    FROM results
""").df().to_string(index=False))

print()
print("=" * 55)
print("QUERY 2: Per-category accuracy")
print("=" * 55)
print(con.sql("""
    SELECT
        final_category AS category,
        COUNT(*) AS n,
        ROUND(AVG(CASE WHEN rule_based_category = final_category
                  THEN 1 ELSE 0 END) * 100, 0) AS rule_pct,
        ROUND(AVG(CASE WHEN llm_category = final_category
                  THEN 1 ELSE 0 END) * 100, 0) AS llm_pct
    FROM results
    GROUP BY final_category
    ORDER BY llm_pct - rule_pct DESC
""").df().to_string(index=False))

print()
print("=" * 55)
print("QUERY 3: Where the baseline failed (Uncategorized rows)")
print("=" * 55)
print(con.sql("""
    SELECT
        final_category AS true_category,
        COUNT(*) AS missed_by_baseline,
        ROUND(AVG(CASE WHEN llm_category = final_category
                  THEN 1 ELSE 0 END) * 100, 0) AS llm_recovered_pct
    FROM results
    WHERE rule_based_category = 'Uncategorized'
    GROUP BY final_category
    ORDER BY missed_by_baseline DESC
""").df().to_string(index=False))

print()
print("=" * 55)
print("QUERY 4: Spend by predicted category (LLM)")
print("=" * 55)
print(con.sql("""
    SELECT
        llm_category AS category,
        COUNT(*) AS transactions,
        ROUND(SUM(amount), 2) AS total_spend
    FROM results
    GROUP BY llm_category
    ORDER BY total_spend DESC
""").df().to_string(index=False))

con.close()
