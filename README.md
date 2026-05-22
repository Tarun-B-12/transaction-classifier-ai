# AI-Powered Transaction Classifier

Classifies messy bank transaction descriptions into spend categories using a Large Language Model, then measures whether the AI approach actually beats a traditional rule-based baseline.

## Business Problem

Banks, budgeting apps, and finance teams need to turn raw transaction descriptions (for example, `SQ *BLUE BOTTLE 9722 DALLAS TX`) into clean spend categories. Rule-based keyword systems handle the obvious cases but break on messy or unusual descriptions. Clean categorization is the foundation for budgeting tools, spend analytics, and fraud detection.

## Target Stakeholder

Product or data teams at a fintech or budgeting app, or a finance operations team categorizing card spend.

## Approach

This project deliberately builds and compares two methods.

1. Rule-based classifier: keyword matching, which serves as the traditional baseline.
2. LLM classifier: the Claude API, processing transactions in batches.
3. Evaluation framework: a hand-labeled test set that measures the accuracy of both methods, including a per-category breakdown and estimated API cost.

The goal is not just to use AI. The goal is to measure whether it is actually the better choice.

## Tools Used

- Python (pandas) for data handling and the pipeline
- Anthropic Claude API for LLM classification
- DuckDB for local SQL analysis of results
- matplotlib for charts
- Jupyter Notebook for exploration
- GitHub for version control

## Dataset

Synthetic dataset of 2,000 transactions, generated to simulate realistic merchant description messiness. This includes payment processor prefixes (TST*, SQ*, POS DEBIT), inconsistent casing, store numbers, and formatting noise. This is not real customer data. It is generated via `src/generate_transactions.py` and is reproducible because it uses a fixed random seed.

Columns: `transaction_id`, `date`, `description`, `amount`, `true_category`.

## Project Status

- [x] Day 1: Setup, dataset, data exploration
- [ ] Day 2: Data cleaning and rule-based baseline classifier
- [ ] Day 3: Hand-labeled test set
- [ ] Day 4: LLM classifier
- [ ] Day 5: Evaluation framework
- [ ] Day 6: SQL analysis and charts
- [ ] Day 7: Documentation and launch

## Repository Structure

```text
transaction-classifier-ai/
  data/
    raw/          original dataset
    processed/    cleaned outputs
  notebooks/      exploration and analysis
  sql/            DuckDB analysis queries
  src/            pipeline scripts
  eval/           test set and evaluation results
  charts/         output charts
  docs/           data dictionary, assumptions
  README.md
```

## Notes

All metrics in this project are project-level and dataset-level, based on a synthetic dataset. They do not represent real-world company performance.