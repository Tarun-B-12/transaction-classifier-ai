"""
generate_transactions.py
------------------------
Generates a synthetic bank-transaction dataset for the AI Transaction
Classifier project.

The point of this dataset is the `description` column: realistically MESSY
merchant strings (abbreviations, store numbers, location codes, payment-
processor prefixes) that a classifier has to interpret. A `true_category`
column is included as ground truth -- used later to build the evaluation set.

This is SYNTHETIC data. It is designed to simulate the kind of messiness
real bank-transaction descriptions have. It is not real customer data.

Usage:
    python generate_transactions.py
Produces:
    transactions.csv
"""

import csv
import random
from datetime import date, timedelta

random.seed(42)  # reproducible -- same file every run

# ---------------------------------------------------------------------------
# Each category maps to a list of "merchant templates".
# {n}   -> a random store number
# {loc} -> a random location code (city abbrev + state)
# Prefixes like "SQ *", "TST*", "POS DEBIT" mimic payment-processor noise.
# ---------------------------------------------------------------------------

LOCATIONS = ["OAKLAND CA", "AUSTIN TX", "DALLAS TX", "NY NY", "CHICAGO IL",
             "SEATTLE WA", "MIAMI FL", "DENVER CO", "BOSTON MA", "PHX AZ"]

CATEGORY_TEMPLATES = {
    "Coffee": [
        "SQ *BLUE BOTTLE {n} {loc}",
        "STARBUCKS STORE {n}",
        "TST* PHILZ COFFEE {loc}",
        "PEETS COFFEE #{n} {loc}",
        "DUNKIN #{n} {loc}",
        "SQ *LOCAL GROUNDS {loc}",
    ],
    "Groceries": [
        "WHOLEFDS MKT {n} {loc}",
        "TRADER JOE'S #{n}",
        "SAFEWAY #{n} {loc}",
        "KROGER {n} {loc}",
        "ALDI {n} {loc}",
        "POS DEBIT H-E-B #{n}",
    ],
    "Gas": [
        "SHELL OIL {n} {loc}",
        "CHEVRON {n}",
        "EXXONMOBIL {n} {loc}",
        "POS DEBIT BP#{n}",
        "76 - {n} {loc}",
        "CIRCLE K # {n}",
    ],
    "Restaurants": [
        "TST* OLIVE & VINE {loc}",
        "SQ *TAQUERIA EL SOL {loc}",
        "CHIPOTLE {n} {loc}",
        "DOORDASH*MCDONALDS",
        "GRUBHUB*THAI BASIL {loc}",
        "TST* THE CORNER DINER",
    ],
    "Shopping": [
        "AMZN MKTP US*{n} AMZN.COM/BILL WA",
        "TARGET {n} {loc}",
        "WALMART SUPERCENTER #{n}",
        "BEST BUY #{n} {loc}",
        "SQ *THE BOOK NOOK {loc}",
        "ETSY.COM - SELLER{n}",
    ],
    "Transport": [
        "UBER *TRIP {loc}",
        "LYFT *RIDE THU 4PM",
        "MTA*NYCT PAYGO {loc}",
        "DART RAIL {loc}",
        "PARKING METER #{n} {loc}",
        "UBER* EATS HELP.UBER.COM",  # deliberately ambiguous-ish
    ],
    "Utilities": [
        "ONCOR ELECTRIC DELIVERY",
        "ATT*BILL PAYMENT {n}",
        "CITY OF DALLAS WATER UT",
        "COMCAST CABLE COMM {n}",
        "TXU ENERGY PAYMENT",
        "VERIZON WIRELESS {n}",
    ],
    "Entertainment": [
        "NETFLIX.COM {n}",
        "SPOTIFY USA {n}",
        "AMC ONLINE #{n}",
        "STEAMGAMES.COM {n} WA",
        "SQ *THE COMEDY CLUB {loc}",
        "REGAL CINEMAS {n} {loc}",
    ],
    "Health": [
        "CVS/PHARMACY #{n} {loc}",
        "WALGREENS #{n}",
        "POS DEBIT QUEST DIAGNOSTIC",
        "LA FITNESS {n} {loc}",
        "SQ *CITY DENTAL {loc}",
        "RITE AID #{n} {loc}",
    ],
}

# rough monetary range per category (min, max) in dollars
CATEGORY_AMOUNTS = {
    "Coffee":        (3, 12),
    "Groceries":     (15, 220),
    "Gas":           (25, 95),
    "Restaurants":   (12, 140),
    "Shopping":      (8, 480),
    "Transport":     (2, 65),
    "Utilities":     (35, 310),
    "Entertainment": (6, 90),
    "Health":        (10, 260),
}


def random_store_number():
    """Store numbers come in varied widths -- part of the messiness."""
    length = random.choice([2, 3, 4, 5])
    return str(random.randint(0, 10 ** length - 1)).zfill(random.choice([0, length]))


def make_description(template):
    desc = template.replace("{n}", random_store_number())
    desc = desc.replace("{loc}", random.choice(LOCATIONS))
    # occasional extra messiness: trailing spaces / doubled spaces / lowercase
    r = random.random()
    if r < 0.08:
        desc = desc + "  "
    elif r < 0.14:
        desc = desc.replace(" ", "  ", 1)
    elif r < 0.18:
        desc = desc.lower()
    return desc


def generate(n_rows=2000, out_path="transactions.csv"):
    start = date(2023, 1, 1)
    end = date(2023, 12, 31)
    span_days = (end - start).days

    categories = list(CATEGORY_TEMPLATES.keys())
    rows = []

    for i in range(1, n_rows + 1):
        category = random.choice(categories)
        template = random.choice(CATEGORY_TEMPLATES[category])
        description = make_description(template)

        lo, hi = CATEGORY_AMOUNTS[category]
        amount = round(random.uniform(lo, hi), 2)

        txn_date = start + timedelta(days=random.randint(0, span_days))

        rows.append({
            "transaction_id": f"T{i:05d}",
            "date": txn_date.isoformat(),
            "description": description,
            "amount": amount,
            "true_category": category,
        })

    # shuffle so categories are not grouped
    random.shuffle(rows)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["transaction_id", "date", "description",
                        "amount", "true_category"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Categories: {', '.join(categories)}")


if __name__ == "__main__":
    generate()
