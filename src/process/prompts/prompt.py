"""
Underwriting prompt builder.

Converts a User dataclass into a compact (system_prompt, user_message) pair
ready for any zero-shot LLM underwriting call.  The financial summary is
derived entirely from Transaction data already on the User object; run
TransactionCleaner and TransactionCategorizer before calling build_prompt().

Token budget philosophy
-----------------------
- System prompt: ~400 tokens — role, output schema, scoring bands, product list.
- User message:  ~300 tokens — JSON financial summary, no raw transaction text.
"""

import json
import os
import sys
from collections import defaultdict
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from custom_dataclasses.user_data import User
    from process.products import Product, PRODUCTS
except ImportError:
    from ...custom_dataclasses.user_data import User
    from ..products import Product, PRODUCTS


# ---------------------------------------------------------------------------
# Financial summary builder
# ---------------------------------------------------------------------------

_INCOME_CATEGORIES = {"Income"}
_OBLIGATION_CATEGORIES = {"Financial Obligations", "Housing"}
_RISK_SUBCATS = {"nsf", "overdraft", "payday"}


def _compute_summary(user: User) -> dict:
    """
    Derive a compact financial summary dict from the User's transaction history.
    All monetary values are rounded to two decimal places and expressed in CAD.
    """
    all_txns = [
        txn
        for account in (user.accounts or [])
        for txn in account.transactions
    ]

    if not all_txns:
        return {"error": "No transaction data available."}

    # --- Date range ---
    dates = [txn.date for txn in all_txns]
    min_date = min(dates)
    max_date = max(dates)
    months = max(
        1,
        (max_date.year - min_date.year) * 12
        + (max_date.month - min_date.month)
        + 1,
    )

    # --- Category aggregation ---
    income_total       = Decimal("0")
    expense_by_cat: Dict[str, Decimal] = defaultdict(Decimal)
    nsf_count          = 0
    overdraft_count    = 0
    payday_loan_count  = 0

    for txn in all_txns:
        cat    = txn.category[0] if txn.category else "Other"
        subcat = (txn.subcategory or "").lower()

        if cat in _INCOME_CATEGORIES and txn.amount > 0:
            income_total += txn.amount
        elif txn.amount < 0:
            expense_by_cat[cat] += abs(txn.amount)

        if "nsf" in subcat:
            nsf_count += 1
        if "overdraft" in subcat:
            overdraft_count += 1
        if "payday" in subcat:
            payday_loan_count += 1

    monthly_income   = float(income_total / months)
    total_expenses   = sum(expense_by_cat.values(), Decimal("0"))
    monthly_expenses = float(total_expenses / months)

    # Top expense categories (exclude tiny noise)
    top_cats = {
        cat: round(float(amt / months), 2)
        for cat, amt in sorted(expense_by_cat.items(), key=lambda x: x[1], reverse=True)
        if float(amt / months) >= 10
    }

    # --- Account summaries ---
    accounts = []
    for acc in (user.accounts or []):
        entry: dict = {
            "type":    acc.type,
            "subtype": acc.subtype,
            "balance": float(round(acc.current_balance, 2)),
        }
        if acc.credit_limit and acc.credit_limit > 0:
            used = acc.credit_limit - (acc.available_balance or acc.current_balance)
            entry["credit_limit"]      = float(acc.credit_limit)
            entry["utilization_pct"]   = round(float(used / acc.credit_limit * 100), 1)
        accounts.append(entry)

    return {
        "period":           f"{min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}",
        "months_analyzed":  months,
        "accounts":         accounts,
        "monthly": {
            "income_avg":   round(monthly_income, 2),
            "expense_avg":  round(monthly_expenses, 2),
            "net_avg":      round(monthly_income - monthly_expenses, 2),
            "by_category":  top_cats,
        },
        "annual_income_est": round(monthly_income * 12, 2),
        "risk": {
            "nsf_count":         nsf_count,
            "overdraft_count":   overdraft_count,
            "payday_loan_count": payday_loan_count,
        },
    }


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a financial underwriter for Wealthsimple. Analyze the client financial summary and return ONLY a valid JSON object — no markdown, no explanation.

Output schema:
{{
  "score": <integer 300-900, creditworthiness score>,
  "decision": <"approved" | "conditional" | "rejected">,
  "summary": <2-3 sentence plain-English explanation of the decision>,
  "rejection_reason": <string if rejected, otherwise null>,
  "recommended_products": [<product names from the list below; empty array if rejected>]
}}

Scoring bands: 750-900 excellent, 650-749 good (approved), 550-649 fair (conditional), 300-549 poor (rejected).
Signals that lower the score: NSF events, overdraft fees, payday loans, negative monthly net cash flow, high credit utilization (>70 %), irregular income.
Signals that raise the score: consistent income, positive monthly net, low obligations relative to income, growing balances.

Product eligibility rules:
- Wealthsimple Credit Card: only recommend when estimated annual income ≥ CAD $60,000.
- Crypto: only recommend when monthly net cash flow is positive AND nsf_count = 0 AND overdraft_count = 0.
- If decision is "rejected", recommended_products must be an empty array.
- Use each product's Guidance note to judge fit; do not recommend products that are clearly unsuitable.

Available products:
{products}"""


def build_prompt(
    user: User,
    products: Optional[List[Product]] = None,
) -> Tuple[str, str]:
    """
    Build the (system_prompt, user_message) pair for an LLM underwriting call.

    Parameters
    ----------
    user : User
        A fully populated User object (transactions cleaned and categorized).
    products : list of Product, optional
        Override the default PRODUCTS catalogue (useful for testing).

    Returns
    -------
    system_prompt : str
    user_message  : str   (compact JSON financial summary)
    """
    product_block = PRODUCTS.products_for_prompt(products)
    system_prompt = _SYSTEM_TEMPLATE.format(products=product_block)

    summary = _compute_summary(user)
    user_message = (
        f"Client ID: {user.user_id}\n"
        f"Financial summary:\n{json.dumps(summary, indent=2)}"
    )

    return system_prompt, user_message
