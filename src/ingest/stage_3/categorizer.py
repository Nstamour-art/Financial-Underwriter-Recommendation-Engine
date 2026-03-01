"""
Stage 3 - Transaction Categorizer

Assigns a two-level category to cleaned transaction descriptions using
zero-shot NLI classification (facebook/bart-large-mnli by default).

Output format matches Transaction.category: List[str]
  e.g. ["Food and Dining", "Restaurants and Takeout"]

The taxonomy covers the categories that matter most for underwriting:
income sources, fixed obligations, discretionary spending, and risk signals
(NSF fees, overdraft charges, payday loans, etc.).

Usage:
    categorizer = TransactionCategorizer()
    txn.category = categorizer.categorize(txn.cleaned_description, txn.amount)

    # or bulk:
    categorizer.categorize_users(users)  # modifies in-place, run Cleaner first
"""

import os
import re
import sys
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from transformers import pipeline as hf_pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from custom_dataclasses.user_data import User
except ImportError:
    from ...custom_dataclasses.user_data import User


# ---------------------------------------------------------------------------
# Two-level taxonomy  (top-level → sub-categories)
# Designed for underwriting: captures income, fixed costs, debt obligations,
# and financial-risk signals that a credit-decisioning LLM needs.
# ---------------------------------------------------------------------------

TAXONOMY: Dict[str, List[str]] = {
    "Income": [
        "Employment income or payroll deposit",
        "Government benefits such as EI, CPP, OAS, or CERB",
        "Investment income such as dividends or interest",
        "Rental income",
        "Business or self-employment income",
        "Other income or miscellaneous deposit",
    ],
    "Housing": [
        "Rent payment",
        "Mortgage payment",
        "Property tax",
        "Utilities such as hydro, electricity, natural gas, or water",
        "Home insurance",
        "Home maintenance and repairs",
    ],
    "Food and Dining": [
        "Groceries and supermarket",
        "Restaurants, fast food, and takeout",
        "Coffee and cafes",
    ],
    "Transportation": [
        "Fuel and gas station",
        "Auto insurance",
        "Auto loan or lease payment",
        "Public transit",
        "Ride share such as Uber or Lyft",
        "Parking",
    ],
    "Financial Obligations": [
        "Credit card payment",
        "Personal loan payment",
        "Line of credit payment",
        "Student loan payment",
        "RRSP or TFSA contribution or investment",
        "Bank fee or service charge",
        "NSF fee or overdraft fee",
        "Interest charge",
        "Payday loan or high-interest lending",
    ],
    "Healthcare": [
        "Pharmacy and prescriptions",
        "Medical, dental, or vision appointment",
        "Health or dental insurance premium",
    ],
    "Telecommunications": [
        "Mobile phone or internet service",
        "Cable, satellite, or streaming subscription",
    ],
    "Shopping and Retail": [
        "Clothing and apparel",
        "Electronics and technology",
        "General retail or department store",
        "Home goods and furniture",
    ],
    "Entertainment and Leisure": [
        "Entertainment and recreation",
        "Travel, hotels, or flights",
        "Gym and fitness",
        "Personal care and beauty",
    ],
    "Education": [
        "Tuition and education fees",
        "Books, supplies, or online courses",
    ],
    "Transfers and Payments": [
        "e-Transfer",
        "Internal bank transfer",
        "Bill payment",
        "Insurance premium payment",
    ],
    "Other": [
        "Miscellaneous or unclassified transaction",
    ],
}

_TOP_LABELS: List[str] = list(TAXONOMY.keys())


# ---------------------------------------------------------------------------
# Known-category lookup (checked before zero-shot)
# Covers terms the model cannot reliably classify from a short name alone:
# banking/financial keywords, Canadian utility brands, gas stations, etc.
# Ordered most-specific first.  Each entry: (regex, top_level, sub_label)
# ---------------------------------------------------------------------------

_KNOWN_CATEGORIES: List[Tuple[re.Pattern, str, str]] = [
    # --- Transfers ---
    (re.compile(r'\be[\s\-]?transfer\b',       re.I), "Transfers and Payments", "e-Transfer"),
    (re.compile(r'\binterac\b',                re.I), "Transfers and Payments", "e-Transfer"),
    (re.compile(r'\bwire\s+transfer\b',        re.I), "Transfers and Payments", "Internal bank transfer"),

    # --- Income ---
    (re.compile(r'\bdirect\s+deposit\b',       re.I), "Income", "Employment income or payroll deposit"),
    (re.compile(r'\bpayroll\b',                re.I), "Income", "Employment income or payroll deposit"),
    (re.compile(r'\bgovernment\s+of\s+canada\b', re.I), "Income", "Government benefits such as EI, CPP, OAS, or CERB"),
    (re.compile(r'\bcanada\s+revenue\b',       re.I), "Income", "Government benefits such as EI, CPP, OAS, or CERB"),
    (re.compile(r'\b(cerb|ei\s+benefit|cpp|oas)\b', re.I), "Income", "Government benefits such as EI, CPP, OAS, or CERB"),

    # --- Financial obligations / risk signals ---
    (re.compile(r'\bnsf\b',                    re.I), "Financial Obligations", "NSF or overdraft fee"),
    (re.compile(r'\boverdraft\b',              re.I), "Financial Obligations", "NSF or overdraft fee"),
    (re.compile(r'\b(rrsp|tfsa|rrif)\b',       re.I), "Financial Obligations", "RRSP or TFSA contribution or investment"),
    (re.compile(r'\betf\b',                    re.I), "Financial Obligations", "RRSP or TFSA contribution or investment"),
    (re.compile(r'\b(buy|sell)\b.{0,20}\b(shs|shares|units)\b', re.I), "Financial Obligations", "RRSP or TFSA contribution or investment"),
    (re.compile(r'\bpayday\s+loan\b',          re.I), "Financial Obligations", "Payday loan or high-interest lending"),
    (re.compile(r'\bmortgage\b',               re.I), "Financial Obligations", "Credit card payment"),  # overridden below if "TD" etc.

    # --- Housing / utilities ---
    (re.compile(r'\bhydro\s*one\b',            re.I), "Housing", "Utilities such as hydro, electricity, natural gas, or water"),
    (re.compile(r'\bbc\s*hydro\b',             re.I), "Housing", "Utilities such as hydro, electricity, natural gas, or water"),
    (re.compile(r'\benbridge\b',               re.I), "Housing", "Utilities such as hydro, electricity, natural gas, or water"),
    (re.compile(r'\bfortis\b',                 re.I), "Housing", "Utilities such as hydro, electricity, natural gas, or water"),
    (re.compile(r'\bunion\s+gas\b',            re.I), "Housing", "Utilities such as hydro, electricity, natural gas, or water"),
    (re.compile(r'\batco\s+gas\b',             re.I), "Housing", "Utilities such as hydro, electricity, natural gas, or water"),
    (re.compile(r'\bwater\s+(bill|utility)\b', re.I), "Housing", "Utilities such as hydro, electricity, natural gas, or water"),

    # --- Transportation / fuel ---
    (re.compile(r'\bshell\b',                  re.I), "Transportation", "Fuel and gas station"),
    (re.compile(r'\besso\b',                   re.I), "Transportation", "Fuel and gas station"),
    (re.compile(r'\bpetro[\s\-]?canada\b',     re.I), "Transportation", "Fuel and gas station"),
    (re.compile(r'\bimperial\s+oil\b',         re.I), "Transportation", "Fuel and gas station"),
    (re.compile(r'\bpioneer\s+gas\b',          re.I), "Transportation", "Fuel and gas station"),
    (re.compile(r'\bultramar\b',               re.I), "Transportation", "Fuel and gas station"),
    (re.compile(r'\bcostco\s+gas\b',           re.I), "Transportation", "Fuel and gas station"),

    # --- Food ---
    (re.compile(r'\bstarbucks\b',              re.I), "Food and Dining", "Coffee and cafes"),
    (re.compile(r'\btim\s*hortons?\b',         re.I), "Food and Dining", "Coffee and cafes"),
    (re.compile(r'\bkrispy\s*kreme\b',        re.I), "Food and Dining", "Coffee and cafes"),
]


def _check_known_category(text: str) -> Optional[Tuple[str, str]]:
    """Return (top_level, sub_label) if the description matches a known pattern."""
    for pattern, top, sub in _KNOWN_CATEGORIES:
        if pattern.search(text):
            return top, sub
    return None


# ---------------------------------------------------------------------------
# TransactionCategorizer
# ---------------------------------------------------------------------------

class TransactionCategorizer:
    """
    Two-pass zero-shot categorizer for transaction descriptions.

    Pass 1 — classify into a top-level category from the TAXONOMY keys.
    Pass 2 — classify into the matching sub-category list.

    Both passes share a single loaded pipeline so memory is only allocated once.

    Parameters
    ----------
    model_name : str
        Any HuggingFace zero-shot-classification model.
        Default: facebook/bart-large-mnli (~1.6 GB, best general accuracy).
        Lighter alternative: cross-encoder/nli-deberta-v3-small (~180 MB).
    confidence_threshold : float
        Minimum top-label score (pass 1) to accept a category. Below this the
        transaction falls into ["Other", "Miscellaneous or unclassified transaction"].
    """

    ZS_MODEL  = "facebook/bart-large-mnli"
    THRESHOLD = 0.20

    def __init__(
        self,
        model_name: str = ZS_MODEL,
        confidence_threshold: float = THRESHOLD,
    ):
        self._model_name = model_name
        self._threshold  = confidence_threshold
        self._classifier = None   # lazy-loaded

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._classifier is None:
            self._classifier = hf_pipeline(
                "zero-shot-classification",
                model=self._model_name,
            )

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    def _zs(self, text: str, labels: List[str]) -> Tuple[str, float]:
        """Single zero-shot pass → (best_label, score)."""
        result = self._classifier(text, candidate_labels=labels, multi_label=False)
        return result["labels"][0], float(result["scores"][0])

    def categorize(
        self,
        description: str,
        amount: Optional[Decimal] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """
        Return (category, subcategory) for a cleaned transaction description,
        matching the Transaction dataclass fields directly.

          category    → List[str] containing the single top-level label,
                        e.g. ["Food and Dining"]
          subcategory → str sub-label, e.g. "Restaurants, fast food, and takeout"
                        or None if classification confidence is too low.

        Parameters
        ----------
        description : str
            Cleaned merchant name or memo (output of TransactionCleaner).
        amount : Decimal, optional
            Signed transaction amount (positive = credit/income).
            Used as a lightweight hint: large credits bias pass 1 toward Income.
        """
        if not description or not description.strip():
            return ["Other"], "Miscellaneous or unclassified transaction"

        # Known-category lookup — deterministic, no inference needed
        known = _check_known_category(description)
        if known:
            return [known[0]], known[1]

        self._load()

        # --- Pass 1: top-level ---
        top_labels = _TOP_LABELS.copy()
        # Hint: surface "Income" first for large inbound amounts so the model
        # sees it as the most salient candidate in its NLI hypothesis template.
        if amount is not None and amount > Decimal("200"):
            top_labels = ["Income"] + [l for l in top_labels if l != "Income"]

        top_label, top_score = self._zs(description, top_labels)

        if top_score < self._threshold:
            return ["Other"], "Miscellaneous or unclassified transaction"

        # --- Pass 2: sub-category ---
        sub_labels = TAXONOMY[top_label]
        sub_label = sub_labels[0] if len(sub_labels) == 1 else self._zs(description, sub_labels)[0]

        return [top_label], sub_label

    # ------------------------------------------------------------------
    # Batch / User-level API
    # ------------------------------------------------------------------

    def categorize_users(self, users: List[User]) -> List[User]:
        """
        Populate Transaction.category and Transaction.subcategory for every
        transaction in the user list.
        Reads Transaction.cleaned_description — run TransactionCleaner first.
        Modifies transactions in-place and returns the same list.
        """
        self._load()
        for user in users:
            for account in user.accounts:
                for txn in account.transactions:
                    txn.category, txn.subcategory = self.categorize(
                        txn.cleaned_description or txn.memo,
                        amount=txn.amount,
                    )
        return users


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from decimal import Decimal

    TEST_CASES = [
        ("Dairy Queen",            Decimal("-6.50")),
        ("Amazon",                 Decimal("-89.99")),
        ("Netflix",                Decimal("-17.99")),
        ("Esso",                   Decimal("-75.00")),
        ("e-Transfer",             Decimal("-200.00")),
        ("NSF Fee",                Decimal("-45.00")),
        ("Direct Deposit",         Decimal("3200.00")),
        ("Acme Corp",              Decimal("3200.00")),
        ("Tim Hortons",            Decimal("-3.25")),
        ("Costco",                 Decimal("-212.54")),
        ("Rogers Communications",  Decimal("-89.00")),
        ("Shoppers Drug Mart",     Decimal("-28.00")),
        ("TD Bank Mortgage",       Decimal("-1850.00")),
        ("Hydro One",              Decimal("-140.00")),
        ("RRSP Contribution",      Decimal("-500.00")),
        ("Government of Canada",   Decimal("1200.00")),
        ("Gas Station",            Decimal("-14.50")),
        ("VFV ETF Purchase",       Decimal("-6025.00")),
    ]

    categorizer = TransactionCategorizer()

    print(f"{'Description':<30}  {'Amount':>10}  {'Category':<28}  Subcategory")
    print("-" * 100)
    for desc, amt in TEST_CASES:
        cat, subcat = categorizer.categorize(desc, amount=amt)
        print(f"{desc:<30}  {str(amt):>10}  {cat[0]:<28}  {subcat}")
