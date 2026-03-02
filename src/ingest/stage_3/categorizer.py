"""
Stage 3 - Transaction Categorizer

Assigns a two-level category to cleaned transaction descriptions using
semantic similarity (sentence-transformers/all-MiniLM-L6-v2).

Sub-labels from TAXONOMY are embedded once at load time.  At inference,
descriptions are batch-embedded and matched via cosine similarity — pure
numpy, no per-item inference loops.

Output format matches Transaction.category: List[str]
  e.g. ["Food and Dining", "Restaurants and Takeout"]

Usage:
    categorizer = TransactionCategorizer()
    txn.category = categorizer.categorize(txn.cleaned_description, txn.amount)

    # or bulk:
    categorizer.categorize_users(users)  # modifies in-place, run Cleaner first
"""

from __future__ import annotations

import os
import re
import sys
from decimal import Decimal
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from custom_dataclasses.user_data import User, Transaction
except ImportError:
    from ...custom_dataclasses.user_data import User, Transaction


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
        "Bars, pubs, and nightclubs",
        "Food delivery",
        "Alcohol stores",
        "Other food and dining",
    ],
    "Transportation": [
        "Fuel and gas station",
        "Auto insurance",
        "Auto loan or lease payment",
        "Public transit",
        "Ride share such as Uber or Lyft",
        "Parking",
        "Vehicle maintenance and repairs",
        "Other transportation",
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
        "Other financial obligation or risk signal",
    ],
    "Healthcare": [
        "Pharmacy and prescriptions",
        "Medical, dental, or vision appointment",
        "Health or dental insurance premium",
        "Other healthcare",
    ],
    "Telecommunications": [
        "Mobile phone or internet service",
        "Cable, satellite, or streaming subscription",
        "Other telecommunications",
    ],
    "Shopping and Retail": [
        "Clothing and apparel",
        "Electronics and technology",
        "General retail or department store",
        "Home goods and furniture",
        "Other shopping and retail",
    ],
    "Entertainment and Leisure": [
        "Entertainment and recreation",
        "Movies and theaters",
        "Concerts and live events",
        "Museums and cultural sites",
        "Travel, hotels, or flights",
        "Gym and fitness",
        "Personal care and beauty",
        "Other entertainment and leisure",
    ],
    "Education": [
        "Tuition and education fees",
        "Books, supplies, or online courses",
        "Student loan payment",
        "Other education",
    ],
    "Transfers and Payments": [
        "e-Transfer",
        "Internal bank transfer",
        "Bill payment",
        "Insurance premium payment",
        "Charitable donation",
        "Other transfer or payment",
    ],
    "Other": [
        "Miscellaneous or unclassified transaction",
    ],
}

# Flat list of every sub-label and a reverse map sub → top-level.
# Used for single-pass classification: classify directly into sub-labels,
# then derive the top-level from the reverse map.
_ALL_SUB_LABELS: List[str] = [sub for subs in TAXONOMY.values() for sub in subs]
_SUB_TO_TOP: Dict[str, str] = {
    sub: top
    for top, subs in TAXONOMY.items()
    for sub in subs
}
_INCOME_SUBS: List[str] = TAXONOMY["Income"]


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
    Semantic-similarity categorizer for transaction descriptions.

    Sub-labels from TAXONOMY are embedded once at load time.  At inference,
    descriptions are batch-embedded and matched via cosine similarity —
    pure numpy, no per-item inference loop.

    The model (all-MiniLM-L6-v2, ~80 MB) is already cached by the
    ColumnIdentifier stage so there is no extra download.

    Parameters
    ----------
    model_name : str
        Any sentence-transformers model.  Default: all-MiniLM-L6-v2.
    confidence_threshold : float
        Minimum cosine similarity to accept a label.  Below this the
        transaction falls into ["Other", "Miscellaneous or unclassified transaction"].
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    THRESHOLD  = 0.25

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        confidence_threshold: float = THRESHOLD,
    ):
        self._model_name  = model_name
        self._threshold   = confidence_threshold
        self._model: Optional[SentenceTransformer] = None
        self._label_vecs: Optional[np.ndarray]     = None  # (n_labels, dim), unit vectors
        self._income_idx: Optional[np.ndarray]     = None  # indices of Income sub-labels

    # ------------------------------------------------------------------
    # Lazy loader — embeds all sub-labels once
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        self._label_vecs = self._model.encode(
            _ALL_SUB_LABELS,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._income_idx = np.array(
            [i for i, lbl in enumerate(_ALL_SUB_LABELS) if lbl in _INCOME_SUBS],
            dtype=np.int32,
        )

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    def categorize(
        self,
        description: str,
        amount: Optional[Decimal] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """
        Return (category, subcategory) for a cleaned transaction description.

          category    → List[str] with the single top-level label
          subcategory → str sub-label, or None if similarity is below threshold
        """
        if not description or not description.strip():
            return ["Other"], "Miscellaneous or unclassified transaction"

        known = _check_known_category(description)
        if known:
            return [known[0]], known[1]

        self._load()
        import numpy as np
        assert self._model is not None
        assert self._label_vecs is not None
        assert self._income_idx is not None

        vec  = self._model.encode([description], normalize_embeddings=True)
        sims = (vec @ self._label_vecs.T)[0].copy()

        if amount is not None and amount > Decimal("200"):
            sims[self._income_idx] *= 1.5

        best_idx   = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < self._threshold:
            return ["Other"], "Miscellaneous or unclassified transaction"

        best_sub = _ALL_SUB_LABELS[best_idx]
        return [_SUB_TO_TOP.get(best_sub, "Other")], best_sub

    # ------------------------------------------------------------------
    # Batch / User-level API
    # ------------------------------------------------------------------

    def categorize_users(
        self,
        users: List[User],
        batch_size: int = 32,
        on_txn_progress: Optional[Callable[[int, int], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> List[User]:
        """
        Populate Transaction.category and Transaction.subcategory for every
        transaction in the user list.
        Reads Transaction.cleaned_description — run TransactionCleaner first.
        Modifies transactions in-place and returns the same list.

        Uses a single batched zero-shot inference call against all sub-labels.
        Known-category items are resolved immediately without inference.
        Top-level category is derived from the winning sub-label via reverse lookup.

        Parameters
        ----------
        batch_size : int
            Number of descriptions per inference batch (default 32).
        on_txn_progress : callable(done, total) | None
            Called after each transaction is resolved (known-category lookups
            and ML write-back both fire this).
        on_status : callable(msg) | None
            Called with a human-readable status string before the batch inference.
        """
        self._load()
        import numpy as np

        # --- Collect all transactions, resolving known-category ones immediately ---
        all_txn_count = sum(
            len(account.transactions)
            for user in users
            for account in (user.accounts or [])
        )
        resolved = 0
        pending: List[Tuple[Transaction, str, Optional[Decimal]]] = []  # (txn, desc, amount)

        for user in users:
            for account in (user.accounts or []):
                for txn in account.transactions:
                    desc = txn.cleaned_description or txn.memo
                    if not desc or not desc.strip():
                        txn.category    = ["Other"]
                        txn.subcategory = "Miscellaneous or unclassified transaction"
                        resolved += 1
                        if on_txn_progress:
                            on_txn_progress(resolved, all_txn_count)
                        continue
                    known = _check_known_category(desc)
                    if known:
                        txn.category    = [known[0]]
                        txn.subcategory = known[1]
                        resolved += 1
                        if on_txn_progress:
                            on_txn_progress(resolved, all_txn_count)
                    else:
                        pending.append((txn, desc, txn.amount))

        if not pending:
            return users

        assert self._model is not None
        assert self._label_vecs is not None
        assert self._income_idx is not None

        # --- Batch embed all pending descriptions ---
        if on_status:
            on_status(f"Classifying {len(pending):,} transactions…")
        descs   = [desc for _, desc, _ in pending]
        amounts = [amt  for _, _, amt  in pending]

        desc_vecs  = self._model.encode(
            descs,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        # sim_matrix[i, j] = cosine similarity of desc i to sub-label j
        sim_matrix = desc_vecs @ self._label_vecs.T  # (n_pending, n_labels)

        # Apply income boost for large inbound amounts (vectorised)
        for i, amt in enumerate(amounts):
            if amt is not None and amt > Decimal("200"):
                sim_matrix[i, self._income_idx] *= 1.5

        best_indices = np.argmax(sim_matrix, axis=1)   # (n_pending,)
        best_scores  = sim_matrix[np.arange(len(pending)), best_indices]

        # --- Write results back ---
        for i, (txn, _, _) in enumerate(pending):
            best_score = float(best_scores[i])
            if best_score < self._threshold:
                txn.category    = ["Other"]
                txn.subcategory = "Miscellaneous or unclassified transaction"
            else:
                best_sub        = _ALL_SUB_LABELS[int(best_indices[i])]
                txn.category    = [_SUB_TO_TOP.get(best_sub, "Other")]
                txn.subcategory = best_sub
            if on_txn_progress:
                on_txn_progress(resolved + i + 1, all_txn_count)

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
