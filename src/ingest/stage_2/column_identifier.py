from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer


class ColumnIdentifier:
    """
    Uses a local sentence transformer to semantically match CSV column headers
    to canonical field roles without hardcoded alias lists.

    ANCHORS lists representative column header names for each role. At inference,
    each candidate column is compared against every example for a role and the
    MAX similarity is used — this avoids the dilution that occurs when averaging
    many concepts into a single embedding (e.g. "Action" or "Details" would be
    lost in a long concatenated anchor string).

    The model (~80 MB) is downloaded once on first use and cached by
    sentence-transformers in ~/.cache/torch/sentence_transformers/.
    """

    # Representative column header names for each canonical role.
    # Use real-world examples that actually appear in bank/brokerage CSV exports.
    ANCHORS: Dict[str, List[str]] = {
        "date": [
            "Date", "Transaction Date", "Posted Date", "Post Date",
            "Trade Date", "TradeDate", "Settlement Date", "Value Date",
            "Txn Date", "Txn_Date", "Transaction_Date",
        ],
        "description": [
            "Description", "Transaction Description", "Memo", "Details",
            "Narration", "Payee", "Merchant", "Merchant Name", "Action",
            "Note", "Reference", "Particulars",
        ],
        "amount": [
            "Amount", "Net Amount", "Total Amount", "Transaction Amount",
            "Value", "Net Value",
        ],
        "debit": [
            "Debit", "Withdrawal", "Withdrawals", "Money Out",
            "Debit Amount", "Dr", "Amount Debited",
        ],
        "credit": [
            "Credit", "Deposit", "Deposits", "Money In",
            "Credit Amount", "Cr", "Amount Credited",
        ],
        "shares": [
            "Shares", "Units", "Quantity", "Qty", "Number of Shares",
            "Volume", "Shares Traded",
        ],
        "price": [
            "Price", "Unit Price", "Share Price", "Cost",
            "Price Per Share", "Market Price", "Rate",
        ],
        "symbol": [
            "Symbol", "Ticker", "Security", "Security Name",
            "Stock", "Instrument", "CUSIP", "ISIN",
        ],
    }

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    THRESHOLD  = 0.45   # higher is safe now that we use max rather than mean similarity

    _model: Optional[SentenceTransformer]      = None
    _anchor_keys: Optional[List[str]]          = None
    _anchor_vecs: Optional[np.ndarray]         = None  # (total_examples, dim)
    _role_slices: Optional[Dict[str, slice]]   = None  # role → slice into _anchor_vecs

    @classmethod
    def _load(cls) -> None:
        if cls._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        cls._model       = SentenceTransformer(cls.MODEL_NAME)
        cls._anchor_keys = list(cls.ANCHORS.keys())

        all_examples: List[str] = []
        slices: Dict[str, slice] = {}
        for role in cls._anchor_keys:
            start = len(all_examples)
            all_examples.extend(cls.ANCHORS[role])
            slices[role] = slice(start, len(all_examples))

        # normalize=True → unit vectors, so dot product == cosine similarity
        cls._anchor_vecs = cls._model.encode(
            all_examples,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        cls._role_slices = slices

    @classmethod
    def identify(cls, columns: List[str]) -> Dict[str, Optional[str]]:
        """
        Returns {canonical_role: original_column_name_or_None} for every role.
        Each physical column is assigned to at most one role (greedy, best-first).
        """
        cls._load()
        import numpy as np
        assert cls._model is not None
        assert cls._anchor_vecs is not None
        assert cls._anchor_keys is not None
        assert cls._role_slices is not None

        col_vecs = cls._model.encode(
            columns,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # full_sim[i, k] = cosine similarity of columns[i] to anchor example k
        full_sim = col_vecs @ cls._anchor_vecs.T

        # role_sim[i, j] = MAX similarity of columns[i] to any example for role j
        role_sim = np.stack(
            [full_sim[:, cls._role_slices[r]].max(axis=1) for r in cls._anchor_keys],
            axis=1,
        )

        result: Dict[str, Optional[str]] = {k: None for k in cls._anchor_keys}
        used: set[int] = set()

        for j, role in enumerate(cls._anchor_keys):
            scores = role_sim[:, j]
            for i in np.argsort(scores)[::-1]:
                if int(i) in used:
                    continue
                if scores[i] >= cls.THRESHOLD:
                    result[role] = columns[int(i)]
                    used.add(int(i))
                break

        return result
