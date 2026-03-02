"""
Audit trail for underwriting decisions.

Persists every pipeline decision to a local SQLite database so each assessment
is traceable, reproducible, and available for review.  The human override path
also writes through this layer so the full decision lifecycle is captured.

Database location: ``<project_root>/audit_logs/audit.db``
Table:             ``decisions``
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Project root — three levels up from src/process/audit.py
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB   = _PROJECT_ROOT / "audit_logs" / "audit.db"


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------

@dataclass
class AuditRecord:
    """One row in the audit table."""

    # Identity
    timestamp:              str                 # ISO-8601 UTC
    session_id:             str                 # Streamlit session ID
    user_id:                str                 # Client / user label

    # Input fingerprint (no PII stored — only a hash)
    input_hash:             str                 # SHA-256 of serialised user data

    # Model outputs
    score:                  Optional[int]       = None
    decision:               Optional[str]       = None
    confidence_llm:         Optional[float]     = None
    confidence_heuristic:   Optional[float]     = None
    confidence:             Optional[float]     = None
    recommended_products:   List[str]           = field(default_factory=list)

    # Data completeness signals used by the heuristic
    data_signals:           Dict[str, bool]     = field(default_factory=dict)

    # Model provenance
    model_versions:         Dict[str, str]      = field(default_factory=dict)

    # Human-in-the-loop
    human_override:         Optional[str]       = None
    override_reason:        Optional[str]       = None

    # DB primary key (populated after INSERT)
    row_id:                 Optional[int]       = None


# ---------------------------------------------------------------------------
# Audit log (SQLite backend)
# ---------------------------------------------------------------------------

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS decisions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp             TEXT    NOT NULL,
    session_id            TEXT    NOT NULL,
    user_id               TEXT    NOT NULL,
    input_hash            TEXT    NOT NULL,
    score                 INTEGER,
    decision              TEXT,
    confidence_llm        REAL,
    confidence_heuristic  REAL,
    confidence            REAL,
    recommended_products  TEXT,
    data_signals          TEXT,
    model_versions        TEXT,
    human_override        TEXT,
    override_reason       TEXT
)"""


class AuditLog:
    """
    Thin wrapper around a SQLite database for decision audit records.

    Parameters
    ----------
    db_path : str or Path, optional
        Defaults to ``<project_root>/audit_logs/audit.db``.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._path = Path(db_path) if db_path else _DEFAULT_DB
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,   # accessed from Streamlit + bg thread
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, rec: AuditRecord) -> int:
        """Insert a record and return the row id."""
        cur = self._conn.execute(
            """\
            INSERT INTO decisions (
                timestamp, session_id, user_id, input_hash,
                score, decision,
                confidence_llm, confidence_heuristic, confidence,
                recommended_products, data_signals, model_versions,
                human_override, override_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.timestamp,
                rec.session_id,
                rec.user_id,
                rec.input_hash,
                rec.score,
                rec.decision,
                rec.confidence_llm,
                rec.confidence_heuristic,
                rec.confidence,
                json.dumps(rec.recommended_products),
                json.dumps(rec.data_signals),
                json.dumps(rec.model_versions),
                rec.human_override,
                rec.override_reason,
            ),
        )
        self._conn.commit()
        rec.row_id = cur.lastrowid
        return cur.lastrowid  # type: ignore[return-value]

    def update_override(
        self,
        row_id: int,
        override: str,
        reason: str,
    ) -> None:
        """Amend a record with a human override decision and rationale."""
        self._conn.execute(
            "UPDATE decisions SET human_override = ?, override_reason = ? WHERE id = ?",
            (override, reason, row_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _row_to_record(self, row: tuple) -> AuditRecord:
        return AuditRecord(
            row_id=row[0],
            timestamp=row[1],
            session_id=row[2],
            user_id=row[3],
            input_hash=row[4],
            score=row[5],
            decision=row[6],
            confidence_llm=row[7],
            confidence_heuristic=row[8],
            confidence=row[9],
            recommended_products=json.loads(row[10]) if row[10] else [],
            data_signals=json.loads(row[11]) if row[11] else {},
            model_versions=json.loads(row[12]) if row[12] else {},
            human_override=row[13],
            override_reason=row[14],
        )

    def get_all(self) -> List[AuditRecord]:
        """Return every record, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM decisions ORDER BY timestamp DESC"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_for_user(self, user_id: str) -> List[AuditRecord]:
        """Return all records for a specific user, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM decisions WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_by_id(self, row_id: int) -> Optional[AuditRecord]:
        """Return a single record by primary key."""
        row = self._conn.execute(
            "SELECT * FROM decisions WHERE id = ?", (row_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hash_user_data(user) -> str:
    """
    SHA-256 fingerprint of a User object's financial data.
    Deterministic but doesn't store PII.
    """
    parts: list[str] = [user.user_id or ""]
    for acct in (user.accounts or []):
        parts.append(f"{acct.type}:{acct.subtype}:{acct.current_balance}")
        for txn in acct.transactions:
            parts.append(f"{txn.date}:{txn.amount}:{txn.memo[:30]}")
    raw = "|".join(parts).encode()
    return hashlib.sha256(raw).hexdigest()


def compute_heuristic_confidence(user, uw: dict) -> tuple[float, dict[str, bool]]:
    """
    Return (confidence, signals_dict) based on data completeness.

    Each signal is a boolean indicating whether that quality criterion is met.
    The heuristic confidence is the fraction of signals that are True.
    """
    all_txns = [
        txn
        for acct in (user.accounts or [])
        for txn in acct.transactions
    ]
    txn_count = len(all_txns)

    # Date span in months
    if all_txns:
        dates = [t.date for t in all_txns]
        span_days = (max(dates) - min(dates)).days
    else:
        span_days = 0

    # Income detected (at least one positive transaction)
    has_income = any(t.amount > 0 for t in all_txns)

    signals: dict[str, bool] = {
        "multiple_accounts":    len(user.accounts or []) >= 2,
        "has_transactions":     txn_count > 0,
        "sufficient_history":   txn_count > 30,
        "12mo_coverage":        span_days >= 330,
        "income_detected":      has_income,
        "valid_risk_level":     uw.get("decision") in ("approved", "conditional", "rejected"),
        "products_recommended": bool(uw.get("recommended_products")),
        "score_in_range":       300 <= (uw.get("score") or 0) <= 900,
    }

    met = sum(signals.values())
    return round(met / len(signals), 3), signals
