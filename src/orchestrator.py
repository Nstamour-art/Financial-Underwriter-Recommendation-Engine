"""
Main pipeline orchestrator for the Financial Underwriter Recommendation Engine.

Wires together all stages for a single end-to-end run:

    Stage 1  - data ingestion   Plaid API  → raw dict
                                CSV upload → DataFrames
    Stage 2  - conversion       raw data   → List[User]
    Stage 3  - NLP cleaning     User memos → cleaned_description
    Stage 3  - categorization   descriptions → category / subcategory
    Process  - underwriting     User → LLM → decision + products

Designed for Streamlit integration
-----------------------------------
on_progress callback
    Signature: on_progress(step: str, detail: str, fraction: float) -> None
    fraction is 0.0-1.0. Map directly to st.progress() and a status label.

OrchestratorResult helpers
    .monthly_spending()   → dict[month_str, dict[category, float]]
    .monthly_cash_flow()  → dict[month_str, dict[income|expenses|net, float]]
    Both return plain dicts so Streamlit can do pd.DataFrame(result.monthly_spending())

Typical Streamlit usage
    progress_bar = st.progress(0)
    status      = st.empty()

    def on_progress(step, detail, fraction):
        progress_bar.progress(fraction)
        status.text(f"{step}: {detail}")

    result = orchestrator.run_from_csv(csv_data, on_progress=on_progress)
    # or:
    result = orchestrator.run_from_plaid_data(plaid_data, on_progress=on_progress)
"""

import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Dict, List, Literal, Optional

# Add src/ to the path so all sibling packages resolve as absolute imports
# regardless of whether this file is run as a script or imported as a module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_dataclasses import User, CSVFileInput
from ingest import (
    PlaidAPI,
    CSVDataConverter,
    PlaidDataConverter,
    TransactionCleaner,
    TransactionCategorizer,
)
from process import UnderwritingOrchestrator, PRODUCTS
    

# ---------------------------------------------------------------------------
# Progress step definitions
# Step name, human-readable label, fraction of total progress when complete.
# ---------------------------------------------------------------------------

_STEPS = [
    ("load",       "Loading data",                0.10),
    ("convert",    "Building user profiles",       0.20),
    ("clean",      "Cleaning transaction memos",   0.60),
    ("categorize", "Categorizing transactions",    0.85),
    ("underwrite", "Running underwriting analysis",0.97),
    ("done",       "Complete",                     1.00),
]

_STEP_FRACTIONS: Dict[str, float] = {name: frac for name, _, frac in _STEPS}
_STEP_LABELS:    Dict[str, str]   = {name: lbl  for name, lbl, _ in _STEPS}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Record of a single pipeline step, consumed by the Streamlit status panel."""
    name:            str
    label:           str
    status:          Literal["completed", "failed", "skipped"]
    detail:          str  = ""
    elapsed_seconds: float = 0.0


@dataclass
class OrchestratorResult:
    """
    Full result of one user's underwriting pipeline run.

    Attributes
    ----------
    user
        Fully processed User object (cleaned + categorized transactions).
        Pass to Streamlit charting helpers via .monthly_spending() etc.
    underwriting
        Parsed LLM response dict:
            score, decision, summary, rejection_reason,
            recommended_products, provider
        None if the underwriting step failed.
    steps
        Ordered list of StepResult objects — drives the Streamlit progress log.
    total_elapsed_seconds
        Wall-clock time for the entire run.
    error
        Top-level error message if a critical stage failed, otherwise None.
    """
    user:                    Optional[User]
    underwriting:            Optional[dict]
    steps:                   List[StepResult]
    total_elapsed_seconds:   float
    error:                   Optional[str] = None

    # ------------------------------------------------------------------
    # Chart-data helpers for Streamlit
    # ------------------------------------------------------------------

    def monthly_spending(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate debit transaction amounts by calendar month and top-level
        category.  Returns a nested dict:
            { "2024-01": {"Housing": 1200.0, "Food and Dining": 340.0, ...}, ... }

        Usage in Streamlit:
            df = pd.DataFrame(result.monthly_spending()).T.fillna(0)
            st.bar_chart(df)
        """
        totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for account in (self.user.accounts or [] if self.user else []):
            for txn in account.transactions:
                if txn.amount >= 0:
                    continue
                month = txn.date.strftime("%Y-%m")
                cat   = txn.category[0] if txn.category else "Other"
                totals[month][cat] += float(abs(txn.amount))
        return {m: dict(cats) for m, cats in sorted(totals.items())}

    def monthly_cash_flow(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate income and expenses by calendar month.  Returns:
            { "2024-01": {"income": 4500.0, "expenses": 3200.0, "net": 1300.0}, ... }

        Usage in Streamlit:
            df = pd.DataFrame(result.monthly_cash_flow()).T
            st.line_chart(df)
        """
        income:   Dict[str, float] = defaultdict(float)
        expenses: Dict[str, float] = defaultdict(float)

        for account in (self.user.accounts or [] if self.user else []):
            for txn in account.transactions:
                month = txn.date.strftime("%Y-%m")
                if txn.amount > 0:
                    income[month] += float(txn.amount)
                else:
                    expenses[month] += float(abs(txn.amount))

        all_months = sorted(set(income) | set(expenses))
        return {
            m: {
                "income":   round(income.get(m, 0.0),   2),
                "expenses": round(expenses.get(m, 0.0), 2),
                "net":      round(income.get(m, 0.0) - expenses.get(m, 0.0), 2),
            }
            for m in all_months
        }

    def category_totals(self) -> Dict[str, float]:
        """
        Total debit spend per top-level category over the entire period.
        Useful for a pie chart or summary table.

        Returns { "Housing": 14400.0, "Food and Dining": 4080.0, ... }
        """
        totals: Dict[str, float] = defaultdict(float)
        for account in (self.user.accounts or [] if self.user else []):
            for txn in account.transactions:
                if txn.amount < 0:
                    cat = txn.category[0] if txn.category else "Other"
                    totals[cat] += float(abs(txn.amount))
        return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    End-to-end pipeline runner.

    Parameters
    ----------
    api_provider : str or None
        Force a specific LLM provider: "anthropic" | "openai".
        None = auto-detect from environment keys (Anthropic preferred).
    cleaner_llm_path : str or None
        Path to the Qwen 3 GGUF model used by TransactionCleaner stage 3.
        None = auto-download on first use.  Pass a path to avoid re-downloading.
    n_gpu_layers : int
        GPU layers for llama_cpp in the memo cleaner (0 = CPU-only).
    ner_threshold : float
        BERT NER confidence threshold in TransactionCleaner (default 0.85).
    categorizer_model : str or None
        Override the zero-shot model used by TransactionCategorizer.
        None = use class default (deberta-v3-small-zeroshot-v1.1-all-33).
    categorizer_batch_size : int
        Batch size for TransactionCategorizer.categorize_users().
    """

    def __init__(
        self,
        api_provider:          Optional[str]   = None,
        cleaner_llm_path:      Optional[str]   = None,
        n_gpu_layers:          int             = 0,
        ner_threshold:         float           = 0.85,
        categorizer_model:     Optional[str]   = None,
        categorizer_batch_size: int            = 32,
    ):
        self._api_provider           = api_provider
        self._cleaner_llm_path       = cleaner_llm_path
        self._n_gpu_layers           = n_gpu_layers
        self._ner_threshold          = ner_threshold
        self._categorizer_model      = categorizer_model
        self._categorizer_batch_size = categorizer_batch_size

        # Lazy-initialised so models are only loaded when a run actually starts.
        self._cleaner:    Optional[TransactionCleaner]    = None
        self._categorizer: Optional[TransactionCategorizer] = None
        self._underwriter: Optional[UnderwritingOrchestrator] = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_cleaner(self) -> TransactionCleaner:
        if self._cleaner is None:
            self._cleaner = TransactionCleaner(
                llm_model_path=self._cleaner_llm_path,
                n_gpu_layers=self._n_gpu_layers,
                ner_threshold=self._ner_threshold,
            )
        return self._cleaner

    def _get_categorizer(self) -> TransactionCategorizer:
        if self._categorizer is None:
            kwargs = {}
            if self._categorizer_model:
                kwargs["model_name"] = self._categorizer_model
            self._categorizer = TransactionCategorizer(**kwargs)
        return self._categorizer

    def _get_underwriter(self) -> UnderwritingOrchestrator:
        if self._underwriter is None:
            self._underwriter = UnderwritingOrchestrator(products=PRODUCTS.products)
        return self._underwriter

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_from_csv(
        self,
        csv_data: Dict[str, List[CSVFileInput]],
        on_progress: Optional[Callable[[str, str, float], None]] = None,
    ) -> List[OrchestratorResult]:
        """
        Run the full pipeline from CSV file inputs.

        Parameters
        ----------
        csv_data : dict
            { user_id: [CSVFileInput, ...] }
            Build with CSVLoader; account_type on each CSVFileInput is set
            by the Streamlit dropdown the user selects for each uploaded file.
        on_progress : callable, optional
            on_progress(step_name, detail, fraction) called after each stage.

        Returns
        -------
        List[OrchestratorResult]  — one entry per user in csv_data.
        """
        def _load(_on_progress):
            users = CSVDataConverter.convert(csv_data)
            _emit(_on_progress, "load", f"Loaded {len(users)} user(s) from CSV")
            return users

        return self._run(_load, on_progress)

    def run_from_plaid_data(
        self,
        plaid_data: Dict[str, dict],
        on_progress: Optional[Callable[[str, str, float], None]] = None,
    ) -> List[OrchestratorResult]:
        """
        Run the full pipeline from pre-fetched Plaid data.

        Parameters
        ----------
        plaid_data : dict
            Raw output of PlaidAPI.fetch_all_sandbox_transactions() or
            PlaidAPI.get_item_data() — keyed by user label.
            Format: { user_id: {"accounts": [...], "transactions_by_account": {...}} }
        on_progress : callable, optional

        Returns
        -------
        List[OrchestratorResult]  — one entry per user in plaid_data.
        """
        def _load(_on_progress):
            users = PlaidDataConverter.convert(plaid_data)
            _emit(_on_progress, "load", f"Loaded {len(users)} user(s) from Plaid")
            return users

        return self._run(_load, on_progress)

    def run_from_plaid_sandbox(
        self,
        start_date: Optional[date] = None,
        end_date:   Optional[date] = None,
        on_progress: Optional[Callable[[str, str, float], None]] = None,
    ) -> List[OrchestratorResult]:
        """
        Convenience method: fetch sandbox transactions from Plaid then run
        the full pipeline.  Requires PLAID_* environment variables.

        Parameters
        ----------
        start_date : date, optional   defaults to 365 days ago
        end_date   : date, optional   defaults to today
        on_progress : callable, optional
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        def _load(_on_progress):
            api = PlaidAPI()
            raw = api.fetch_all_sandbox_transactions(start_date=start_date, end_date=end_date)
            users = PlaidDataConverter.convert(raw)
            _emit(_on_progress, "load", f"Fetched {len(users)} user(s) from Plaid sandbox")
            return users

        return self._run(_load, on_progress)

    # ------------------------------------------------------------------
    # Core pipeline (shared by all entry points)
    # ------------------------------------------------------------------

    def _run(
        self,
        load_fn: Callable,
        on_progress: Optional[Callable[[str, str, float], None]],
    ) -> List[OrchestratorResult]:
        """
        Execute the pipeline for a batch of users returned by load_fn.
        Returns one OrchestratorResult per user.
        """
        wall_start = time.monotonic()
        results: List[OrchestratorResult] = []

        # --- Stage 1 + 2: load & convert ---
        t0 = time.monotonic()
        try:
            users: List[User] = load_fn(on_progress)
            load_elapsed = time.monotonic() - t0
            load_step = StepResult(
                name="load", label=_STEP_LABELS["load"],
                status="completed", elapsed_seconds=load_elapsed,
            )
        except Exception as exc:
            load_step = StepResult(
                name="load", label=_STEP_LABELS["load"],
                status="failed", detail=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )
            _emit(on_progress, "load", f"Failed: {exc}", _STEP_FRACTIONS["load"])
            # Cannot continue without data.
            return [OrchestratorResult(
                user=User(user_id="unknown"),
                underwriting=None,
                steps=[load_step],
                total_elapsed_seconds=time.monotonic() - wall_start,
                error=f"Data load failed: {exc}",
            )]

        _emit(on_progress, "convert", "Converting to user profiles", _STEP_FRACTIONS["convert"])

        # --- Stage 3a: clean memos ---
        _emit(on_progress, "clean", "Loading NLP models…", _STEP_FRACTIONS["load"])
        t0 = time.monotonic()
        try:
            cleaner = self._get_cleaner()
            cleaner.clean_users(users)
            clean_elapsed = time.monotonic() - t0
            clean_step = StepResult(
                name="clean", label=_STEP_LABELS["clean"],
                status="completed",
                detail=f"Cleaned memos for {sum(len(u.accounts or []) for u in users)} accounts",
                elapsed_seconds=clean_elapsed,
            )
            _emit(on_progress, "clean",
                    f"Cleaned {sum(len(a.transactions) for u in users for a in (u.accounts or []))} transactions",
                    _STEP_FRACTIONS["clean"])
        except Exception as exc:
            clean_step = StepResult(
                name="clean", label=_STEP_LABELS["clean"],
                status="failed", detail=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )
            _emit(on_progress, "clean", f"Failed: {exc}", _STEP_FRACTIONS["clean"])

        # --- Stage 3b: categorize ---
        _emit(on_progress, "categorize", "Loading zero-shot classifier…", _STEP_FRACTIONS["clean"])
        t0 = time.monotonic()
        try:
            categorizer = self._get_categorizer()
            categorizer.categorize_users(users, batch_size=self._categorizer_batch_size)
            cat_elapsed = time.monotonic() - t0
            cat_step = StepResult(
                name="categorize", label=_STEP_LABELS["categorize"],
                status="completed",
                detail=f"Categorized transactions",
                elapsed_seconds=cat_elapsed,
            )
            _emit(on_progress, "categorize", "Categorization complete", _STEP_FRACTIONS["categorize"])
        except Exception as exc:
            cat_step = StepResult(
                name="categorize", label=_STEP_LABELS["categorize"],
                status="failed", detail=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )
            _emit(on_progress, "categorize", f"Failed: {exc}", _STEP_FRACTIONS["categorize"])

        # --- Process: underwrite each user ---
        underwriter = self._get_underwriter()

        for user in users:
            user_steps = [load_step, clean_step, cat_step]
            t0 = time.monotonic()
            _emit(on_progress, "underwrite",
                    f"Analysing {user.user_id}…", _STEP_FRACTIONS["categorize"])
            try:
                uw_result = underwriter.run(user, api_provider=self._api_provider)
                uw_elapsed = time.monotonic() - t0
                uw_step = StepResult(
                    name="underwrite", label=_STEP_LABELS["underwrite"],
                    status="completed",
                    detail=(
                        f"Decision: {uw_result.get('decision', '?')}  "
                        f"Score: {uw_result.get('score', '?')}"
                    ),
                    elapsed_seconds=uw_elapsed,
                )
                _emit(on_progress, "done", "Pipeline complete", _STEP_FRACTIONS["done"])
            except Exception as exc:
                uw_result = None
                uw_step = StepResult(
                    name="underwrite", label=_STEP_LABELS["underwrite"],
                    status="failed", detail=str(exc),
                    elapsed_seconds=time.monotonic() - t0,
                )
                _emit(on_progress, "underwrite", f"Failed: {exc}", _STEP_FRACTIONS["underwrite"])

            user_steps.append(uw_step)
            results.append(OrchestratorResult(
                user=user,
                underwriting=uw_result,
                steps=user_steps,
                total_elapsed_seconds=time.monotonic() - wall_start,
                error=uw_step.detail if uw_step.status == "failed" else None,
            ))

        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _emit(
    cb: Optional[Callable[[str, str, float], None]],
    step: str,
    detail: str,
    fraction: Optional[float] = None,
) -> None:
    """Fire the progress callback if one was supplied."""
    if cb is None:
        return
    frac = fraction if fraction is not None else _STEP_FRACTIONS.get(step, 0.0)
    cb(_STEP_LABELS.get(step, step), detail, frac)
    

if __name__ == "__main__":
    # Quick smoke test of the orchestrator with the Plaid sandbox data.
    orchestrator = Orchestrator()
    results = orchestrator.run_from_plaid_sandbox()
    for res in results:
        print(f"User: {res.user.user_id if res.user else 'unknown'}, Decision: {res.underwriting.get('decision') if res.underwriting else 'N/A'}, Error: {res.error}")
