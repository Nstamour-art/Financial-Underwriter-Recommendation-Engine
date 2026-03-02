"""
Wealthsimple Underwriting Tool — Streamlit front end.

Run with:
    streamlit run main.py

Fonts and theme are configured in .streamlit/config.toml.
Static assets (fonts, images) are served from static/ at the project root.
"""

from __future__ import annotations

import html as html_lib
import io
import json
import os
import sys
import threading
import time
import uuid
from datetime import date, timedelta
from typing import Dict, List, Literal, Optional

_BadgeColor = Literal['red', 'orange', 'yellow', 'blue', 'green', 'violet', 'gray', 'grey', 'primary']

import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_UI_DIR    = os.path.dirname(os.path.abspath(__file__))  # src/ui/
_SRC       = os.path.dirname(_UI_DIR)                    # src/
_PROJ_ROOT = os.path.dirname(_SRC)                       # project root
sys.path.insert(0, _SRC)

from custom_dataclasses import CSVFileInput
from ingest.stage_1.plaid_api import PlaidAPI
from orchestrator import Orchestrator, OrchestratorResult
from process import PRODUCTS
from process.audit import AuditLog
from ui.styles import (
    CATEGORY_COLORS, CF_COLORS, PLOT_FONT, SIDEBAR_CSS,
    WS_BG, WS_TEXT, WS_TEXT_LIGHT, WS_BORDER,
    WS_GREEN, WS_AMBER, WS_SUCCESS, WS_ERROR,
    get_color, colors_for,
)


# ---------------------------------------------------------------------------
# Account type options
# ---------------------------------------------------------------------------
_ACCOUNT_TYPES: Dict[str, Optional[tuple]] = {
    "Auto-detect":        None,
    "Chequing":           ("depository", "checking"),
    "Savings":            ("depository", "savings"),
    "Credit Card":        ("credit",     "credit card"),
    "Student Loan":       ("loan",       "student"),
    "Mortgage":           ("loan",       "mortgage"),
    "Investment (TFSA)":  ("investment", "tfsa"),
    "Investment (RRSP)":  ("investment", "rrsp"),
    "Investment (Other)": ("investment", "brokerage"),
}


# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WS Underwriting",
    page_icon=os.path.join(_PROJ_ROOT, "static", "imgs",
                                "icon1.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    for k, v in {
        "results":           None,
        "running":           False,
        "selected_idx":      0,
        "override_decision": None,
        "override_reason":   "",
        "employee_notes":    "",
        "pending_config":    None,
        "_pipeline_ctx":     None,
        "csv_client_id":     str(uuid.uuid4()),
        # CSV dialog state
        "csv_client_name":   "",
        "csv_file_config":   {},   # {filename: {"acct_type": str, "group": str}}
        "csv_uploads":       [],   # list of UploadedFile objects
        "csv_ready":         False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# CSV Upload Dialog
# ---------------------------------------------------------------------------

_DEFAULT_GROUPS = [f"Account {chr(65 + i)}" for i in range(10)]  # A–J


@st.dialog("Configure CSV Upload", width="large")
def _csv_upload_dialog() -> None:
    """
    Modal dialog for CSV upload configuration.

    Lets the user:
        1.  Enter a client name (cosmetic — UUID stays internal).
        2.  Upload one or more CSV statement files.
        3.  For each file, pick an account type and an *account group*.
            Files sharing the same group are merged (concatenated) before
            processing — this is how multiple monthly statements for the
            same account become a single 12-month history.
    """
    st.caption(
        "Upload transaction CSVs and organise them into account groups. "
        "Files in the **same group** are merged together — use this to "
        "combine monthly statements for one account."
    )

    client_name = st.text_input(
        "Client name",
        value=st.session_state.csv_client_name,
        placeholder="e.g. Jane Smith",
    ) or ""

    st.divider()

    uploaded = st.file_uploader(
        "Transaction files",
        type=["csv"],
        accept_multiple_files=True,
        key="dialog_csv_uploader",
    )

    if uploaded:
        st.divider()
        st.markdown("##### File configuration")
        st.caption(
            "Assign each file an **account type**. Files with the same type "
            "are automatically grouped into the same account (merged). "
            "Use the **group** column to override — e.g. if you have two "
            "separate chequing accounts."
        )

        # Build existing config or defaults for new files
        existing: dict = dict(st.session_state.csv_file_config)
        for f in uploaded:
            if f.name not in existing:
                existing[f.name] = {"acct_type": "Auto-detect", "group": ""}

        # --- First pass: collect account types so we can auto-assign groups ---
        acct_types_chosen: Dict[str, str] = {}
        for idx, f in enumerate(uploaded):
            prev = existing.get(f.name, {})
            acct_types_chosen[f.name] = st.session_state.get(
                f"dlg_acct_{idx}", prev.get("acct_type", "Auto-detect")
            )

        # Auto-assign groups: each distinct account type gets its own group letter.
        # Files with "Auto-detect" each get their own group since we can't know
        # whether they belong together.
        _type_to_group: Dict[str, str] = {}
        _auto_counter = 0
        auto_groups: Dict[str, str] = {}
        for fname in [f.name for f in uploaded]:
            atype = acct_types_chosen[fname]
            if atype == "Auto-detect":
                # Each auto-detect file gets its own group
                auto_groups[fname] = _DEFAULT_GROUPS[_auto_counter % len(_DEFAULT_GROUPS)]
                _auto_counter += 1
            else:
                if atype not in _type_to_group:
                    _type_to_group[atype] = _DEFAULT_GROUPS[
                        (_auto_counter + len(_type_to_group)) % len(_DEFAULT_GROUPS)
                    ]
                auto_groups[fname] = _type_to_group[atype]
        # Deduplicate: ensure type-based groups don't collide with auto-detect ones
        # by assigning them sequentially from the pool
        used_indices: set[int] = set()
        _type_to_group_final: Dict[str, str] = {}
        next_idx = 0
        for fname in [f.name for f in uploaded]:
            atype = acct_types_chosen[fname]
            if atype == "Auto-detect":
                while next_idx in used_indices:
                    next_idx += 1
                auto_groups[fname] = _DEFAULT_GROUPS[next_idx % len(_DEFAULT_GROUPS)]
                used_indices.add(next_idx)
                next_idx += 1
            else:
                if atype not in _type_to_group_final:
                    while next_idx in used_indices:
                        next_idx += 1
                    _type_to_group_final[atype] = _DEFAULT_GROUPS[next_idx % len(_DEFAULT_GROUPS)]
                    used_indices.add(next_idx)
                    next_idx += 1
                auto_groups[fname] = _type_to_group_final[atype]

        # --- Render config rows ---
        file_config: Dict[str, dict] = {}
        for idx, f in enumerate(uploaded):
            prev = existing.get(f.name, {})
            cols = st.columns([3, 2, 2])
            with cols[0]:
                st.text(f.name)
            with cols[1]:
                acct_type = st.selectbox(
                    "Type",
                    list(_ACCOUNT_TYPES.keys()),
                    index=list(_ACCOUNT_TYPES.keys()).index(
                        prev.get("acct_type", "Auto-detect")),
                    key=f"dlg_acct_{idx}",
                    label_visibility="collapsed",
                )
            with cols[2]:
                # Use the auto-assigned group as default, but let user override
                default_group = auto_groups.get(f.name, _DEFAULT_GROUPS[0])
                group = st.selectbox(
                    "Group",
                    _DEFAULT_GROUPS,
                    index=_DEFAULT_GROUPS.index(default_group)
                        if default_group in _DEFAULT_GROUPS else 0,
                    key=f"dlg_group_{idx}",
                    label_visibility="collapsed",
                )
            file_config[f.name] = {"acct_type": acct_type, "group": group}

        # Validate: warn if different account types share a group
        groups_check: Dict[str, set[str]] = {}
        for fname, cfg in file_config.items():
            groups_check.setdefault(cfg["group"], set()).add(cfg["acct_type"])
        for g, types in groups_check.items():
            real_types = types - {"Auto-detect"}
            if len(real_types) > 1:
                st.warning(
                    f"**{g}** has mixed account types ({', '.join(sorted(real_types))}). "
                    f"Files in the same group should be statements for the same account.",
                    icon="⚠️",
                )

        # Show merge preview
        groups: Dict[str, List[str]] = {}
        for fname, cfg in file_config.items():
            groups.setdefault(cfg["group"], []).append(fname)
        merged_groups = {g: fnames for g, fnames in groups.items() if len(fnames) > 1}
        if merged_groups:
            st.divider()
            st.markdown("##### Merge preview")
            for g, fnames in merged_groups.items():
                st.info(
                    f"**{g}** — {len(fnames)} files will be merged: "
                    + ", ".join(f"`{n}`" for n in fnames),
                    icon="🔗",
                )
    else:
        file_config = {}

    st.divider()

    c_cancel, c_confirm = st.columns(2)
    with c_cancel:
        if st.button("Cancel", width='stretch'):
            st.rerun()
    with c_confirm:
        confirm_disabled = not uploaded or not client_name.strip()
        if st.button("Confirm", type="primary", width='stretch',
                        disabled=confirm_disabled):
            st.session_state.csv_client_name = client_name.strip()
            st.session_state.csv_uploads     = uploaded
            st.session_state.csv_file_config = file_config
            st.session_state.csv_ready       = True
            st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    logo_path = os.path.join(_PROJ_ROOT, "static", "imgs",
                                "ws-wordmark-refresh.3499def3.svg")
    if os.path.exists(logo_path):
        st.logo(logo_path)

    with st.sidebar:
        st.title("Settings & Controls")
        st.caption("Underwriting Demo — select data source, date range, and LLM provider, then run the analysis.")
        st.divider()

        st.text(body="Data Source")
        source = st.radio("source", ["CSV Upload", "Plaid Sandbox"],
                            label_visibility="collapsed")
        config: dict = {"source": source}

        if source == "CSV Upload":
            config["client_id"]   = st.session_state.csv_client_id
            config["client_name"] = st.session_state.csv_client_name
            config["uploaded"]    = st.session_state.csv_uploads
            config["file_config"] = st.session_state.csv_file_config

            if st.button("Configure CSV Upload", width='stretch'):
                _csv_upload_dialog()

            # Show summary of configured files
            if st.session_state.csv_ready and st.session_state.csv_uploads:
                name = st.session_state.csv_client_name or "Unnamed"
                n_files = len(st.session_state.csv_uploads)
                groups = set(
                    v["group"] for v in st.session_state.csv_file_config.values()
                )
                st.success(
                    f"**{name}** — {n_files} file(s) in {len(groups)} account group(s)",
                    icon="📁",
                )
        else:
            sandbox_users = PlaidAPI.list_sandbox_users()  # [(label, username), ...]
            all_labels = [label for label, _ in sandbox_users]

            st.text(body="Sandbox user")
            selected_label: Optional[str] = st.selectbox(
                "users",
                options=[None] + all_labels,
                format_func=lambda x: "Select a user…" if x is None else x,
                label_visibility="collapsed",
                disabled=not all_labels,
            )
            config["selected_users"] = [selected_label] if selected_label else []
            config["start_date"] = date.today() - timedelta(days=365)
            config["end_date"]   = date.today()

        st.divider()

        st.text(body="LLM Provider")
        prov = st.radio("prov",
                        ["Auto (prefer Anthropic)", "Anthropic", "OpenAI"],
                        label_visibility="collapsed")
        config["api_provider"] = {
            "Auto (prefer Anthropic)": None,
            "Anthropic":               "anthropic",
            "OpenAI":                  "openai",
        }[prov]

        with st.expander("Advanced"):
            config["gpu_layers"]    = st.number_input(
                "GPU layers (llama_cpp)", 0, 128, 0)
            config["ner_threshold"] = st.slider(
                "NER confidence threshold", 0.5, 1.0, 0.85, 0.01)
            config["batch_size"]    = st.number_input(
                "Categorizer batch size", 4, 128, 32)

        st.divider()

        run_disabled = (
            (source == "CSV Upload"
                and not st.session_state.csv_ready)
            or (source == "Plaid Sandbox"
                and not config.get("selected_users"))
        )
        if st.button("Run Analysis", type="primary",
                        disabled=run_disabled or st.session_state.running):
            st.session_state.pending_config = config
            st.rerun()

        if st.session_state.results is not None:
            if st.button("Clear & start over"):
                st.session_state.results           = None
                st.session_state.override_decision = None
                st.session_state.employee_notes    = ""
                st.session_state.csv_client_name   = ""
                st.session_state.csv_file_config   = {}
                st.session_state.csv_uploads       = []
                st.session_state.csv_ready         = False
                st.session_state.csv_client_id     = str(uuid.uuid4())
                st.rerun()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _build_csv_data(config: dict) -> Dict[str, List[CSVFileInput]]:
    """
    Build the ``{client_id: [CSVFileInput, ...]}`` dict consumed by the
    orchestrator, **merging files that share an account group**.

    Files assigned to the same group have their DataFrames concatenated
    (row-wise) so multiple monthly statements become one continuous history.
    """
    client_id   = config["client_id"].strip()
    file_config = config.get("file_config", {})
    uploaded    = config.get("uploaded", [])

    # Index uploaded files by name for quick lookup
    file_by_name = {f.name: f for f in uploaded}

    # Group filenames by account group
    groups: Dict[str, List[str]] = {}
    for fname, cfg in file_config.items():
        groups.setdefault(cfg["group"], []).append(fname)

    # Build one CSVFileInput per group (merging DataFrames when > 1 file)
    csv_inputs: List[CSVFileInput] = []
    for group_label, fnames in groups.items():
        dfs: List[pd.DataFrame] = []
        acct_type_key = "Auto-detect"
        representative_name = fnames[0]

        for fname in fnames:
            uf = file_by_name.get(fname)
            if uf is None:
                continue
            dfs.append(pd.read_csv(
                io.StringIO(uf.getvalue().decode("utf-8-sig"))
            ))
            # Use the account type from file_config (last one wins if mixed,
            # but they should all be the same within a group)
            acct_type_key = file_config[fname].get("acct_type", "Auto-detect")

        if not dfs:
            continue

        merged_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # Use the group label as a friendlier filepath stem when merging
        display_name = (
            group_label if len(fnames) > 1 else representative_name
        )

        csv_inputs.append(CSVFileInput(
            filepath=display_name,
            df=merged_df,
            account_type=_ACCOUNT_TYPES[acct_type_key],
        ))

    # Fall back: if file_config is empty (shouldn't happen), treat each file individually
    if not csv_inputs:
        for f in uploaded:
            csv_inputs.append(CSVFileInput(
                filepath=f.name,
                df=pd.read_csv(io.StringIO(f.getvalue().decode("utf-8-sig"))),
                account_type=None,
            ))

    return {client_id: csv_inputs}


def _check_csv_date_coverage(csv_data: Dict[str, List[CSVFileInput]]) -> Optional[str]:
    """
    Scan each CSV DataFrame for a date column and check that the combined
    date range spans at least 365 days.  Returns an error string if not,
    or None if coverage is sufficient (or indeterminate).
    """
    all_dates: List[date] = []
    for file_inputs in csv_data.values():
        for fi in file_inputs:
            for col in fi.df.columns:
                try:
                    parsed = pd.to_datetime(fi.df[col], errors="coerce", dayfirst=False)
                    if parsed.notna().sum() >= len(fi.df) * 0.7:
                        all_dates.extend(parsed.dropna().dt.date.tolist())
                        break  # found the date column for this file
                except Exception:
                    continue

    if not all_dates:
        return None  # can't determine dates; let the pipeline handle it

    days = (max(all_dates) - min(all_dates)).days
    if days < 365:
        months = max(1, round(days / 30.44))
        plural = "s" if months != 1 else ""
        return (
            f"The uploaded files only cover {days} days (~{months} month{plural}). "
            f"At least 12 months of transaction history is required for accurate underwriting. "
            f"Please upload additional files to extend the date range."
        )
    return None


# ---------------------------------------------------------------------------
# Pipeline runner (threaded)
# ---------------------------------------------------------------------------

def _start_pipeline(config: dict) -> None:
    """
    Run the full pipeline in a daemon thread.

    Models are loaded lazily on first call and kept alive in the Streamlit
    process via @st.cache_resource, so subsequent runs skip cold-start entirely.
    """
    ctx: dict = {
        "done":    False,
        "results": None,
        "error":   None,
        "frac":    0.0,
        "label":   "Initialising…",
        "detail":  "",
    }
    st.session_state._pipeline_ctx     = ctx
    st.session_state.running           = True
    st.session_state.results           = None
    st.session_state.override_decision = None
    st.session_state.employee_notes    = ""

    # Read uploaded files on the main thread (UploadedFile objects can't cross threads)
    csv_data = _build_csv_data(config) if config["source"] == "CSV Upload" else None

    if csv_data is not None:
        coverage_error = _check_csv_date_coverage(csv_data)
        if coverage_error:
            ctx["error"] = coverage_error
            ctx["done"]  = True
            return

    def on_progress(label: str, detail: str, frac: float) -> None:
        ctx["frac"]   = frac
        ctx["label"]  = label
        ctx["detail"] = detail

    def on_sub_progress(done: int, total: int) -> None:
        ctx["detail"] = f"{done:,} / {total:,} transactions"

    def _run() -> None:
        try:
            orchestrator = Orchestrator(
                api_provider           = config.get("api_provider"),
                n_gpu_layers           = config.get("gpu_layers", 0),
                ner_threshold          = config.get("ner_threshold", 0.85),
                categorizer_batch_size = config.get("batch_size", 32),
            )

            if config["source"] == "CSV Upload" and csv_data:
                results = orchestrator.run_from_csv(
                    csv_data,
                    on_progress     = on_progress,
                    on_sub_progress = on_sub_progress,
                    client_name     = config.get("client_name") or None,
                )
            else:
                _sd = config.get("start_date")
                start_date = _sd if isinstance(_sd, date) else date.fromisoformat(_sd) if _sd else None
                _ed = config.get("end_date")
                end_date = _ed if isinstance(_ed, date) else date.fromisoformat(_ed) if _ed else None
                results = orchestrator.run_from_plaid_sandbox(
                    start_date      = start_date,
                    end_date        = end_date,
                    selected_users  = config.get("selected_users") or None,
                    on_progress     = on_progress,
                    on_sub_progress = on_sub_progress,
                )

            ctx["results"] = results

        except Exception as exc:
            ctx["error"] = str(exc)
        finally:
            ctx["done"] = True

    threading.Thread(target=_run, daemon=True).start()


@st.cache_resource(show_spinner=False)
def _warm_models() -> bool:
    """
    Pre-load NLP models into cache so the first pipeline run has no cold-start.
    Called once per Streamlit process via a background thread at startup.
    Returns True when complete (cached value signals subsequent calls instantly).
    """
    try:
        from ingest.stage_3.cleaner import TransactionCleaner
        from ingest.stage_3.categorizer import TransactionCategorizer
        _c = TransactionCleaner()
        _c._load_spacy()
        _c._load_ner()
        _cat = TransactionCategorizer()
        _cat._load()
    except Exception:
        pass
    return True


def _render_progress() -> None:
    """Render a single pipeline progress bar with live stage + transaction text."""
    ctx    = st.session_state.get("_pipeline_ctx") or {}
    frac   = float(ctx.get("frac", 0.0))
    label  = ctx.get("label", "Initialising…")
    detail = ctx.get("detail", "")

    text = label + (f" — {detail}" if detail else "")
    st.progress(frac, text=text)


def _error_result(msg: str) -> OrchestratorResult:
    from custom_dataclasses import User
    from orchestrator import StepResult
    return OrchestratorResult(
        user=None, underwriting=None,
        steps=[StepResult("run", "Pipeline", "failed", msg, 0)],
        total_elapsed_seconds=0, error=msg,
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _md(text: str) -> str:
    """Escape characters that Streamlit's Markdown renderer treats specially."""
    return text.replace("$", r"\$")


def _score_color(score: Optional[int]) -> str:
    if score is None:  return WS_TEXT_LIGHT
    if score >= 750:   return WS_GREEN
    if score >= 650:   return "#7ab648"
    if score >= 550:   return WS_AMBER
    return WS_ERROR


def _decision_color(decision: Optional[str]) -> _BadgeColor:
    d = (decision or "").lower()
    if d == "approved":    return "green"
    if d == "conditional": return "orange"
    if d == "rejected":    return "red"
    return "gray"


def _risk_signals(result: OrchestratorResult) -> List[tuple]:
    if not result.user or not result.user.accounts:
        return [("No data available", "medium")]

    signals: List[tuple] = []
    for acct in result.user.accounts:
        for txn in acct.transactions:
            desc = (txn.cleaned_description or txn.memo or "").lower()
            if "nsf" in desc or "non-sufficient" in desc:
                signals.append(("NSF / Non-sufficient funds", "high"))
                break
            if "overdraft" in desc:
                signals.append(("Overdraft activity", "high"))
                break

    cf = result.monthly_cash_flow()
    if cf:
        neg   = sum(1 for m in cf.values() if m["net"] < 0)
        total = len(cf)
        if neg > total / 2:
            signals.append((f"Negative net cash flow in {neg}/{total} months", "high"))
        elif neg > 0:
            signals.append((f"Negative net cash flow in {neg}/{total} months", "medium"))

    for acct in result.user.accounts:
        if acct.type == "credit" and acct.credit_limit and acct.current_balance:
            util = float(abs(acct.current_balance)) / float(acct.credit_limit)
            if util > 0.9:
                signals.append((f"Credit utilisation {util:.0%} — {acct.name}", "high"))
            elif util > 0.7:
                signals.append((f"Credit utilisation {util:.0%} — {acct.name}", "medium"))

    if not signals:
        signals.append(("No significant risk signals detected", "low"))
    return signals


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

def _tab_overview(result: OrchestratorResult) -> None:
    uw         = result.underwriting or {}
    score      = uw.get("score")
    decision   = uw.get("decision")
    summary    = uw.get("summary", "")
    rej_reason = uw.get("rejection_reason")
    provider   = uw.get("provider", "—")

    if result.error and not uw:
        st.error(f"Pipeline error: {result.error}")
        return

    c_gauge, c_info, c_meta = st.columns([1, 2, 1])

    with c_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score or 300,
            number={
                "font": {
                    "size": 36,
                    "color": _score_color(score),
                    "family": "The Future, Helvetica Neue, sans-serif",
                },
                "valueformat": "d",
            },
            gauge={
                "axis": {
                    "range": [300, 900],
                    "tickwidth": 1,
                    "tickcolor": WS_BORDER,
                    "tickfont": {"size": 8, "color": WS_TEXT_LIGHT},
                },
                "bar":       {"color": _score_color(score), "thickness": 0.22},
                "bgcolor":   "#ffffff",
                "borderwidth": 0,
                "steps": [
                    {"range": [300, 549], "color": "#fce8e1"},
                    {"range": [549, 649], "color": "#fef3c7"},
                    {"range": [649, 749], "color": "#eef5eb"},
                    {"range": [749, 900], "color": "#e8f1e4"},
                ],
                "threshold": {
                    "line":      {"color": _score_color(score), "width": 4},
                    "thickness": 0.75,
                    "value":     score or 300,
                },
            },
        ))
        fig.update_layout(
            height=200,
            margin=dict(t=20, b=10, l=15, r=15),
            font={"family": "The Future, Helvetica Neue, sans-serif"},
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width='stretch',
                        config={"displayModeBar": False})
        st.caption("Underwriting Score")

    with c_info:
        user_label = result.user.name or result.user.user_id if result.user else "—"
        st.subheader(user_label)
        if decision:
            st.badge(decision.title(), color=_decision_color(decision))
        if summary:
            st.write(_md(summary))
        if rej_reason:
            st.error(f"**Rejection reason:** {_md(rej_reason)}")

    with c_meta:
        st.metric("LLM provider", provider.title())
        st.metric("Runtime", f"{result.total_elapsed_seconds:.1f}s")

    # --- Highlight product card ---
    rec_products = uw.get("recommended_products", [])
    top_reason   = uw.get("top_product_reason")
    if rec_products:
        top_name = rec_products[0]
        # Look up the Product object for richer detail
        _prod_obj = next(
            (p for p in PRODUCTS.products if p.name.lower() == top_name.lower()),
            None,
        )
        with st.container(border=True):
            _hl_left, _hl_right = st.columns([1, 3])
            with _hl_left:
                st.markdown(
                    '<p style="font-size:12px;color:#686664;margin:0 0 4px 0;">' 
                    'TOP RECOMMENDED PRODUCT</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"### {html_lib.escape(top_name)}")
                if _prod_obj:
                    st.badge(_prod_obj.type.title(), color="green")
            with _hl_right:
                if _prod_obj:
                    st.markdown(
                        f'<p style="margin:0">{html_lib.escape(_prod_obj.description)}</p>',
                        unsafe_allow_html=True,
                    )
                if top_reason:
                    st.info(top_reason, icon="\u2728")
                if _prod_obj and _prod_obj.min_annual_income:
                    st.caption(
                        f"Minimum annual income: **${_prod_obj.min_annual_income:,}**"
                    )

    st.divider()

    cf = result.monthly_cash_flow()
    if cf:
        avg_inc = sum(m["income"]   for m in cf.values()) / len(cf)
        avg_exp = sum(m["expenses"] for m in cf.values()) / len(cf)
        avg_net = avg_inc - avg_exp
        est_ann = avg_inc * 12
    else:
        avg_inc = avg_exp = avg_net = est_ann = 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg monthly income",   f"${avg_inc:,.0f}")
    m2.metric("Avg monthly expenses", f"${avg_exp:,.0f}")
    m3.metric("Avg monthly net",      f"${avg_net:,.0f}")
    m4.metric("Est. annual income",   f"${est_ann:,.0f}")

    st.divider()

    c_acct, c_risk = st.columns([3, 2])

    with c_acct:
        st.subheader("Accounts")
        if result.user and result.user.accounts:
            for acct in result.user.accounts:
                bal = float(acct.current_balance or 0)
                with st.container(border=True):
                    col_name, col_bal = st.columns([3, 1])
                    with col_name:
                        st.write(f"**{acct.name}**")
                        meta = (f"{acct.type} / {acct.subtype or '—'}"
                                f" · {len(acct.transactions)} transactions")
                        if acct.credit_limit and float(acct.credit_limit) > 0:
                            util = abs(bal) / float(acct.credit_limit)
                            meta += f" · {util:.0%} utilisation"
                        st.caption(meta)
                    with col_bal:
                        st.write(f"**${abs(bal):,.2f}**")

    with c_risk:
        st.subheader("Risk Signals")
        sev_color: Dict[str, _BadgeColor] = {"high": "red", "medium": "orange", "low": "green"}
        for label, sev in _risk_signals(result):
            st.badge(label, color=sev_color[sev])

    st.divider()

    with st.expander("Pipeline log"):
        for step in result.steps:
            icon = "✓" if step.status == "completed" else "✗"
            det  = f" — {step.detail}" if step.detail else ""
            st.write(f"{icon} **{step.label}**{det} `{step.elapsed_seconds:.2f}s`")


# ---------------------------------------------------------------------------
# Tab: Spending
# ---------------------------------------------------------------------------

def _tab_spending(result: OrchestratorResult) -> None:
    cf  = result.monthly_cash_flow()
    sp  = result.monthly_spending()
    cat = result.category_totals()

    if not cf:
        st.info("No transaction data available.")
        return

    st.subheader("Monthly Cash Flow")
    cf_df   = pd.DataFrame(cf).T.reset_index().rename(columns={"index": "Month"})
    cf_long = cf_df.melt(id_vars="Month", value_vars=["income", "expenses", "net"],
                            var_name="Type", value_name="CAD")
    fig_cf = px.line(
        cf_long, x="Month", y="CAD", color="Type",
        color_discrete_map=CF_COLORS,
        markers=True, template="plotly_white",
    )
    fig_cf.update_layout(
        font=PLOT_FONT,
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor=WS_BG, paper_bgcolor=WS_BG,
        margin=dict(t=10, b=30), height=280,
    )
    st.plotly_chart(fig_cf, width='stretch', config={"displayModeBar": False})

    if sp:
        st.subheader("Monthly Spending by Category")
        sp_df = (pd.DataFrame(sp).T.fillna(0)
                    .reset_index().rename(columns={"index": "Month"}))
        cats = [c for c in sp_df.columns if c != "Month"]
        sp_long = sp_df.melt(id_vars="Month", value_vars=cats,
                                var_name="Category", value_name="CAD")
        _sp_domain, _sp_range = colors_for(cats)
        chart_sp = (
            alt.Chart(sp_long)
            .mark_bar()
            .encode(
                x=alt.X("Month:N", sort=None),
                y=alt.Y("CAD:Q", stack="zero", title="CAD"),
                color=alt.Color("Category:N",
                    scale=alt.Scale(domain=_sp_domain, range=_sp_range),
                    legend=alt.Legend(orient="bottom")),
                tooltip=["Month", "Category", alt.Tooltip("CAD:Q", format="$,.0f")],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_sp, width="stretch")

    if cat:
        st.subheader("Spending Breakdown")
        cat_df = (pd.DataFrame([{"Category": k, "CAD": v} for k, v in cat.items()])
                    .sort_values("CAD", ascending=False)
                    .reset_index(drop=True))
        _pie_colors = [get_color(c) for c in cat_df["Category"]]
        c_pie, c_tbl = st.columns([2, 1])
        with c_pie:
            fig_pie = px.pie(
                cat_df, names="Category", values="CAD", hole=0.44,
                template="plotly_white",
                color_discrete_sequence=_pie_colors,
            )
            fig_pie.update_traces(textinfo="percent+label", textfont_size=10)
            fig_pie.update_layout(
                font=PLOT_FONT, showlegend=False,
                plot_bgcolor=WS_BG, paper_bgcolor=WS_BG,
                margin=dict(t=5, b=5, l=5, r=5), height=320,
            )
            st.plotly_chart(fig_pie, width='stretch',
                            config={"displayModeBar": False})
        with c_tbl:
            st.dataframe(
                cat_df.assign(CAD=cat_df["CAD"].map("${:,.2f}".format))
                        .rename(columns={"CAD": "Amount (CAD)"}),
                width='stretch', hide_index=True,
            )


# ---------------------------------------------------------------------------
# Tab: Products
# ---------------------------------------------------------------------------

def _product_card(product, *, highlighted: bool = False) -> str:
    """Return an HTML card string for a single product."""
    bg = ("background:#e6f7f4; border:1px solid #00C8A0;"
            if highlighted else "background:#ffffff; border:1px solid #e4e2e1;")
    income = (f'<p style="margin:8px 0 0 0;font-size:12px;color:#686664;">'
                f'Min income: ${product.min_annual_income:,}/yr</p>'
                if product.min_annual_income else "")
    return f"""
    <div style="{bg} border-radius:10px; padding:16px; margin:8px 0;
                    height:300px; box-sizing:border-box; overflow:hidden;">
        <span style="background:#e4e2e1;color:#686664;padding:5px 10px;
                        border-radius:10px;font-size:12px;">
            {html_lib.escape(product.type)}
        </span>
        <h4 style="margin:10px 0 6px 0;font-size:16px;">{html_lib.escape(product.name)}</h4>
        <p style="margin:0;font-size:13px;color:#32302f;">{html_lib.escape(product.description)}</p>
        {income}
    </div>"""


def _render_product_grid(products, *, highlighted: bool = False) -> None:
    """Lay out a list of products in a 3-column grid."""
    cols = st.columns(3, gap="small")
    for i, product in enumerate(products):
        with cols[i % 3]:
            st.markdown(_product_card(product, highlighted=highlighted),
                        unsafe_allow_html=True)


def _tab_products(result: OrchestratorResult) -> None:
    uw  = result.underwriting or {}
    rec = {p.lower() for p in uw.get("recommended_products", [])}

    recommended = [p for p in PRODUCTS.products if p.name.lower() in rec]
    others      = [p for p in PRODUCTS.products if p.name.lower() not in rec]

    # --- Recommended section ---
    st.markdown("#### Recommended by LLM")
    if recommended:
        _render_product_grid(recommended, highlighted=True)
    else:
        st.info("No products recommended by LLM.", icon="ℹ️")

    # --- Remaining products ---
    if others:
        st.markdown("#### Other Products")
        _render_product_grid(others, highlighted=False)


# ---------------------------------------------------------------------------
# Tab: Transactions
# ---------------------------------------------------------------------------

def _tab_transactions(result: OrchestratorResult) -> None:
    if not result.user or not result.user.accounts:
        st.info("No transaction data.")
        return

    rows = [
        {
            "Date":        txn.date,
            "Account":     acct.name,
            "Description": txn.cleaned_description or txn.memo,
            "Raw Memo":    txn.memo,
            "Category":    txn.category[0] if txn.category else "—",
            "Subcategory": txn.subcategory or "—",
            "Type":        txn.transaction_type.title(),
            "Amount":      float(txn.amount),
        }
        for acct in result.user.accounts
        for txn in acct.transactions
    ]
    if not rows:
        st.info("No transactions found.")
        return

    df = pd.DataFrame(rows).sort_values("Date", ascending=False)

    fc1, fc2, fc3, fc4 = st.columns([2, 2, 1, 1])
    with fc1:
        search = st.text_input("Search", placeholder="e.g. Starbucks",
                                label_visibility="collapsed")
    with fc2:
        cat_f = st.selectbox("Category",
                                ["All"] + sorted(df["Category"].unique().tolist()),
                                label_visibility="collapsed")
    with fc3:
        type_f = st.selectbox("Type", ["All", "Debit", "Credit"],
                                label_visibility="collapsed")
    with fc4:
        acct_f = st.selectbox("Account",
                                ["All"] + sorted(df["Account"].unique().tolist()),
                                label_visibility="collapsed")

    mask = pd.Series(True, index=df.index)
    if search:  mask &= df["Description"].str.contains(search, case=False, na=False)
    if cat_f  != "All": mask &= df["Category"] == cat_f
    if type_f != "All": mask &= df["Type"]     == type_f
    if acct_f != "All": mask &= df["Account"]  == acct_f

    filtered = df[mask].copy()
    filtered["Amount"] = filtered["Amount"].map("${:,.2f}".format)

    st.caption(f"{len(filtered):,} of {len(df):,} transactions")
    st.dataframe(filtered, width='stretch', hide_index=True, height=470)


# ---------------------------------------------------------------------------
# Tab: Decision Review  (replaces legacy Employee Tools)
# ---------------------------------------------------------------------------

def _tab_review(result: OrchestratorResult) -> None:
    uw = result.underwriting or {}
    uid = result.user.user_id if result.user else "result"

    # ── Section A: Human Decision Boundary ──────────────────────────────
    st.subheader("Human Decision Point")
    st.markdown(
        "**This system recommends, but it does not make final approval.** "
        "Final credit decisions must remain with a qualified underwriter because:"
    )
    st.markdown(
        "1. **Regulatory accountability** -  Human oversight of model decisions is required on a regular review and auditing basis.\n"
        "2. **Context the model cannot see** - Life events, verbal disclosures, and relationship history are invisible to the pipeline.\n"
        "3. **Fairness auditing** - A human reviewer can catch proxy discrimination the model may encode."
    )

    confidence     = float(uw.get("confidence", 0.0))
    decision       = uw.get("decision", "unknown")

    if confidence < 0.6 or decision == "conditional":
        st.error(
            f"Escalation required — Combined confidence: {confidence:.0%}, "
            f"Decision: {decision.title()}.  "
            "This application should be reviewed by a senior underwriter before any commitment."
        )
    elif confidence < 0.8:
        st.warning(
            f"Borderline confidence ({confidence:.0%}).  "
            "The system's recommendation should be validated against the applicant's full profile."
        )
    else:
        st.success(
            f"High confidence ({confidence:.0%}).  "
            "System recommendation can be accepted with standard spot-check procedures."
        )

    st.divider()

    # ── Section B: Confidence Breakdown ─────────────────────────────────
    st.subheader("Confidence Breakdown")

    c_llm, c_heur, c_comb = st.columns(3)
    c_llm.metric("LLM self-reported",  f"{uw.get('confidence_llm', 0):.0%}")
    c_heur.metric("Heuristic (data)",  f"{uw.get('confidence_heuristic', 0):.0%}")
    c_comb.metric("Combined",          f"{confidence:.0%}")

    signals: dict = uw.get("data_signals", {})
    if signals:
        st.caption("Data completeness signals")
        _SIGNAL_LABELS = {
            "multiple_accounts":  "2+ accounts linked",
            "has_transactions":   "Transaction data present",
            "sufficient_history": ">30 transactions",
            "12mo_coverage":      "12-month date coverage",
            "income_detected":    "Income stream detected",
            "valid_risk_level":   "Valid risk classification",
            "products_recommended": "Products recommended",
            "score_in_range":     "Score within 300–900",
        }
        cols = st.columns(4)
        for i, (key, met) in enumerate(signals.items()):
            label = _SIGNAL_LABELS.get(key, key)
            icon  = "✓" if met else "✗"
            with cols[i % 4]:
                if met:
                    st.badge(f"{icon} {label}", color="green")
                else:
                    st.badge(f"{icon} {label}", color="red")

    st.divider()

    # ── Section C: Override & Audit ─────────────────────────────────────
    st.subheader("Decision Override")
    st.caption(
        "Record a manual override. The original model decision and your "
        "override are both preserved in the audit trail."
    )
    current = uw.get("decision", "—")
    choice  = st.selectbox(
        "Decision",
        [f"Keep model decision ({current})", "approved", "conditional", "rejected"],
        key="review_override_select",
    )
    if not choice.startswith("Keep"):
        st.session_state.override_decision = choice
        reason = st.text_area(
            "Override rationale (required)",
            value=st.session_state.override_reason,
            placeholder="Explain why the model decision is being changed…",
            height=80,
            key="review_override_reason",
        )
        st.session_state.override_reason = reason or ""
        st.warning(f"Decision overridden to **{choice}**")

        # Persist override to audit DB
        if result.audit_id and reason:
            try:
                AuditLog().update_override(result.audit_id, choice, reason)
            except Exception:
                pass
    else:
        st.session_state.override_decision = None
        st.session_state.override_reason   = ""

    st.divider()

    st.subheader("Internal Notes")
    notes = st.text_area(
        "notes",
        value=st.session_state.employee_notes,
        placeholder="Underwriting notes, client context…",
        height=120,
        label_visibility="collapsed",
        key="review_notes",
    )
    st.session_state.employee_notes = notes

    st.divider()

    # ── Audit history ───────────────────────────────────────────────────
    st.subheader("Audit History")
    try:
        records = AuditLog().get_for_user(uid)
        if records:
            rows = [
                {
                    "Timestamp":  r.timestamp[:19].replace("T", " "),
                    "Score":      r.score,
                    "Decision":   (r.human_override or r.decision or "—").title(),
                    "Confidence": f"{(r.confidence or 0):.0%}",
                    "Override":   r.human_override.title() if r.human_override else "—",
                    "Reason":     r.override_reason or "—",
                }
                for r in records
            ]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        else:
            st.caption("No prior audit records for this client.")
    except Exception:
        st.caption("Audit log unavailable.")

    st.divider()

    # ── Export ──────────────────────────────────────────────────────────
    st.subheader("Export")
    export = {
        "user_id":              uid,
        "underwriting":         uw,
        "override_decision":    st.session_state.override_decision,
        "override_reason":      st.session_state.override_reason,
        "employee_notes":       st.session_state.employee_notes,
        "audit_id":             result.audit_id,
        "confidence": {
            "llm":       uw.get("confidence_llm"),
            "heuristic": uw.get("confidence_heuristic"),
            "combined":  uw.get("confidence"),
        },
        "data_signals":         signals,
        "pipeline_steps": [
            {"name": s.name, "status": s.status,
                "detail": s.detail, "elapsed_s": s.elapsed_seconds}
            for s in result.steps
        ],
    }

    c_json, c_csv = st.columns(2)
    with c_json:
        st.download_button(
            "Download JSON report",
            data=json.dumps(export, indent=2, default=str),
            file_name=f"underwriting_{uid}.json",
            mime="application/json",
        )
    with c_csv:
        if result.user and result.user.accounts:
            txn_rows = [
                {
                    "date":        str(txn.date),
                    "account":     acct.name,
                    "description": txn.cleaned_description or txn.memo,
                    "category":    txn.category[0] if txn.category else "",
                    "subcategory": txn.subcategory or "",
                    "type":        txn.transaction_type,
                    "amount":      float(txn.amount),
                }
                for acct in result.user.accounts
                for txn in acct.transactions
            ]
            st.download_button(
                "Download transactions CSV",
                data=pd.DataFrame(txn_rows).to_csv(index=False),
                file_name=f"transactions_{uid}.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------

def _welcome() -> None:
    st.title("Wealthsimple Underwriting Tool", text_alignment='center')
    st.markdown(
        """
        This internal tool is designed to assist underwriters by providing data-driven insights and recommendations based on clients' financial data. It leverages advanced machine learning models to analyze transaction history, identify risk signals, and suggest suitable financial products.
        """,
        text_alignment='center'
    )
    splash_path = os.path.join(_PROJ_ROOT, "static", "imgs", "videoframe_4033.png")
    if os.path.exists(splash_path):
        with open(splash_path, "rb") as _f:
            st.image(_f.read(), width='stretch')