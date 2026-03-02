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
    initial_sidebar_state="expanded",
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
        "employee_notes":    "",
        "pending_config":    None,
        "_pipeline_ctx":     None,
        "csv_client_id":     str(uuid.uuid4()),
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
            config["client_id"] = st.session_state.csv_client_id

            st.text(body="Transaction files")
            uploaded = st.file_uploader(
                "csv", type=["csv"], accept_multiple_files=True,
                label_visibility="collapsed",
            )
            config["uploaded"] = uploaded

            if uploaded:
                st.text(body="Account types")
                acct_types: Dict[str, str] = {}
                for f in uploaded:
                    acct_types[f.name] = st.selectbox(
                        f.name, list(_ACCOUNT_TYPES.keys()),
                        index=0, key=f"acct_{f.name}",
                    )
                config["acct_types"] = acct_types
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
                and (not config.get("uploaded")
                    or not config.get("client_id", "").strip()))
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
                st.rerun()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _build_csv_data(config: dict) -> Dict[str, List[CSVFileInput]]:
    client_id  = config["client_id"].strip()
    acct_types = config.get("acct_types", {})
    files = [
        CSVFileInput(
            filepath=f.name,
            df=pd.read_csv(io.StringIO(f.getvalue().decode("utf-8-sig"))),
            account_type=_ACCOUNT_TYPES[acct_types.get(f.name, "Auto-detect")],
        )
        for f in config["uploaded"]
    ]
    return {client_id: files}


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
    Launch the pipeline in a daemon thread and track progress via a shared
    ctx dict stored in session state.  The Streamlit script polls ctx on each
    rerun — no session state is written from the background thread.
    """
    ctx: dict = {
        "done":             False,
        "cancelled":        False,
        "cancel_requested": False,
        "results":          None,
        "error":            None,
        "frac":             0.0,
        "label":            "Initialising…",
        "detail":           "",
    }
    st.session_state._pipeline_ctx     = ctx
    st.session_state.running           = True
    st.session_state.results           = None
    st.session_state.override_decision = None
    st.session_state.employee_notes    = ""

    # Read uploaded files on the main thread before hand-off
    csv_data = _build_csv_data(config) if config["source"] == "CSV Upload" else None

    def _run() -> None:
        def on_progress(label: str, detail: str, frac: float) -> None:
            if ctx["cancel_requested"]:
                raise InterruptedError("Cancelled by user")
            ctx["frac"]   = frac
            ctx["label"]  = label
            ctx["detail"] = detail

        def on_sub_progress(done: int, total: int) -> None:
            ctx["detail"] = f"{done:,} / {total:,} transactions"

        try:
            if csv_data is not None:
                coverage_error = _check_csv_date_coverage(csv_data)
                if coverage_error:
                    ctx["error"] = coverage_error
                    return

            orchestrator = Orchestrator(
                api_provider           = config.get("api_provider"),
                n_gpu_layers           = config.get("gpu_layers", 0),
                ner_threshold          = config.get("ner_threshold", 0.85),
                categorizer_batch_size = config.get("batch_size", 32),
            )
            if config["source"] == "CSV Upload" and csv_data is not None:
                ctx["results"] = orchestrator.run_from_csv(
                    csv_data, on_progress=on_progress,
                    on_sub_progress=on_sub_progress)
            else:
                ctx["results"] = orchestrator.run_from_plaid_sandbox(
                    start_date=config["start_date"],
                    end_date=config["end_date"],
                    selected_users=config.get("selected_users"),
                    on_progress=on_progress,
                    on_sub_progress=on_sub_progress,
                )
        except InterruptedError:
            ctx["cancelled"] = True
        except Exception as exc:
            ctx["error"] = str(exc)
        finally:
            ctx["done"] = True

    threading.Thread(target=_run, daemon=True).start()


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
        st.plotly_chart(fig, use_container_width=True,
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
    st.plotly_chart(fig_cf, use_container_width=True, config={"displayModeBar": False})

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
            st.plotly_chart(fig_pie, use_container_width=True,
                            config={"displayModeBar": False})
        with c_tbl:
            st.dataframe(
                cat_df.assign(CAD=cat_df["CAD"].map("${:,.2f}".format))
                        .rename(columns={"CAD": "Amount (CAD)"}),
                use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# Tab: Products
# ---------------------------------------------------------------------------

def _tab_products(result: OrchestratorResult) -> None:
    uw  = result.underwriting or {}
    rec = {p.lower() for p in uw.get("recommended_products", [])}

    st.caption("Green cards were recommended by the underwriting model.")

    cols = st.columns(3, gap="small")
    for i, product in enumerate(PRODUCTS.products):
        is_rec = product.name.lower() in rec
        bg     = "background:#e6f7f4; border:1px solid #00C8A0;" if is_rec else "background:#ffffff; border:1px solid #e4e2e1;"
        income = (f'<p style="margin:8px 0 0 0;font-size:12px;color:#686664;">'
                  f'Min income: ${product.min_annual_income:,}/yr</p>'
                  if product.min_annual_income else "")
        card = f"""
        <div style="{bg} border-radius:10px; padding:16px; margin:8px 0; height:300px; box-sizing:border-box; overflow:hidden;">
            <span style="background:#e4e2e1;color:#686664;padding:5px 10px;border-radius:10px;font-size:12px;">
                {html_lib.escape(product.type)}
            </span>
            <h4 style="margin:10px 0 6px 0;font-size:16px;">{html_lib.escape(product.name)}</h4>
            <p style="margin:0;font-size:13px;color:#32302f;">{html_lib.escape(product.description)}</p>
            {income}
        </div>"""
        with cols[i % 3]:
            st.markdown(card, unsafe_allow_html=True)


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
    st.dataframe(filtered, use_container_width=True, hide_index=True, height=470)


# ---------------------------------------------------------------------------
# Tab: Employee Tools
# ---------------------------------------------------------------------------

def _tab_employee(result: OrchestratorResult) -> None:
    uw = result.underwriting or {}

    st.subheader("Decision Override")
    st.caption(
        "Record a manual override. Stored in this session and included "
        "in the exported JSON report."
    )
    current = uw.get("decision", "—")
    choice  = st.selectbox(
        "Decision",
        [f"Keep model decision ({current})", "approved", "conditional", "rejected"],
    )
    if not choice.startswith("Keep"):
        st.session_state.override_decision = choice
        st.warning(f"Decision overridden to **{choice}**")
    else:
        st.session_state.override_decision = None

    st.divider()

    st.subheader("Internal Notes")
    notes = st.text_area(
        "notes",
        value=st.session_state.employee_notes,
        placeholder="Underwriting notes, override rationale, client context…",
        height=140,
        label_visibility="collapsed",
    )
    st.session_state.employee_notes = notes

    st.divider()

    st.subheader("Export")
    uid    = result.user.user_id if result.user else "result"
    export = {
        "user_id":           uid,
        "underwriting":      uw,
        "override_decision": st.session_state.override_decision,
        "employee_notes":    st.session_state.employee_notes,
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
        st.image(splash_path, width='content',)
    
    st.markdown(
        "To get started, please select a data source and configure the analysis parameters in the sidebar, then click 'Run Analysis'.",
        text_alignment='center'
    )
    st.markdown("Once the analysis is complete, you can explore the results across different tabs.", text_alignment='center')
    
    st.divider()
    st.markdown(
        """
        **Please note:** This demo uses synthetic data and is intended for illustrative purposes only. The underwriting decisions and product recommendations generated by this tool should not be considered final or used in real-world applications without further review by a qualified underwriter.
        """,
        text_alignment='center'
    )