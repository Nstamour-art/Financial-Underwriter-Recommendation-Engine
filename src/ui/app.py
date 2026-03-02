"""
Wealthsimple Underwriting Tool — Streamlit front end.

Run with:
    streamlit run main.py

Fonts and theme are configured in .streamlit/config.toml.
Static assets (fonts, images) are served from static/ at the project root.
"""

from __future__ import annotations

import io
import json
import os
import sys
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
from orchestrator import Orchestrator, OrchestratorResult
from process import PRODUCTS


# ---------------------------------------------------------------------------
# Colors — only needed for Plotly charts (theme handles everything else)
# ---------------------------------------------------------------------------
WS_BG         = "#fcfcfc"
WS_TEXT       = "#32302f"
WS_TEXT_LIGHT = "#686664"
WS_BORDER     = "#e4e2e1"
WS_SUCCESS    = "#486635"
WS_ERROR      = "#a43d12"
WS_GREEN      = "#00C8A0"
WS_AMBER      = "#F59E0B"


# ---------------------------------------------------------------------------
# Account type options
# ---------------------------------------------------------------------------
_ACCOUNT_TYPES: Dict[str, Optional[tuple]] = {
    "Chequing":           ("depository", "checking"),
    "Savings":            ("depository", "savings"),
    "Credit Card":        ("credit",     "credit card"),
    "Student Loan":       ("loan",       "student"),
    "Mortgage":           ("loan",       "mortgage"),
    "Investment (TFSA)":  ("investment", "tfsa"),
    "Investment (RRSP)":  ("investment", "rrsp"),
    "Investment (Other)": ("investment", "brokerage"),
    "Auto-detect":        None,
}


# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WS Underwriting",
    page_icon="💚",
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
        "progress_label":    "",
        "progress_detail":   "",
        "progress_frac":     0.0,
        "progress_log":      [],
        "selected_idx":      0,
        "override_decision": None,
        "employee_notes":    "",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> Optional[dict]:
    logo_path = os.path.join(_PROJ_ROOT, "static", "imgs",
                             "ws-wordmark-refresh.3499def3.svg")
    if os.path.exists(logo_path):
        st.logo(logo_path)

    with st.sidebar:
        st.title("Underwriting")
        st.caption("Internal advisory tool")
        st.divider()

        st.markdown("**Data Source**")
        source = st.radio("source", ["CSV Upload", "Plaid Sandbox"],
                          label_visibility="collapsed")
        config: dict = {"source": source}

        if source == "CSV Upload":
            st.markdown("**Client ID**")
            config["client_id"] = st.text_input(
                "cid", value="client_001", label_visibility="collapsed")

            st.markdown("**Transaction files**")
            uploaded = st.file_uploader(
                "csv", type=["csv"], accept_multiple_files=True,
                label_visibility="collapsed",
            )
            config["uploaded"] = uploaded

            if uploaded:
                st.markdown("**Account types**")
                acct_types: Dict[str, str] = {}
                for f in uploaded:
                    acct_types[f.name] = st.selectbox(
                        f.name, list(_ACCOUNT_TYPES.keys()),
                        index=0, key=f"acct_{f.name}",
                    )
                config["acct_types"] = acct_types
        else:
            st.markdown("**Date range**")
            c1, c2 = st.columns(2)
            with c1:
                config["start_date"] = st.date_input(
                    "From", value=date.today() - timedelta(days=365))
            with c2:
                config["end_date"] = st.date_input("To", value=date.today())

        st.divider()

        st.markdown("**LLM Provider**")
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
            source == "CSV Upload"
            and (not config.get("uploaded")
                 or not config.get("client_id", "").strip())
        )
        if st.button("Run Analysis", type="primary",
                     disabled=run_disabled or st.session_state.running):
            return config

        if st.session_state.results is not None:
            if st.button("Clear & start over"):
                st.session_state.results           = None
                st.session_state.progress_log      = []
                st.session_state.override_decision = None
                st.session_state.employee_notes    = ""
                st.rerun()

    return None


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


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(config: dict) -> None:
    st.session_state.running           = True
    st.session_state.results           = None
    st.session_state.progress_log      = []
    st.session_state.progress_frac     = 0.0
    st.session_state.override_decision = None
    st.session_state.employee_notes    = ""

    orchestrator = Orchestrator(
        api_provider           = config.get("api_provider"),
        n_gpu_layers           = config.get("gpu_layers", 0),
        ner_threshold          = config.get("ner_threshold", 0.85),
        categorizer_batch_size = config.get("batch_size", 32),
    )
    log: list = []

    def on_progress(label: str, detail: str, frac: float) -> None:
        st.session_state.progress_label  = label
        st.session_state.progress_detail = detail
        st.session_state.progress_frac   = frac
        log.append({"label": label, "detail": detail, "frac": frac})
        st.session_state.progress_log = list(log)

    try:
        if config["source"] == "CSV Upload":
            results = orchestrator.run_from_csv(
                _build_csv_data(config), on_progress=on_progress)
        else:
            results = orchestrator.run_from_plaid_sandbox(
                start_date=config["start_date"],
                end_date=config["end_date"],
                on_progress=on_progress,
            )
        st.session_state.results = results
    except Exception as exc:
        st.session_state.results = [_error_result(str(exc))]
    finally:
        st.session_state.running = False


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
            st.write(summary)
        if rej_reason:
            st.error(f"**Rejection reason:** {rej_reason}")

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

    _PLOT_FONT = dict(family="The Future, Helvetica Neue, sans-serif",
                      color=WS_TEXT)

    st.subheader("Monthly Cash Flow")
    cf_df   = pd.DataFrame(cf).T.reset_index().rename(columns={"index": "Month"})
    cf_long = cf_df.melt(id_vars="Month", value_vars=["income", "expenses", "net"],
                         var_name="Type", value_name="CAD")
    fig_cf = px.line(
        cf_long, x="Month", y="CAD", color="Type",
        color_discrete_map={"income": WS_SUCCESS, "expenses": WS_ERROR, "net": "#6366f1"},
        markers=True, template="plotly_white",
    )
    fig_cf.update_layout(
        font=_PLOT_FONT,
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
        chart_sp = (
            alt.Chart(sp_long)
            .mark_bar()
            .encode(
                x=alt.X("Month:N", sort=None),
                y=alt.Y("CAD:Q", stack="zero", title="CAD"),
                color=alt.Color("Category:N", legend=alt.Legend(orient="bottom")),
                tooltip=["Month", "Category", alt.Tooltip("CAD:Q", format="$,.0f")],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_sp, width="stretch")

    if cat:
        st.subheader("Spending Breakdown")
        cat_df = (pd.DataFrame([{"Category": k, "CAD": v} for k, v in cat.items()])
                  .sort_values("CAD", ascending=False))
        c_pie, c_tbl = st.columns([2, 1])
        with c_pie:
            fig_pie = px.pie(
                cat_df, names="Category", values="CAD", hole=0.44,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_pie.update_traces(textinfo="percent+label", textfont_size=10)
            fig_pie.update_layout(
                font=_PLOT_FONT, showlegend=False,
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

    st.caption("Highlighted cards were recommended by the underwriting model.")

    cols = st.columns(3)
    for i, product in enumerate(PRODUCTS.products):
        is_rec = product.name.lower() in rec
        with cols[i % 3]:
            with st.container(border=True):
                if is_rec:
                    st.badge("Recommended", color="green")
                st.badge(product.type, color="gray")
                st.subheader(product.name)
                st.write(product.description)
                if product.min_annual_income:
                    st.caption(f"Min income: ${product.min_annual_income:,}/yr")


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
    st.title("Underwriting Console")
    st.caption("Credit scoring & product recommendations — internal use only")
    st.divider()

    for col, icon, title, body in zip(
        st.columns(3),
        ["📁", "🏦", "🤖"],
        ["CSV Upload", "Plaid Sandbox", "LLM Underwriting"],
        [
            "Upload one or more transaction CSV files and assign account "
            "types to run the full NLP and underwriting pipeline.",
            "Pull live-format data directly from the Plaid sandbox "
            "environment without uploading any files.",
            "Claude (Anthropic) or GPT-4o mini (OpenAI) generates a "
            "credit score, approval decision, and product recommendations.",
        ],
    ):
        with col:
            with st.container(border=True):
                st.write(icon)
                st.subheader(title)
                st.write(body)

    st.caption("Configure a data source in the sidebar, then click **Run Analysis**.")
