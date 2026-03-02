"""
Wealthsimple Underwriting Tool — entry point.

Run with:
    streamlit run main.py
"""

from __future__ import annotations

import os
import sys
from typing import List

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — add src/ so all project packages resolve correctly
# ---------------------------------------------------------------------------
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJ_ROOT, "src"))

from ui.app import (  # noqa: E402
    OrchestratorResult,
    _init_state,
    _render_sidebar,
    _run_pipeline,
    _tab_employee,
    _tab_overview,
    _tab_products,
    _tab_spending,
    _tab_transactions,
    _welcome,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()

    run_config = _render_sidebar()

    if run_config is not None:
        with st.spinner("Running underwriting pipeline…"):
            _run_pipeline(run_config)
        st.rerun()

    if st.session_state.results is not None:
        results: List[OrchestratorResult] = st.session_state.results

        if len(results) > 1:
            user_ids = [
                r.user.user_id if r.user else f"Result {i}"
                for i, r in enumerate(results)
            ]
            st.session_state.selected_idx = st.selectbox(
                "Client", range(len(user_ids)),
                format_func=lambda i: user_ids[i],
            )

        result = results[st.session_state.selected_idx]

        t1, t2, t3, t4, t5 = st.tabs(
            ["Overview", "Spending", "Products", "Transactions", "Employee Tools"]
        )
        with t1: _tab_overview(result)
        with t2: _tab_spending(result)
        with t3: _tab_products(result)
        with t4: _tab_transactions(result)
        with t5: _tab_employee(result)

    elif st.session_state.running:
        st.subheader("Running analysis…")
        st.progress(st.session_state.progress_frac)
        st.write(f"**{st.session_state.progress_label}** — "
                 f"{st.session_state.progress_detail}")
        for entry in st.session_state.progress_log:
            st.write(f"✓ **{entry['label']}** — {entry['detail']} "
                     f"`{entry['frac']*100:.0f}%`")
    else:
        _welcome()


if __name__ == "__main__":
    main()
