"""
Wealthsimple Underwriting Tool — entry point.

Run with:
    streamlit run main.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import List

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — add src/ so all project packages resolve correctly
# ---------------------------------------------------------------------------
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJ_ROOT, "src"))

from ui.app import (  # noqa: E402
    OrchestratorResult,
    _error_result,
    _init_state,
    _render_progress,
    _render_sidebar,
    _start_pipeline,
    _tab_overview,
    _tab_products,
    _tab_review,
    _tab_spending,
    _tab_transactions,
    _welcome,
)

_POLL_INTERVAL = 0.35  # seconds between progress refreshes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()

    _render_sidebar()

    # ----- kick off the pipeline when the sidebar queues a config -----
    pending = st.session_state.pending_config
    if pending is not None and not st.session_state.running:
        st.session_state.pending_config = None
        _start_pipeline(pending)
        st.rerun()

    # ----- pipeline is running — show live progress & poll -----
    if st.session_state.running:
        ctx = st.session_state.get("_pipeline_ctx") or {}

        if ctx.get("done"):
            # Pipeline finished — harvest results
            st.session_state.running = False
            st.session_state._pipeline_ctx = None
            if ctx.get("cancelled"):
                pass  # reset to welcome screen
            elif ctx.get("error"):
                st.session_state.results = [_error_result(ctx["error"])]
            elif ctx.get("results"):
                st.session_state.results = ctx["results"]
            st.rerun()

        # Still running — render progress then auto-refresh
        st.subheader("Running analysis…")
        _render_progress()
        if st.button("Cancel", type="secondary"):
            ctx["cancel_requested"] = True
        time.sleep(_POLL_INTERVAL)
        st.rerun()

    # ----- show results or welcome screen -----
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
            ["Overview", "Spending", "Products", "Transactions", "Decision Review"]
        )
        with t1: _tab_overview(result)
        with t2: _tab_spending(result)
        with t3: _tab_products(result)
        with t4: _tab_transactions(result)
        with t5: _tab_review(result)
    else:
        _welcome()


if __name__ == "__main__":
    main()
