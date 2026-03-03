"""
Centralised styles for the Wealthsimple Underwriting Tool.

All colours, CSS, and chart-level style constants live here.
Import what you need in app.py — nothing else should define colours.

  Colours  →  WS_* constants + CATEGORY_COLORS
  Charts   →  PLOT_FONT, CF_COLORS, CAT_DOMAIN, CAT_RANGE
  CSS      →  SIDEBAR_CSS
"""

# ---------------------------------------------------------------------------
# Brand colour tokens
# ---------------------------------------------------------------------------
WS_BG         = "#fcfcfc"    # page background
WS_TEXT       = "#32302f"    # primary text
WS_TEXT_LIGHT = "#686664"    # secondary / muted text
WS_BORDER     = "#e4e2e1"    # hairline borders
WS_GREEN      = "#00C8A0"    # positive / income (WS primary)
WS_ORANGE     = "#ff8902"    # WS primary (CTAs, sidebar accent)
WS_AMBER      = "#F59E0B"    # caution / mid-range scores
WS_SUCCESS    = "#486635"    # deep green for "approved" badges
WS_ERROR      = "#a43d12"    # deep red for "declined" badges

# ---------------------------------------------------------------------------
# Category → hex colour
# ---------------------------------------------------------------------------
# Edit values here to change a category's colour across ALL charts.
#
# Source of truth for category names:
#   src/ingest/stage_3/categorizer.py  → TAXONOMY keys
#   src/ingest/stage_2/plaid_converter.py → _INV_CATEGORY values
# ---------------------------------------------------------------------------
CATEGORY_COLORS: dict[str, str] = {
    # ── Core spending ───────────────────────────────────────────────────────
    "Housing":                   "#9B7EC8",   # muted violet
    "Food and Dining":           "#C4956A",   # warm tan
    "Transportation":            "#5C8FAB",   # steel blue
    "Shopping and Retail":       "#C48A8A",   # dusty rose
    "Entertainment and Leisure": "#B8A369",   # warm gold
    "Healthcare":                "#7DAA92",   # sage
    "Telecommunications":        "#8B93C4",   # slate indigo
    "Education":                 "#6AAA9B",   # muted teal
    "Financial Obligations":     "#A8766B",   # terracotta

    # ── Money movement ──────────────────────────────────────────────────────
    "Transfers and Payments":    "#9EADBA",   # cool grey-blue
    "Investments":               "#4E9A8C",   # teal-green
    "Transfers":                 "#9EADBA",   # Plaid alias → same as Transfers and Payments
    "Fees & Charges":            "#C48A8A",   # Plaid alias → same as Shopping and Retail

    # ── Income / positive ───────────────────────────────────────────────────
    "Income":                    "#00C8A0",   # WS green — reserved for positives

    # ── Catch-all ───────────────────────────────────────────────────────────
    "Other":                     "#BDBAB7",   # neutral warm grey
}

# Case-insensitive lookup — handles Plaid's UPPERCASE category names
_CATEGORY_COLORS_LOWER: dict[str, str] = {k.lower(): v for k, v in CATEGORY_COLORS.items()}

_FALLBACK_COLOR = "#BDBAB7"  # neutral warm grey for any unrecognised category


def get_color(category: str) -> str:
    """Return the brand colour for a category name, case-insensitively."""
    return CATEGORY_COLORS.get(category) or _CATEGORY_COLORS_LOWER.get(category.lower(), _FALLBACK_COLOR)


def colors_for(categories: list[str]) -> tuple[list[str], list[str]]:
    """Return (domain, range) lists for an Altair Scale, built from the actual categories in the data."""
    domain = list(dict.fromkeys(categories))   # deduplicated, order-preserving
    rang   = [get_color(c) for c in domain]
    return domain, rang

# ---------------------------------------------------------------------------
# Chart-level style constants
# ---------------------------------------------------------------------------

# Plotly font dict — pass as font=PLOT_FONT in update_layout()
PLOT_FONT = dict(family="The Future, Helvetica Neue, sans-serif", color=WS_TEXT)

# Cash flow line chart colours — income / expenses / net
CF_COLORS: dict[str, str] = {
    "income":   WS_GREEN,
    "expenses": "#D4756B",   # muted coral-red
    "net":      "#8B93C4",   # slate indigo
}

# ---------------------------------------------------------------------------
# CSS snippets — inject with st.markdown(X, unsafe_allow_html=True)
# ---------------------------------------------------------------------------

# Inverts the Wealthsimple wordmark so it reads white on the dark sidebar
SIDEBAR_CSS = """
<style>
[data-testid="stSidebarHeader"] img,
[data-testid="stLogo"] img {
    filter: brightness(0) invert(1);
}

[data-testid="stBadge"] {
    padding: 3px 10px;
}
</style>
"""
