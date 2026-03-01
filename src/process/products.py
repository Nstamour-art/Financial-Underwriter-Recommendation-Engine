"""
Wealthsimple product catalogue consumed by the underwriting prompt.

The Product dataclass lives in src/custom_dataclasses/product.py.
This module defines the master PRODUCTS list and retrieval helpers used
by the prompt builder and any downstream filtering logic.
"""

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from custom_dataclasses import Product, ProductType, ProductCatalog
except ImportError:
    from ..custom_dataclasses import Product, ProductType, ProductCatalog


# ---------------------------------------------------------------------------
# Master catalogue
# ---------------------------------------------------------------------------

PRODUCTS: ProductCatalog = ProductCatalog(
    products=[
        # ------------------------------------------------------------------
        # Savings
        # ------------------------------------------------------------------
        Product(
            name="Wealthsimple Cash",
            type="savings",
            description="High-interest CAD savings account",
            recommendation_notes=(
                "Best for clients who want a safe, liquid place for emergency funds "
                "or short-term savings with no market risk. Ideal as a first product "
                "for any approved applicant, especially those new to Wealthsimple or "
                "those building a financial foundation."
            ),
        ),
        Product(
            name="US Dollar Savings",
            type="savings",
            description="High-interest USD savings account",
            recommendation_notes=(
                "Best for clients who earn or spend in USD regularly — e.g. remote "
                "workers paid in USD, frequent US travellers, or clients with US "
                "investment income. Avoids repeated currency conversion costs."
            ),
        ),

        # ------------------------------------------------------------------
        # Investing — Managed (Robo-advisor)
        # ------------------------------------------------------------------
        Product(
            name="Managed TFSA",
            type="investing",
            description="Tax-Free Savings Account, Wealthsimple-managed portfolio",
            recommendation_notes=(
                "Ideal for clients who want tax-free growth without managing their "
                "own investments. Best for long-term savers, those new to investing, "
                "or anyone who prefers a hands-off approach. Annual contribution room "
                "applies; withdrawals are re-added the following calendar year."
            ),
        ),
        Product(
            name="Managed RRSP",
            type="investing",
            description="Registered Retirement Savings Plan, Wealthsimple-managed portfolio",
            recommendation_notes=(
                "Best for clients with earned income who want to reduce their current "
                "tax bill and save for retirement. Particularly valuable for mid-to-high "
                "income earners who benefit most from the RRSP deduction. The managed "
                "approach suits clients who prefer not to actively select investments."
            ),
        ),
        Product(
            name="Managed FHSA",
            type="investing",
            description="First Home Savings Account, Wealthsimple-managed portfolio",
            recommendation_notes=(
                "Designed for first-time home buyers saving for a down payment. "
                "Contributions are tax-deductible and growth is tax-free. $8,000/yr "
                "contribution limit ($40,000 lifetime). Recommend for clients under 40 "
                "who have not owned a home and plan to purchase within 15 years and "
                "prefer a hands-off investment approach."
            ),
        ),
        Product(
            name="Non-Registered Managed",
            type="investing",
            description="Taxable managed investment account (robo-advisor)",
            recommendation_notes=(
                "Best for clients who have already maximized their registered accounts "
                "(TFSA, RRSP, FHSA) and want to continue growing wealth through a "
                "professionally managed, diversified portfolio. Suited to higher-income "
                "clients with surplus savings and a long investment horizon."
            ),
        ),

        # ------------------------------------------------------------------
        # Investing — Self-Directed
        # ------------------------------------------------------------------
        Product(
            name="Self-Directed TFSA",
            type="investing",
            description="Tax-Free Savings Account, client-directed stocks and ETFs",
            recommendation_notes=(
                "Best for financially confident clients who want to select their own "
                "stocks and ETFs while enjoying tax-free growth. Ideal for experienced "
                "investors seeking lower management fees than robo-advisor portfolios. "
                "Requires comfort with investment decisions and market volatility."
            ),
        ),
        Product(
            name="Self-Directed RRSP",
            type="investing",
            description="RRSP with client-directed stocks and ETF trading",
            recommendation_notes=(
                "Best for experienced self-directed investors who want full control "
                "over their retirement portfolio while benefiting from the RRSP tax "
                "deduction. Suited to clients with earned income and the knowledge to "
                "manage a diversified portfolio independently."
            ),
        ),
        Product(
            name="Self-Directed FHSA",
            type="investing",
            description="First Home Savings Account, client-directed trading",
            recommendation_notes=(
                "Same eligibility as Managed FHSA but for first-time home buyers who "
                "are experienced investors and want full control over how their down "
                "payment savings are invested within the tax shelter."
            ),
        ),
        Product(
            name="Stocks and ETFs",
            type="investing",
            description="Self-directed taxable brokerage account for stocks and ETFs",
            recommendation_notes=(
                "Best for self-directed investors with surplus funds beyond registered "
                "account limits who want to build a taxable investment portfolio. "
                "Suited to financially stable, experienced clients who are comfortable "
                "with investment decisions and capital gains tax implications."
            ),
        ),

        # ------------------------------------------------------------------
        # Crypto
        # ------------------------------------------------------------------
        Product(
            name="Crypto",
            type="crypto",
            description="Cryptocurrency trading account (BTC, ETH, and others)",
            recommendation_notes=(
                "Higher risk — only recommend for clients with consistently positive "
                "monthly net cash flow, zero NSF or overdraft events, and a stable "
                "income. Best for clients who already have core savings and investments "
                "in place and have expressed interest in alternative assets. Do not "
                "recommend to anyone showing any financial stress signals."
            ),
        ),

        # ------------------------------------------------------------------
        # Credit Card
        # ------------------------------------------------------------------
        Product(
            name="Wealthsimple Credit Card",
            type="credit",
            description=(
                "Premium Visa credit card. $20/month fee, waived with either "
                "≥ CAD $4,000/month in direct deposits or "
                "≥ CAD $100,000 in combined Wealthsimple deposits and investments."
            ),
            min_annual_income=60_000,
            recommendation_notes=(
                "Best for clients earning ≥ CAD $60,000/year who would benefit from "
                "a premium rewards card. The $20/month fee is automatically waived for "
                "clients with $4,000+/month in direct deposit income or $100,000+ in "
                "Wealthsimple assets, making it effectively free for higher earners "
                "already using Wealthsimple. Recommend when the applicant clearly "
                "meets the income or asset waiver threshold; note the $20/month fee "
                "if neither waiver condition is obviously met."
            ),
        ),
    ]
)
