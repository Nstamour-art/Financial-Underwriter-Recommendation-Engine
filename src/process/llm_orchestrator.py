"""
Underwriting orchestrator.

Routes a User through prompt building and LLM inference, returning a
structured underwriting result dict.  LLM provider is selected automatically
based on available environment keys:
    ANTHROPIC_API_KEY → Claude (claude-sonnet-4-6)
    OPENAI_API_KEY    → OpenAI (gpt-4o-mini)

If both keys are present, Anthropic is preferred.
"""

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from custom_dataclasses import User
from process.products import PRODUCTS, Product
from process.prompts.prompt import build_prompt
from process.llm.anthropic_api import call_anthropic
from process.llm.open_ai_api import call_openai


# ---------------------------------------------------------------------------
# Result schema (for reference; actual return value is a plain dict)
# ---------------------------------------------------------------------------
#
# {
#   "score":                int,          # 300–900
#   "decision":             str,          # "approved" | "conditional" | "rejected"
#   "summary":              str,          # 2–3 sentence plain-English explanation
#   "rejection_reason":     str | None,   # populated when decision == "rejected"
#   "recommended_products": list[str],    # empty when rejected
#   "provider":             str,          # "anthropic" | "openai"  (added by orchestrator)
# }


class UnderwritingOrchestrator:
    """
    Runs the full underwriting pipeline for a single User and returns a
    structured decision dict.

    Parameters
    ----------
    products : list of Product, optional
        Override the default PRODUCTS catalogue (useful for testing or
        offering a subset of products to specific client segments).
    """

    def __init__(self, products: Optional[List[Product]] = None):
        self._products = products or PRODUCTS

    def run(self, user: User, api_provider: Optional[str] = None) -> dict:
        """
        Execute the underwriting pipeline.

        1. Build the (system_prompt, user_message) pair from the User object.
        2. Route to the available LLM provider.
        3. Return the parsed JSON result, augmented with a 'provider' field.

        Parameters
        ----------
        user : User
            Fully populated User with cleaned and categorized transactions.

        Returns
        -------
        dict matching the result schema documented above.

        Raises
        ------
        EnvironmentError
            When neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set.
        """
        system_prompt, user_message = build_prompt(user, self._products)

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key    = os.getenv("OPENAI_API_KEY")
        
        if api_provider == "anthropic" and not anthropic_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set, cannot use Anthropic API.")
        elif api_provider == "openai" and not openai_key:
            raise EnvironmentError("OPENAI_API_KEY not set, cannot use OpenAI API.")
        elif api_provider == "anthropic" or (api_provider is None and anthropic_key):
            result   = call_anthropic(system_prompt, user_message)
            provider = "anthropic"
        elif api_provider == "openai" or (api_provider is None and openai_key):
            result   = call_openai(system_prompt, user_message)
            provider = "openai"
        else:
            raise EnvironmentError(
                "No LLM API key configured. "
                "Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment."
            )

        result["provider"] = provider
        return result
