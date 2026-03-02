"""
Anthropic Claude API client for the underwriting pipeline.

Reads ANTHROPIC_API_KEY from the environment (set by the caller or .env).
Returns the LLM response as a parsed dict matching the underwriting schema.
"""

import json
import re

import anthropic
from anthropic.types import TextBlock


# Model used for underwriting inference.
# claude-sonnet-4-6 balances accuracy and cost; swap for claude-haiku-4-5
# if lower latency / cost is preferred at the expense of reasoning depth.
_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 512


def _parse_json(text: str) -> dict:
    """Strip optional markdown fences and parse JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def call_anthropic(system_prompt: str, user_message: str) -> dict:
    """
    Send the underwriting prompt to Claude and return the parsed JSON response.

    Parameters
    ----------
    system_prompt : str
        The underwriter role definition, output schema, and product list.
    user_message : str
        The compact financial summary for a single client.

    Returns
    -------
    dict with keys: score, decision, summary, rejection_reason,
                    recommended_products
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    text_block = next(b for b in response.content if isinstance(b, TextBlock))
    return _parse_json(text_block.text)
