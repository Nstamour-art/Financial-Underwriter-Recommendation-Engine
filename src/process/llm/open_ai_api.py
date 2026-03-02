"""
OpenAI API client for the underwriting pipeline.

Reads OPENAI_API_KEY from the environment (set by the caller or .env).
Returns the LLM response as a parsed dict matching the underwriting schema.
"""

import json

import openai


# gpt-4o-mini is cost-effective and fast; swap for gpt-4o if deeper
# financial reasoning is required.
_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 512


def call_openai(system_prompt: str, user_message: str) -> dict:
    """
    Send the underwriting prompt to OpenAI and return the parsed JSON response.

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
    client = openai.OpenAI()  # reads OPENAI_API_KEY from env

    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    )

    return json.loads(response.choices[0].message.content or "{}")
