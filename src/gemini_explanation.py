"""Optional Google Gemini summaries grounded in verified model output.

This module never computes a prediction and never creates feature explanations.
It receives only the probability, risk category, threshold, and already verified
local SHAP contributions from ``src.explainability``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class GeminiResult:
    """Result of an optional Gemini request without exposing SDK errors."""

    text: str | None
    status: str
    used_gemini: bool


def _contribution_lines(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items[:5]:
        feature = str(item.get("feature", "unknown feature"))
        try:
            value = float(item.get("shap_value", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        lines.append(f"- {feature}: SHAP contribution {value:.6f}")
    return "\n".join(lines) or "- None listed"


def build_investigator_prompt(
    probability: float,
    risk_category: str,
    threshold: float,
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
) -> str:
    """Create a prompt from prediction results and verified SHAP output only."""

    return f"""Write a concise investigator-facing summary for an insurance claim risk assessment.

Use only the structured model output below. Do not invent claim facts, causes,
evidence, missing fields, or investigation findings. SHAP values describe model
associations, not causal relationships. State clearly that the result is a risk
assessment and not proof of fraud.

Estimated fraud probability: {float(probability):.6f}
Risk category: {risk_category}
Frozen decision threshold: {float(threshold):.6f}

Verified factors increasing the model score:
{_contribution_lines(positive)}

Verified factors decreasing the model score:
{_contribution_lines(negative)}

Return 2-4 plain-language sentences. Mention the strongest listed factors,
recommend human review proportional to the risk category, and avoid legal or
definitive fraud conclusions."""


def _status_for_exception(error: Exception) -> str:
    message = str(error).lower()
    if any(token in message for token in ("quota", "resource exhausted", "429", "rate limit", "too many requests")):
        return "quota_exceeded"
    if any(token in message for token in ("timeout", "timed out", "connection", "network", "dns")):
        return "network_error"
    return "api_error"


def generate_investigator_summary(
    *,
    api_key: str | None,
    probability: float,
    risk_category: str,
    threshold: float,
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    model: str = DEFAULT_GEMINI_MODEL,
) -> GeminiResult:
    """Generate a grounded summary, returning a safe status on every failure."""

    if not isinstance(api_key, str) or not api_key.strip():
        return GeminiResult(None, "missing_key", False)

    prompt = build_investigator_prompt(probability, risk_category, threshold, positive, negative)
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a careful insurance investigation assistant. "
                    "Summarize only verified model outputs supplied in the prompt."
                ),
                temperature=0.2,
                max_output_tokens=300,
            ),
        )
        text = (response.text or "").strip()
        if not text:
            return GeminiResult(None, "empty_response", False)
        return GeminiResult(text, "ok", True)
    except ImportError:
        return GeminiResult(None, "sdk_missing", False)
    except Exception as error:  # SDK/network/quota errors must not break inference.
        return GeminiResult(None, _status_for_exception(error), False)
