"""
Core agent logic: processes conversation history and returns structured response.
Uses Google Gemini (new google-genai SDK) as the LLM.
"""

import json
import os
import re
import logging
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

from agent.prompts import build_system_prompt, build_retrieval_context
from retrieval.retriever import get_retriever

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-1.5-flash"


@lru_cache(maxsize=1)
def _get_gemini_client():
    """Lazy-init Gemini client — not called at import time."""
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


def _extract_query_from_history(messages: list[dict]) -> str:
    """Build a search query from the accumulated user messages."""
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_texts[-4:])  # Last 4 user turns max


def _detect_comparison_request(messages: list[dict]) -> tuple[bool, list[str]]:
    """Detect if the user is asking to compare specific assessments."""
    if not messages:
        return False, []
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    compare_patterns = [
        r"(?:difference|compare|versus|vs\.?|between)\s+(.+?)\s+(?:and|vs\.?)\s+(.+?)(?:\?|$)",
        r"(?:what(?:'s| is) the difference between)\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
    ]
    for pattern in compare_patterns:
        m = re.search(pattern, last_user, re.IGNORECASE)
        if m:
            return True, [m.group(1).strip(), m.group(2).strip()]
    return False, []


def _build_contents(
    messages: list[dict],
    retrieval_context: str,
) -> list[dict]:
    """
    Convert API messages to google-genai Content format.
    Injects retrieval context into the last user message.
    """
    contents = []
    for i, msg in enumerate(messages):
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"]

        if role == "user" and i == len(messages) - 1 and retrieval_context:
            content = f"{content}\n\n[CATALOG CONTEXT]\n{retrieval_context}"

        contents.append({"role": role, "parts": [{"text": content}]})
    return contents


def _validate_and_clean_response(
    raw: dict,
    retriever,
) -> dict:
    """
    Validate the LLM response against strict schema rules.
    - Ensure recommendations only contain catalog URLs
    - Clamp recommendations to 1-10
    - Ensure required fields exist
    """
    reply = str(raw.get("reply", "")).strip()
    raw_recs = raw.get("recommendations", [])
    end_of_conv = bool(raw.get("end_of_conversation", False))

    if not isinstance(raw_recs, list):
        raw_recs = []

    clean_recs = []
    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        url = rec.get("url", "")
        name = rec.get("name", "")
        test_type = rec.get("test_type", "")

        if not url or not retriever.is_valid_url(url):
            # Try to recover by looking up by name
            item = retriever.get_by_name(name)
            if item:
                url = item["url"]
                if not test_type and item.get("test_types"):
                    test_type = item["test_types"][0]
            else:
                logger.warning(f"Dropping recommendation with invalid URL: {url!r} name={name!r}")
                continue

        clean_recs.append({
            "name": name or "",
            "url": url,
            "test_type": test_type or "",
        })

    clean_recs = clean_recs[:10]

    return {
        "reply": reply or "I'm here to help you find the right SHL assessment.",
        "recommendations": clean_recs,
        "end_of_conversation": end_of_conv,
    }


def process_chat(messages: list[dict]) -> dict[str, Any]:
    """
    Main entry point. Takes full conversation history, returns structured response.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str}

    Returns:
        {"reply": str, "recommendations": list, "end_of_conversation": bool}
    """
    from google.genai import types as genai_types

    retriever = get_retriever()

    query = _extract_query_from_history(messages)

    is_comparison, compare_names = _detect_comparison_request(messages)
    candidates = []

    if is_comparison and compare_names:
        for name in compare_names:
            item = retriever.get_by_name(name)
            if item:
                candidates.append(item)
            else:
                results = retriever.search(name, top_k=3)
                candidates.extend(results[:1])
    else:
        candidates = retriever.search(query, top_k=15)

    retrieval_context = build_retrieval_context(candidates)
    system_prompt = build_system_prompt(retriever.get_all())
    contents = _build_contents(messages, retrieval_context)

    client = _get_gemini_client()

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text.strip()

        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if m:
                raw = json.loads(m.group(1))
            else:
                logger.error(f"Failed to parse LLM response as JSON: {raw_text[:300]}")
                raw = {"reply": raw_text, "recommendations": [], "end_of_conversation": False}

    except Exception as e:
        logger.exception(f"LLM call failed: {e}")
        return {
            "reply": "I'm having trouble processing your request. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    return _validate_and_clean_response(raw, retriever)
