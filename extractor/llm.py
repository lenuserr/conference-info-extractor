"""
LLM-based extraction via Ollama (local models, no paid APIs).

Uses chain-of-extraction: split into two passes to reduce hallucinations
in small models (7-10B parameters).

Pass 1 — "basic": name, acronym, dates, venue, deadlines, publisher
Pass 2 — "details": topics, keynote speakers
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Ollama settings
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "mistral:latest"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BASIC_PROMPT = """\
You are a precise data extraction assistant. Extract conference metadata from the text below.
Return ONLY a valid JSON object — no markdown, no explanation, no extra text.

If a field cannot be determined from the text, use null for scalars and [] for arrays.
Dates must be in YYYY-MM-DD format. Country must be the full English name (e.g. "Spain", not "ES").
edition_number is the numeric edition (e.g. 38 for "38th Annual Conference"), or null.

Required JSON structure:
{{
  "full_name": "string",
  "acronym": "string",
  "edition_number": int or null,
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
  "city": "string or null",
  "country": "string or null",
  "submission_deadline": "YYYY-MM-DD or null",
  "notification_date": "YYYY-MM-DD or null",
  "camera_ready_date": "YYYY-MM-DD or null",
  "publisher": "string or null",
  "series": "string or null"
}}

--- TEXT START ---
{text}
--- TEXT END ---

JSON:"""

_DETAILS_PROMPT = """\
You are a precise data extraction assistant. From the conference text below,
extract ONLY the following two things. Return ONLY valid JSON — no markdown, no explanation.

1. "topics" — a list of research topics/themes/tracks of this conference.
   Return [] if none are found.
2. "keynote_speakers" — a list of keynote/invited speakers. Each entry:
   {{"name": "string", "affiliation": "string or null", "country": "string or null"}}
   Return [] if none are found.

Required JSON structure:
{{
  "topics": ["string", ...],
  "keynote_speakers": [
    {{"name": "...", "affiliation": "...", "country": "..."}}
  ]
}}

--- TEXT START ---
{text}
--- TEXT END ---

JSON:"""


# ---------------------------------------------------------------------------
# Ollama interaction
# ---------------------------------------------------------------------------

def _call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.1,
) -> Optional[str]:
    """Send a prompt to Ollama and return the raw text response."""
    try:
        import ollama as _ollama_lib
        client = _ollama_lib.Client(host=base_url)
        response = client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": 2048},
        )
        return response.get("response", "")
    except ImportError:
        logger.warning("ollama library not installed, falling back to HTTP")
    except Exception as exc:
        logger.warning("ollama library call failed: %s — falling back to HTTP", exc)

    # Fallback: direct HTTP call to Ollama API
    import requests
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 2048},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as exc:
        logger.error("Ollama HTTP call failed: %s", exc)
        return None


def _parse_json_from_response(raw: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response, tolerating markdown fences etc."""
    if not raw:
        return None

    # Try direct parse first
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first { ... } block
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_basic(
    text: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> Optional[Dict[str, Any]]:
    """
    Pass 1: extract basic metadata (name, dates, venue, deadlines, publisher).
    """
    prompt = _BASIC_PROMPT.format(text=text[:8000])
    raw = _call_ollama(prompt, model=model, base_url=base_url)
    result = _parse_json_from_response(raw)
    if result is None:
        logger.warning("Pass 1 (basic) failed to produce valid JSON")
    return result


def extract_details(
    text: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> Optional[Dict[str, Any]]:
    """
    Pass 2: extract topics and keynote speakers.
    """
    prompt = _DETAILS_PROMPT.format(text=text[:8000])
    raw = _call_ollama(prompt, model=model, base_url=base_url)
    result = _parse_json_from_response(raw)
    if result is None:
        logger.warning("Pass 2 (details) failed to produce valid JSON")
    return result
