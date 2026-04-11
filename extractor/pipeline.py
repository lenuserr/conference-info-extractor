"""
Main extraction pipeline: scrape → LLM extract (chain) → validate → retry → output.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .scraper import SiteContent, fetch_conference_site
from .llm import extract_basic, extract_details, DEFAULT_MODEL, DEFAULT_BACKEND, get_default_url
from .validator import full_validate

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def _build_output(
    url: str,
    basic: Optional[Dict[str, Any]],
    details: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge basic + details into the target JSON structure."""
    b = basic or {}
    d = details or {}

    return {
        "conference": {
            "full_name": b.get("full_name") or "",
            "acronym": b.get("acronym") or "",
            "url": url,
            "edition_number": b.get("edition_number"),
        },
        "dates": {
            "start_date": b.get("start_date"),
            "end_date": b.get("end_date"),
        },
        "venue": {
            "city": b.get("city"),
            "country": b.get("country"),
        },
        "deadlines": {
            "submission": b.get("submission_deadline"),
            "notification": b.get("notification_date"),
            "camera_ready": b.get("camera_ready_date"),
        },
        "topics": d.get("topics", []),
        "keynote_speakers": d.get("keynote_speakers", []),
        "program_committee": d.get("program_committee", []),
        "publication": {
            "publisher": b.get("publisher"),
            "series": b.get("series"),
        },
    }


def _count_null_fields(data: Dict[str, Any]) -> Tuple[int, int]:
    """Count (null_fields, total_scalar_fields) for retry decision."""
    total = 0
    nulls = 0

    for section in ["conference", "dates", "venue", "deadlines", "publication"]:
        sub = data.get(section, {})
        for k, v in sub.items():
            total += 1
            if v is None or v == "":
                nulls += 1

    return nulls, total


def _empty_result(url: str) -> Dict[str, Any]:
    """Return a fully-null result."""
    return {
        "conference": {"full_name": "", "acronym": "", "url": url, "edition_number": None},
        "dates": {"start_date": None, "end_date": None},
        "venue": {"city": None, "country": None},
        "deadlines": {"submission": None, "notification": None, "camera_ready": None},
        "topics": [],
        "keynote_speakers": [],
        "program_committee": [],
        "publication": {"publisher": None, "series": None},
    }


def extract_conference(
    url: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    base_url: Optional[str] = None,
    vllm_extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: scrape → extract → validate → retry → return JSON.

    Args:
        url:             Conference website URL.
        model:           Model name (e.g. "mistral:latest" for Ollama, "Qwen/Qwen2.5-7B" for vLLM).
        backend:         "ollama" or "vllm".
        base_url:        Server URL. If None, uses default for the chosen backend.
        vllm_extra_args: Extra CLI args forwarded to ``vllm serve`` when the
                         vLLM backend auto-starts a server (ignored for ollama).

    Returns a dict with keys:
      - "data": the extracted conference JSON
      - "confidence": per-field confidence map
      - "warnings": list of validation warnings
      - "meta": extraction metadata (model, backend, attempts, pages_fetched)
    """
    if base_url is None:
        base_url = get_default_url(backend)

    # --- Step 1: Scrape ---
    logger.info("Fetching conference site: %s", url)
    site: SiteContent = fetch_conference_site(url)

    if not site.pages:
        logger.error("No pages fetched for %s", url)
        return {
            "data": _empty_result(url),
            "confidence": {},
            "warnings": ["Could not fetch any pages"],
            "meta": {"model": model, "backend": backend, "attempts": 0, "pages_fetched": 0},
        }

    full_text = site.full_text
    source_text = site.raw_html_main  # for validation

    best_result = None
    best_warnings: List[str] = []
    best_confidence: Dict[str, str] = {}
    attempts = 0

    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        logger.info("Extraction attempt %d/%d with backend=%s model=%s", attempt, MAX_RETRIES, backend, model)

        # --- Step 2: LLM extraction (chain-of-extraction) ---
        basic = extract_basic(
            full_text, model=model, backend=backend, base_url=base_url,
            vllm_extra_args=vllm_extra_args,
        )
        details = extract_details(
            full_text, model=model, backend=backend, base_url=base_url,
            vllm_extra_args=vllm_extra_args,
        )

        if basic is None and details is None:
            logger.warning("Both extraction passes returned None on attempt %d", attempt)
            continue

        merged = _build_output(url, basic, details)

        # --- Step 3: Validate ---
        validated, confidence, warnings = full_validate(merged, source_text)

        # --- Step 4: Retry decision ---
        nulls, total = _count_null_fields(validated)
        null_pct = nulls / total if total > 0 else 1.0

        logger.info(
            "Attempt %d: %d/%d fields null (%.0f%%), %d warnings",
            attempt, nulls, total, null_pct * 100, len(warnings),
        )

        # Keep best result (fewest nulls)
        if best_result is None or nulls < _count_null_fields(best_result)[0]:
            best_result = validated
            best_warnings = warnings
            best_confidence = confidence

        if null_pct <= 0.35:
            logger.info("Sufficient data extracted, stopping retries.")
            break
        else:
            logger.info("Too many nulls (%.0f%%), retrying...", null_pct * 100)

    if best_result is None:
        best_result = _empty_result(url)

    return {
        "data": best_result,
        "confidence": best_confidence,
        "warnings": best_warnings,
        "meta": {
            "model": model,
            "backend": backend,
            "attempts": attempts,
            "pages_fetched": len(site.pages),
        },
    }
