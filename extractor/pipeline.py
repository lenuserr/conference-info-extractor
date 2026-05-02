"""
Main extraction pipeline: scrape → category-aware selection →
LLM extract (4 categories × 3 fallback levels + brute-force) →
validate → merge → output.

Two algorithms run for each category:

  Algorithm 1 (targeted, 3 levels):
    L1  Navigation — select pages by link_text / URL path keywords
    L2  Content   — search keywords inside page text (new pages only)
    L3  Remaining — everything not yet sent for this category

  Algorithm 2 (brute-force):
    Send ALL pages to the LLM in one shot.

Final merge: algorithm 1 results take priority; algorithm 2 fills gaps.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from .scraper import SiteContent, fetch_conference_site
from .content_selection import Category, PageSelector, build_context
from .llm import (
    extract_other,
    extract_topics,
    extract_speakers,
    extract_committee,
    DEFAULT_MODEL,
    DEFAULT_BACKEND,
    get_default_url,
)
from .validator import validate_category

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category → extraction function mapping
# ---------------------------------------------------------------------------

_EXTRACT_FN: Dict[Category, Callable] = {
    Category.OTHER: extract_other,
    Category.TOPICS: extract_topics,
    Category.SPEAKERS: extract_speakers,
    Category.COMMITTEE: extract_committee,
}


# ---------------------------------------------------------------------------
# Helpers: detect whether a category result has useful data
# ---------------------------------------------------------------------------

def _has_other_data(result: Dict[str, Any]) -> bool:
    """Check if OTHER result has enough data to skip further levels.

    We consider it sufficient if we have at least the conference name
    AND at least one of: dates, venue, or deadlines.
    """
    if not result:
        return False
    has_name = bool(result.get("full_name"))
    has_dates = bool(result.get("start_date"))
    has_venue = bool(result.get("city"))
    has_deadline = bool(result.get("submission_deadline"))
    return has_name and (has_dates or has_venue or has_deadline)


def _has_list_data(result: Dict[str, Any], key: str) -> bool:
    """Check if a list field (topics, speakers, committee) is non-empty."""
    if not result:
        return False
    val = result.get(key, [])
    return isinstance(val, list) and len(val) > 0


_HAS_DATA: Dict[Category, Callable[[Dict[str, Any]], bool]] = {
    Category.OTHER: _has_other_data,
    Category.TOPICS: lambda r: _has_list_data(r, "topics"),
    Category.SPEAKERS: lambda r: _has_list_data(r, "keynote_speakers"),
    Category.COMMITTEE: lambda r: _has_list_data(r, "program_committee"),
}


# ---------------------------------------------------------------------------
# Helpers: merge results within a category
# ---------------------------------------------------------------------------

def _merge_other(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Merge OTHER results: fill nulls in base from extra."""
    merged = dict(base)
    for key in (
        "full_name", "acronym", "edition_number",
        "start_date", "end_date",
        "city", "country",
        "submission_deadline", "notification_date", "camera_ready_date",
        "publisher", "series",
    ):
        if not merged.get(key) and extra.get(key):
            merged[key] = extra[key]
    return merged


def _merge_list(base: Dict[str, Any], extra: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Merge list results: if base is empty, use extra."""
    merged = dict(base)
    if not merged.get(key) and extra.get(key):
        merged[key] = extra[key]
    return merged


def _merge_category(
    category: Category,
    base: Dict[str, Any],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge two results for the same category — base takes priority."""
    if not base:
        return extra or {}
    if not extra:
        return base
    if category == Category.OTHER:
        return _merge_other(base, extra)
    elif category == Category.TOPICS:
        return _merge_list(base, extra, "topics")
    elif category == Category.SPEAKERS:
        return _merge_list(base, extra, "keynote_speakers")
    elif category == Category.COMMITTEE:
        return _merge_list(base, extra, "program_committee")
    return base


# ---------------------------------------------------------------------------
# Per-category extraction: targeted (3 levels) and brute-force
# ---------------------------------------------------------------------------

def _run_extract(
    category: Category,
    context: str,
    *,
    model: str,
    backend: str,
    base_url: str,
    vllm_extra_args: Optional[List[str]],
    prompts_dir: Optional[str],
    all_warnings: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Call the LLM extraction function, then validate against context.

    Validation checks that extracted values actually appear in the
    context that was sent to the LLM. Hallucinated fields are nullified.
    """
    fn = _EXTRACT_FN[category]
    raw = fn(
        context,
        model=model,
        backend=backend,
        base_url=base_url,
        vllm_extra_args=vllm_extra_args,
        prompts_dir=prompts_dir,
    )
    if raw is None:
        return None

    validated, warnings = validate_category(category, raw, context)
    if warnings:
        logger.info(
            "Validation [%s]: %d warning(s): %s",
            category.value, len(warnings), warnings,
        )
        if all_warnings is not None:
            all_warnings.extend(warnings)

    return validated


def _extract_targeted(
    site: SiteContent,
    category: Category,
    *,
    model: str,
    backend: str,
    base_url: str,
    vllm_extra_args: Optional[List[str]],
    prompts_dir: Optional[str],
    all_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Algorithm 1: targeted extraction with 3 fallback levels.

    Each level sends only new (not yet processed) pages to the LLM.
    Each LLM response is validated against the context before merging.
    Stops early if the result already has sufficient data.
    """
    selector = PageSelector(site, category)
    has_data = _HAS_DATA[category]
    result: Dict[str, Any] = {}

    # Level 1: navigation (link_text + URL path)
    pages_1 = selector.select_by_navigation()
    if pages_1:
        ctx = build_context(pages_1)
        logger.info(
            "Algo1 L1 [%s]: %d page(s), %d chars",
            category.value, len(pages_1), len(ctx),
        )
        r = _run_extract(
            category, ctx,
            model=model, backend=backend, base_url=base_url,
            vllm_extra_args=vllm_extra_args, prompts_dir=prompts_dir,
            all_warnings=all_warnings,
        )
        if r:
            result = r
            if has_data(result):
                logger.info("Algo1 L1 [%s]: sufficient data found", category.value)
                return result

    # Level 2: content (keyword search in page text)
    pages_2 = selector.select_by_content()
    if pages_2:
        ctx = build_context(pages_2)
        logger.info(
            "Algo1 L2 [%s]: %d page(s), %d chars",
            category.value, len(pages_2), len(ctx),
        )
        r = _run_extract(
            category, ctx,
            model=model, backend=backend, base_url=base_url,
            vllm_extra_args=vllm_extra_args, prompts_dir=prompts_dir,
            all_warnings=all_warnings,
        )
        if r:
            result = _merge_category(category, result, r)
            if has_data(result):
                logger.info("Algo1 L2 [%s]: sufficient data found", category.value)
                return result

    # Level 3: everything remaining
    pages_3 = selector.select_remaining()
    if pages_3:
        ctx = build_context(pages_3)
        logger.info(
            "Algo1 L3 [%s]: %d page(s), %d chars",
            category.value, len(pages_3), len(ctx),
        )
        r = _run_extract(
            category, ctx,
            model=model, backend=backend, base_url=base_url,
            vllm_extra_args=vllm_extra_args, prompts_dir=prompts_dir,
            all_warnings=all_warnings,
        )
        if r:
            result = _merge_category(category, result, r)

    return result


def _extract_bruteforce(
    site: SiteContent,
    category: Category,
    *,
    model: str,
    backend: str,
    base_url: str,
    vllm_extra_args: Optional[List[str]],
    prompts_dir: Optional[str],
    all_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Algorithm 2: brute-force — send ALL pages to the LLM.
    Independent from algorithm 1, always runs.
    Response is validated against the full context.
    """
    all_pages = list(site.pages)
    if not all_pages:
        return {}
    ctx = build_context(all_pages)
    logger.info(
        "Algo2 [%s]: %d page(s), %d chars",
        category.value, len(all_pages), len(ctx),
    )
    r = _run_extract(
        category, ctx,
        model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args, prompts_dir=prompts_dir,
        all_warnings=all_warnings,
    )
    return r or {}


# ---------------------------------------------------------------------------
# Build final output structure
# ---------------------------------------------------------------------------

def _build_output(
    url: str,
    other: Dict[str, Any],
    topics: Dict[str, Any],
    speakers: Dict[str, Any],
    committee: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge all category results into the target JSON structure."""
    o = other or {}
    t = topics or {}
    s = speakers or {}
    c = committee or {}

    return {
        "conference": {
            "full_name": o.get("full_name") or "",
            "acronym": o.get("acronym") or "",
            "url": url,
            "edition_number": o.get("edition_number"),
        },
        "dates": {
            "start_date": o.get("start_date"),
            "end_date": o.get("end_date"),
        },
        "venue": {
            "city": o.get("city"),
            "country": o.get("country"),
        },
        "deadlines": {
            "submission": o.get("submission_deadline"),
            "notification": o.get("notification_date"),
            "camera_ready": o.get("camera_ready_date"),
        },
        "topics": t.get("topics", []),
        "keynote_speakers": s.get("keynote_speakers", []),
        "program_committee": c.get("program_committee", []),
        "publication": {
            "publisher": o.get("publisher"),
            "series": o.get("series"),
        },
    }


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_conference(
    url: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    base_url: Optional[str] = None,
    vllm_extra_args: Optional[List[str]] = None,
    prompts_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: scrape → extract (4 categories × 2 algorithms) →
    validate → merge → return JSON.

    Args:
        url:             Conference website URL.
        model:           Model name (e.g. "mistral:latest" for Ollama,
                         "Qwen/Qwen2.5-7B" for vLLM).
        backend:         "ollama" or "vllm".
        base_url:        Server URL. If None, uses default for the backend.
        vllm_extra_args: Extra CLI args for ``vllm serve`` (ignored for ollama).
        prompts_dir:     If set, save rendered prompts and raw LLM responses
                         to this directory for debugging.

    Returns a dict with keys:
      - "data":       the extracted conference JSON
      - "warnings":   list of validation warnings
      - "meta":       extraction metadata
    """
    if base_url is None:
        base_url = get_default_url(backend)

    # --- Step 1: Scrape all pages once ---
    logger.info("Fetching conference site: %s", url)
    site: SiteContent = fetch_conference_site(url)

    if not site.pages:
        logger.error("No pages fetched for %s", url)
        return {
            "data": _empty_result(url),
            "warnings": ["Could not fetch any pages"],
            "meta": {
                "model": model, "backend": backend,
                "pages_fetched": 0,
                "algo1_levels": {},
                "algo2_ran": False,
            },
        }

    logger.info("Fetched %d page(s) for %s", len(site.pages), url)

    # --- Step 2: Run both algorithms for each category ---
    llm_kwargs = dict(
        model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args, prompts_dir=prompts_dir,
    )

    category_results: Dict[Category, Dict[str, Any]] = {}
    warnings: List[str] = []

    for category in Category:
        logger.info("=== Category: %s ===", category.value)

        # Algorithm 1: targeted (3 levels)
        targeted = _extract_targeted(
            site, category, **llm_kwargs, all_warnings=warnings,
        )

        # Algorithm 2: brute-force (all pages)
        bruteforce = _extract_bruteforce(
            site, category, **llm_kwargs, all_warnings=warnings,
        )

        # Merge: targeted takes priority, brute-force fills gaps
        merged = _merge_category(category, targeted, bruteforce)
        category_results[category] = merged

        has_data = _HAS_DATA[category]
        if not has_data(merged):
            warnings.append(f"No data found for category: {category.value}")
            logger.warning("No data found for category: %s", category.value)

    # --- Step 3: Build final output ---
    data = _build_output(
        url,
        other=category_results.get(Category.OTHER, {}),
        topics=category_results.get(Category.TOPICS, {}),
        speakers=category_results.get(Category.SPEAKERS, {}),
        committee=category_results.get(Category.COMMITTEE, {}),
    )

    # --- Step 4: Cross-category validation ---
    # Check that speakers and committee don't overlap
    speaker_names = {
        s.get("name", "").strip().lower()
        for s in data.get("keynote_speakers", [])
        if s.get("name")
    }
    committee_names = {
        c.get("name", "").strip().lower()
        for c in data.get("program_committee", [])
        if c.get("name")
    }
    overlap = speaker_names & committee_names
    if overlap:
        warnings.append(
            f"Speaker/committee overlap detected ({len(overlap)} name(s)): "
            f"{', '.join(sorted(overlap))}"
        )
        logger.warning(
            "Speaker/committee overlap: %s — removing from committee",
            overlap,
        )
        # Remove overlapping names from committee (speakers take priority)
        data["program_committee"] = [
            c for c in data["program_committee"]
            if c.get("name", "").strip().lower() not in overlap
        ]

    return {
        "data": data,
        "warnings": warnings,
        "meta": {
            "model": model,
            "backend": backend,
            "pages_fetched": len(site.pages),
        },
    }
