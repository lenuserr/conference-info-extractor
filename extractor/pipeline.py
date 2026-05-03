"""
Main extraction pipeline: scrape → category-aware selection →
LLM extract (4 categories × 3 fallback levels + brute-force) →
validate → merge → output.

Two modes:
  - **Live** (``extract_conference``): scrapes the site, builds contexts,
    runs LLM — all in one process. Needs internet + LLM.
  - **Prepared** (``extract_from_prepared``): reads pre-built contexts
    from ``prepare_contexts.py``, runs LLM only. No internet needed.

Both modes share the same core extraction logic.
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
    """Check if OTHER result has enough data to skip further levels."""
    if not result:
        return False
    has_name = bool(result.get("full_name"))
    has_dates = bool(result.get("start_date"))
    has_venue = bool(result.get("city"))
    has_deadline = bool(result.get("submission_deadline"))
    return has_name and (has_dates or has_venue or has_deadline)


def _has_list_data(result: Dict[str, Any], key: str) -> bool:
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
# Completeness checks — stricter than _HAS_DATA, used to skip brute-force
# ---------------------------------------------------------------------------

def _is_other_complete(result: Dict[str, Any]) -> bool:
    """All key OTHER fields are filled — no need for brute-force."""
    if not result:
        return False
    has_name = bool(result.get("full_name"))
    has_dates = bool(result.get("start_date") and result.get("end_date"))
    has_venue = bool(result.get("city") and result.get("country"))
    has_deadline = bool(result.get("submission_deadline"))
    return has_name and has_dates and has_venue and has_deadline


_IS_COMPLETE: Dict[Category, Callable[[Dict[str, Any]], bool]] = {
    Category.OTHER: _is_other_complete,
    Category.TOPICS: lambda r: _has_list_data(r, "topics"),
    Category.SPEAKERS: lambda r: _has_list_data(r, "keynote_speakers"),
    Category.COMMITTEE: lambda r: _has_list_data(r, "program_committee"),
}


# ---------------------------------------------------------------------------
# Helpers: merge results within a category
# ---------------------------------------------------------------------------

_OTHER_FIELDS = (
    "full_name", "acronym", "edition_number",
    "start_date", "end_date",
    "city", "country",
    "submission_deadline", "notification_date", "camera_ready_date",
    "publisher", "series",
)


def _merge_other(
    base: Dict[str, Any],
    extra: Dict[str, Any],
    sources: Dict[str, str],
    level: str,
) -> Dict[str, Any]:
    merged = dict(base)
    for key in _OTHER_FIELDS:
        if not merged.get(key) and extra.get(key):
            merged[key] = extra[key]
            sources[key] = level
    return merged


def _merge_list(
    base: Dict[str, Any],
    extra: Dict[str, Any],
    key: str,
    sources: Dict[str, str],
    level: str,
) -> Dict[str, Any]:
    merged = dict(base)
    if not merged.get(key) and extra.get(key):
        merged[key] = extra[key]
        sources[key] = level
    return merged


def _merge_category(
    category: Category,
    base: Dict[str, Any],
    extra: Dict[str, Any],
    sources: Dict[str, str],
    level: str,
) -> Dict[str, Any]:
    if not base:
        if extra:
            # First result — record sources for all non-empty fields
            if category == Category.OTHER:
                for key in _OTHER_FIELDS:
                    if extra.get(key):
                        sources.setdefault(key, level)
            elif category == Category.TOPICS:
                if extra.get("topics"):
                    sources.setdefault("topics", level)
            elif category == Category.SPEAKERS:
                if extra.get("keynote_speakers"):
                    sources.setdefault("keynote_speakers", level)
            elif category == Category.COMMITTEE:
                if extra.get("program_committee"):
                    sources.setdefault("program_committee", level)
        return extra or {}
    if not extra:
        return base
    if category == Category.OTHER:
        return _merge_other(base, extra, sources, level)
    elif category == Category.TOPICS:
        return _merge_list(base, extra, "topics", sources, level)
    elif category == Category.SPEAKERS:
        return _merge_list(base, extra, "keynote_speakers", sources, level)
    elif category == Category.COMMITTEE:
        return _merge_list(base, extra, "program_committee", sources, level)
    return base


# ---------------------------------------------------------------------------
# LLM call + validation
# ---------------------------------------------------------------------------

def _run_extract(
    category: Category,
    context: str,
    *,
    model: str,
    backend: str,
    base_url: str,
    vllm_extra_args: Optional[List[str]],
    prompts_dir: Optional[str] = None,
    all_warnings: Optional[List[str]] = None,
    all_failures: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Call the LLM extraction function, then validate against context."""
    fn = _EXTRACT_FN[category]
    raw = fn(
        context,
        model=model,
        backend=backend,
        base_url=base_url,
        vllm_extra_args=vllm_extra_args,
        prompts_dir=prompts_dir,
        all_failures=all_failures,
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


# ---------------------------------------------------------------------------
# Build contexts from SiteContent (for live mode)
# ---------------------------------------------------------------------------

def build_all_contexts(site: SiteContent) -> Dict[str, Dict[str, str]]:
    """
    Build contexts for all categories × all levels from a SiteContent.

    Returns a dict matching the structure of prepare_contexts.py output::

        {
            "other": {"L1": "...", "L2": "...", "L3": "...", "bruteforce": "..."},
            "topics": {...},
            "speakers": {...},
            "committee": {...},
        }
    """
    contexts: Dict[str, Dict[str, str]] = {}
    all_pages_ctx = build_context(list(site.pages))

    for category in Category:
        selector = PageSelector(site, category)

        pages_1 = selector.select_by_navigation()
        pages_2 = selector.select_by_content()
        pages_3 = selector.select_remaining()

        contexts[category.value] = {
            "L1": build_context(pages_1) if pages_1 else "",
            "L2": build_context(pages_2) if pages_2 else "",
            "L3": build_context(pages_3) if pages_3 else "",
            "bruteforce": all_pages_ctx,
        }

    return contexts


# ---------------------------------------------------------------------------
# Core extraction logic (shared between live and prepared modes)
# ---------------------------------------------------------------------------

def _build_output(
    url: str,
    other: Dict[str, Any],
    topics: Dict[str, Any],
    speakers: Dict[str, Any],
    committee: Dict[str, Any],
) -> Dict[str, Any]:
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


def _extract_from_contexts(
    url: str,
    contexts: Dict[str, Dict[str, str]],
    *,
    model: str,
    backend: str,
    base_url: str,
    vllm_extra_args: Optional[List[str]],
    prompts_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[str], Dict[str, str], List[Dict[str, Any]]]:
    """
    Core extraction logic: run 4 categories × (targeted + brute-force)
    using pre-built context strings.

    Returns (data, warnings, sources, failures).
    ``sources`` maps field names to the level that filled them
    (e.g. ``{"full_name": "L1", "topics": "bruteforce"}``).
    ``failures`` is a list of dicts with failed LLM parse attempts,
    each containing ``url``, ``category``, ``level``, ``reason``,
    ``prompt``, ``raw_response``.
    """
    llm_kwargs = dict(
        model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args, prompts_dir=prompts_dir,
    )

    category_results: Dict[Category, Dict[str, Any]] = {}
    warnings: List[str] = []
    sources: Dict[str, str] = {}  # field_name -> level
    failures: List[Dict[str, Any]] = []  # failed LLM attempts

    for category in Category:
        cat_key = category.value
        cat_contexts = contexts.get(cat_key, {})
        has_data = _HAS_DATA[category]

        logger.info("=== Category: %s ===", cat_key)

        # Algorithm 1: targeted (3 levels)
        targeted: Dict[str, Any] = {}

        for level in ("L1", "L2", "L3"):
            ctx = cat_contexts.get(level, "")
            if not ctx:
                continue

            logger.info("Algo1 %s [%s]: %d chars", level, cat_key, len(ctx))
            n_before = len(failures)
            r = _run_extract(
                category, ctx,
                all_warnings=warnings, all_failures=failures,
                **llm_kwargs,
            )
            # Enrich any new failures with context
            for f in failures[n_before:]:
                f["url"] = url
                f["category"] = cat_key
                f["level"] = level
            if r:
                targeted = _merge_category(category, targeted, r, sources, level)
                if has_data(targeted):
                    logger.info("Algo1 %s [%s]: sufficient data found", level, cat_key)
                    break

        # Algorithm 2: brute-force (skip if targeted already found everything)
        is_complete = _IS_COMPLETE[category]
        bruteforce: Dict[str, Any] = {}

        if is_complete(targeted):
            logger.info(
                "Algo1 [%s]: complete data found, skipping brute-force",
                cat_key,
            )
        else:
            bf_ctx = cat_contexts.get("bruteforce", "")
            if bf_ctx:
                logger.info("Algo2 [%s]: %d chars", cat_key, len(bf_ctx))
                n_before = len(failures)
                r = _run_extract(
                    category, bf_ctx,
                    all_warnings=warnings, all_failures=failures,
                    **llm_kwargs,
                )
                for f in failures[n_before:]:
                    f["url"] = url
                    f["category"] = cat_key
                    f["level"] = "bruteforce"
                if r:
                    bruteforce = r

        # Merge: targeted priority, brute-force fills gaps
        merged = _merge_category(category, targeted, bruteforce, sources, "bruteforce")
        category_results[category] = merged

        if not has_data(merged):
            warnings.append(f"No data found for category: {cat_key}")
            logger.warning("No data found for category: %s", cat_key)

    # Build final output
    data = _build_output(
        url,
        other=category_results.get(Category.OTHER, {}),
        topics=category_results.get(Category.TOPICS, {}),
        speakers=category_results.get(Category.SPEAKERS, {}),
        committee=category_results.get(Category.COMMITTEE, {}),
    )

    # Cross-category validation: speakers vs committee overlap
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
        data["program_committee"] = [
            c for c in data["program_committee"]
            if c.get("name", "").strip().lower() not in overlap
        ]

    logger.info("Field sources: %s", sources)
    if failures:
        logger.warning("%d LLM parse failure(s) for %s", len(failures), url)
    return data, warnings, sources, failures


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
    Live mode: scrape → build contexts → extract → validate → merge.

    Needs both internet access and an LLM backend.
    """
    if base_url is None:
        base_url = get_default_url(backend)

    logger.info("Fetching conference site: %s", url)
    site: SiteContent = fetch_conference_site(url)

    if not site.pages:
        logger.error("No pages fetched for %s", url)
        return {
            "data": _empty_result(url),
            "warnings": ["Could not fetch any pages"],
            "meta": {"model": model, "backend": backend, "pages_fetched": 0},
        }

    logger.info("Fetched %d page(s) for %s", len(site.pages), url)

    contexts = build_all_contexts(site)
    data, warnings, sources, failures = _extract_from_contexts(
        url, contexts,
        model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args, prompts_dir=prompts_dir,
    )

    return {
        "data": data,
        "warnings": warnings,
        "sources": sources,
        "failures": failures,
        "meta": {
            "model": model,
            "backend": backend,
            "pages_fetched": len(site.pages),
        },
    }


def extract_from_prepared(
    prepared: Dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    base_url: Optional[str] = None,
    vllm_extra_args: Optional[List[str]] = None,
    prompts_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepared mode: read pre-built contexts → extract → validate → merge.

    No internet needed — only an LLM backend. The ``prepared`` dict
    is the output of ``prepare_contexts.py`` (one site).
    """
    if base_url is None:
        base_url = get_default_url(backend)

    url = prepared["url"]

    if prepared.get("error"):
        return {
            "data": _empty_result(url),
            "warnings": [prepared["error"]],
            "meta": {
                "model": model, "backend": backend,
                "pages_fetched": 0,
            },
        }

    # Convert prepared format to simple {category: {level: context_str}}
    contexts: Dict[str, Dict[str, str]] = {}
    for cat_key, cat_data in prepared.get("categories", {}).items():
        contexts[cat_key] = {
            level: level_data.get("context", "")
            for level, level_data in cat_data.items()
        }

    data, warnings, sources, failures = _extract_from_contexts(
        url, contexts,
        model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args, prompts_dir=prompts_dir,
    )

    return {
        "data": data,
        "warnings": warnings,
        "sources": sources,
        "failures": failures,
        "meta": {
            "model": model,
            "backend": backend,
            "pages_fetched": prepared.get("pages_fetched", 0),
        },
    }
