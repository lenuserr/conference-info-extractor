"""
Target-aware page selection and context building.

The scraper fetches a conference site's main page plus a handful of
keyword-discovered subpages. That's a wide net — we typically end up with
5-8 pages of mixed content. Handing all of that to the LLM for every
extraction task is noisy and encourages hallucinations, especially for
narrow list fields like ``keynote_speakers`` and ``program_committee``.

This module selects, per extraction *target*, only the pages most likely
to contain that target's information and builds a compact LLM-ready
context string from them. The main page is almost always kept as a
baseline — conference home pages tend to carry identity, dates, and
venue info regardless of what other subpages exist.

Adding a new target is one entry in :data:`TARGET_LEXICONS` (and
optionally :data:`MAX_PAGES_PER_TARGET`). The pipeline picks it up by
calling :func:`build_context_for_target` for the new target and wiring
a matching extraction pass.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from .scraper import PageContent, SiteContent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

class Target(str, Enum):
    """
    Extraction target categories. Each target drives its own LLM pass and
    its own page selection. The enum value doubles as a short log label.
    """

    BASIC = "basic"          # identity, dates, venue, deadlines, publication, topics
    SPEAKERS = "speakers"    # keynote_speakers
    COMMITTEE = "committee"  # program_committee


# Keywords per target, applied to URL path, page title, and the text head.
# Keep them lowercased — the scorer normalizes whitespace/hyphens/underscores
# before comparing, so ``"call for papers"`` matches ``"call-for-papers"`` in
# a URL path or ``"Call For Papers"`` in a title.
TARGET_LEXICONS: Dict[Target, Tuple[str, ...]] = {
    Target.BASIC: (
        # Dates / deadlines
        "important dates", "dates", "deadline", "schedule",
        # Call for papers
        "call for papers", "call", "cfp", "submission", "submit", "papers",
        # Venue / location
        "venue", "location", "travel", "hotel", "accommodation",
        # Publication
        "publication", "proceedings", "indexing", "journal", "publisher",
        # Topics / scope
        "topic", "track", "theme", "area", "scope",
        # General anchors — conference home/about pages often carry identity
        "about", "home", "overview",
    ),
    Target.SPEAKERS: (
        "keynote", "invited speaker", "invited", "speaker", "plenary",
        # "program" pages frequently list speakers alongside the agenda
        "program",
    ),
    Target.COMMITTEE: (
        "committee", "program committee", "tpc", "organizing",
        "organizer", "chair", "chairs", "reviewer", "board",
        "people", "team",
    ),
}


# How many pages to include per target (main page counts toward the limit).
MAX_PAGES_PER_TARGET: Dict[Target, int] = {
    Target.BASIC: 6,
    Target.SPEAKERS: 3,
    Target.COMMITTEE: 3,
}


# Minimum per-target score required for a non-main page to be included.
# Main page is always kept (see ``include_main`` below). We'd rather show
# the LLM a smaller well-matched context than dilute it with unrelated
# pages — even for BASIC, pages scoring 0 on the (deliberately wide)
# BASIC lexicon are almost certainly irrelevant (committee, registration,
# sponsors, past-events, etc.).
_MIN_SCORE = 1


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@dataclass
class PageClassification:
    """A page plus its per-target scores (for selection and debugging)."""

    page: PageContent
    is_main: bool
    scores: Dict[Target, int] = field(default_factory=dict)


# Replace hyphens/underscores/slashes with spaces so multi-word keywords
# like "call for papers" match "call-for-papers" in URLs.
_NORMALIZE_RE = re.compile(r"[-_/]+")
_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    if not s:
        return ""
    s = _NORMALIZE_RE.sub(" ", s.lower())
    return _WS_RE.sub(" ", s).strip()


def _score_page(page: PageContent, keywords: Sequence[str]) -> int:
    """
    Score ``page`` against ``keywords`` with a simple weighted sum.

    Weights:
      - URL path:   3 per match
      - Title:      2 per match
      - Text head:  1 per match  (first 500 chars of cleaned page text)

    Keywords and the fields they're compared against are normalized the
    same way, so ``"call for papers"`` matches ``call-for-papers`` paths,
    ``Call For Papers`` titles, and ``"Call for Papers ..."`` text.
    """
    path = _normalize(urlparse(page.url).path)
    title = _normalize(page.title)
    head = _normalize(page.text[:500])

    score = 0
    for raw_kw in keywords:
        kw = _normalize(raw_kw)
        if not kw:
            continue
        if kw in path:
            score += 3
        if kw in title:
            score += 2
        if kw in head:
            score += 1
    return score


def classify_pages(site: SiteContent) -> List[PageClassification]:
    """Score every page in ``site`` against every :class:`Target`."""
    main_url = site.root_url
    classified: List[PageClassification] = []
    for page in site.pages:
        scores = {
            target: _score_page(page, lex)
            for target, lex in TARGET_LEXICONS.items()
        }
        classified.append(
            PageClassification(
                page=page,
                is_main=(page.url == main_url),
                scores=scores,
            )
        )
    return classified


# ---------------------------------------------------------------------------
# Selection + context building
# ---------------------------------------------------------------------------

def select_pages(
    site: SiteContent,
    target: Target,
    *,
    max_pages: Optional[int] = None,
    include_main: bool = True,
) -> List[PageContent]:
    """
    Return the pages most relevant for ``target``, ranked by per-target score.

    Main page is always kept first (when ``include_main`` is True) —
    conference home pages typically carry identity, venue, and date info
    even when more specific pages exist.

    Non-main pages must score at least ``_MIN_SCORE`` on the target's
    lexicon to be included. This is deliberate: an unrelated page (e.g.
    a sponsors list when we're looking for speakers) only adds noise, and
    the LLM does noticeably better on a smaller, focused context than on
    a padded-with-slop one.
    """
    if not site.pages:
        return []

    classifications = classify_pages(site)
    limit = (
        max_pages
        if max_pages is not None
        else MAX_PAGES_PER_TARGET.get(target, 5)
    )

    main_classes = [c for c in classifications if c.is_main]
    other_classes = [c for c in classifications if not c.is_main]
    other_classes = [
        c for c in other_classes if c.scores.get(target, 0) >= _MIN_SCORE
    ]
    other_classes.sort(
        key=lambda c: c.scores.get(target, 0),
        reverse=True,
    )

    selected: List[PageContent] = []
    slots = limit
    if include_main and main_classes:
        selected.append(main_classes[0].page)
        slots -= 1
    if slots > 0:
        selected.extend(c.page for c in other_classes[:slots])
    return selected


def build_context(pages: Sequence[PageContent]) -> str:
    """
    Concatenate pages into the ``=== PAGE: url ===``-framed text block
    that the LLM prompts expect.
    """
    parts: List[str] = []
    for p in pages:
        parts.append(f"=== PAGE: {p.url} ===\n{p.title}\n{p.text}")
    return "\n\n".join(parts)


def build_context_for_target(
    site: SiteContent,
    target: Target,
    *,
    max_pages: Optional[int] = None,
) -> str:
    """
    Convenience wrapper: pick target-relevant pages and build the LLM
    context string in one call. Emits a debug log with the chosen URLs.
    """
    pages = select_pages(site, target, max_pages=max_pages)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Context for %s: %d page(s) — %s",
            target.value,
            len(pages),
            [p.url for p in pages],
        )
    return build_context(pages)


def describe_selection(
    site: SiteContent,
) -> Dict[Target, List[Tuple[str, int]]]:
    """
    For debugging: return ``target → [(url, score), ...]`` sorted by score
    descending. Useful when a site produces unexpected selection results
    and you want to see the raw scores.
    """
    classifications = classify_pages(site)
    result: Dict[Target, List[Tuple[str, int]]] = {}
    for target in Target:
        pairs = [
            (c.page.url, c.scores.get(target, 0)) for c in classifications
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        result[target] = pairs
    return result
