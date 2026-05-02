"""
Category-aware page selection with 3-level fallback.

The scraper downloads the main page and all discovered subpages once.
This module then selects, per extraction *category*, which pages to
send to the LLM — progressing through three levels of broadening
search if earlier levels come up empty or the LLM returns gaps.

Levels:
  1. **Navigation** — match keywords against link_text + URL path only
     (the cheapest signal: how the site labels each page in its menu).
  2. **Content** — match keywords against the full text of every page
     (catches pages whose URL/link_text was uninformative but whose
     body contains the relevant information).
  3. **Remaining** — return every page not yet sent for this category
     (last-resort brute-force within the targeted algorithm).

The main page is always included at level 1, regardless of keyword
matches — conference home pages nearly always carry identity, dates,
and venue information.

A parallel brute-force algorithm (algorithm 2) independently sends
*all* pages to the LLM. Its results are merged with algorithm 1's
output at the pipeline level — this module only handles selection.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Dict, List, Sequence, Set, Tuple
from urllib.parse import urlparse

from .scraper import PageContent, SiteContent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

class Category(str, Enum):
    """
    Extraction categories.  Each category gets its own LLM prompt and
    its own page-selection pass (with up to 3 fallback levels).
    """
    OTHER = "other"          # conference name, dates, venue, deadlines, publication
    TOPICS = "topics"        # topics / scope / tracks
    SPEAKERS = "speakers"    # keynote speakers
    COMMITTEE = "committee"  # program committee / organizing committee


# Keywords per category.  Derived from real subpage URL analysis across
# 3,626 conference sites (72,239 subpages).
#
# Level 1 matches these against link_text and URL path (navigation).
# Level 2 matches these against page body text (content search).
CATEGORY_KEYWORDS: Dict[Category, Tuple[str, ...]] = {
    Category.OTHER: (
        # Dates / deadlines
        "program", "programme", "schedule",
        "important dates", "importantdates", "important_dates",
        "dates", "deadlines", "calendar",
        # Venue / location
        "venue", "conference venue", "location",
        "travel", "accommodation", "accomodation",
        "hotel", "hotels", "visa",
    ),
    Category.TOPICS: (
        "cfp", "call for papers", "call_for_paper", "call_for_papers",
        "callforpapers", "topics", "scope",
        "submission", "papers",
    ),
    Category.SPEAKERS: (
        "keynote", "keynotes", "keynote speakers",
        "speaker", "speakers",
        "invited", "invited speakers",
        "plenary", "panel", "panels", "tutorials",
    ),
    Category.COMMITTEE: (
        "committee", "committees",
        "organizing committee", "program committee",
        "technical program committee",
        "organization", "organizers",
        "team", "people", "chairs", "board",
        "editorial policy", "editorialpolicy",
    ),
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Replace hyphens/underscores/slashes with spaces so multi-word keywords
# like "call for papers" match "call-for-papers" in URLs.
_NORMALIZE_RE = re.compile(r"[-_/]+")
_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    if not s:
        return ""
    s = _NORMALIZE_RE.sub(" ", s.lower())
    return _WS_RE.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _matches_navigation(page: PageContent, keywords: Sequence[str]) -> bool:
    """
    Level 1 check: does the page's link_text or URL path contain any
    of the keywords?  This is the cheapest and most reliable signal —
    it's how the site itself labels the page in its navigation.
    """
    link = _normalize(page.link_text)
    path = _normalize(urlparse(page.url).path)
    for kw in keywords:
        nkw = _normalize(kw)
        if not nkw:
            continue
        if nkw in link or nkw in path:
            return True
    return False


def _matches_content(page: PageContent, keywords: Sequence[str]) -> bool:
    """
    Level 2 check: does the page's text body contain any of the keywords?
    Searches the full page text (not just the head) for broader recall.
    """
    text = _normalize(page.text)
    for kw in keywords:
        nkw = _normalize(kw)
        if not nkw:
            continue
        if nkw in text:
            return True
    return False


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def build_context(pages: Sequence[PageContent]) -> str:
    """
    Concatenate pages into the LLM-ready context block.

    Format per page::

        === PAGE: https://conf.org/committee/ ===
        Link: PROGRAM COMMITTEE
        Title: Conference 2026 - Program Committee
        <page text>

    The ``Link:`` line is the anchor text from the <a> tag that led to
    this page — invaluable context for the LLM to understand what kind
    of page it's reading.  Omitted for the main page (no inbound link).
    """
    parts: List[str] = []
    for p in pages:
        header = f"=== PAGE: {p.url} ==="
        if p.link_text:
            header += f"\nLink: {p.link_text}"
        header += f"\nTitle: {p.title}"
        parts.append(f"{header}\n{p.text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# PageSelector — stateful 3-level selector per category
# ---------------------------------------------------------------------------

class PageSelector:
    """
    Stateful page selector for a single category.

    Tracks which pages have already been sent to the LLM so that each
    subsequent level only returns new, unseen pages.  The main page is
    always included at level 1.

    Usage::

        selector = PageSelector(site, Category.COMMITTEE)

        # Level 1: navigation (link_text + URL path)
        pages_1 = selector.select_by_navigation()
        ctx_1 = selector.build_context(pages_1)
        # ... send to LLM, check result ...

        # Level 2: content (keyword search in page text)
        pages_2 = selector.select_by_content()
        ctx_2 = selector.build_context(pages_2)
        # ... send to LLM ...

        # Level 3: everything remaining
        pages_3 = selector.select_remaining()
        ctx_3 = selector.build_context(pages_3)
    """

    def __init__(self, site: SiteContent, category: Category) -> None:
        self.site = site
        self.category = category
        self.keywords = CATEGORY_KEYWORDS[category]
        self._used_urls: Set[str] = set()

    @property
    def main_page(self) -> PageContent | None:
        """The main (root) page, if present."""
        for p in self.site.pages:
            if p.url == self.site.root_url:
                return p
        return None

    def _mark_used(self, pages: List[PageContent]) -> None:
        for p in pages:
            self._used_urls.add(p.url)

    def _available(self) -> List[PageContent]:
        """Pages not yet returned by any level."""
        return [p for p in self.site.pages if p.url not in self._used_urls]

    def select_by_navigation(self) -> List[PageContent]:
        """
        Level 1: select pages whose link_text or URL path matches
        category keywords.  Main page is always included.
        """
        selected: List[PageContent] = []

        # Always include main page
        main = self.main_page
        if main and main.url not in self._used_urls:
            selected.append(main)

        for page in self.site.pages:
            if page.url in self._used_urls or page in selected:
                continue
            if _matches_navigation(page, self.keywords):
                selected.append(page)

        self._mark_used(selected)
        logger.debug(
            "L1 navigation [%s]: %d page(s) — %s",
            self.category.value,
            len(selected),
            [p.url for p in selected],
        )
        return selected

    def select_by_content(self) -> List[PageContent]:
        """
        Level 2: among remaining pages, select those whose text body
        contains category keywords.
        """
        available = self._available()
        selected = [p for p in available if _matches_content(p, self.keywords)]
        self._mark_used(selected)
        logger.debug(
            "L2 content [%s]: %d page(s) — %s",
            self.category.value,
            len(selected),
            [p.url for p in selected],
        )
        return selected

    def select_remaining(self) -> List[PageContent]:
        """
        Level 3: return all pages not yet sent for this category.
        """
        remaining = self._available()
        self._mark_used(remaining)
        logger.debug(
            "L3 remaining [%s]: %d page(s) — %s",
            self.category.value,
            len(remaining),
            [p.url for p in remaining],
        )
        return remaining

    def select_all(self) -> List[PageContent]:
        """
        Algorithm 2 (brute-force): return ALL pages regardless of
        what was already used.  Does NOT mark pages as used — this
        runs independently from the 3-level targeted algorithm.
        """
        return list(self.site.pages)

    def build_context(self, pages: Sequence[PageContent]) -> str:
        """Build LLM context string from a list of pages."""
        return build_context(pages)
