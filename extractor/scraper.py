"""
Web scraping module: fetch conference pages, discover subpages, clean HTML → text.

Uses a two-tier content extraction strategy:
  1. trafilatura (readability-style) — extracts main content, proven on millions of sites.
  2. Minimal fallback — strips only script/style/noscript; guarantees no content loss.
If trafilatura returns too little text we fall back to the minimal approach.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Subpage keywords that typically contain useful conference metadata
_SUBPAGE_KEYWORDS = [
    "call", "cfp", "submission", "submit", "important", "dates", "deadline",
    "keynote", "invited", "speaker", "program", "schedule", "venue",
    "location", "registration", "committee", "organiz", "publish",
    "proceeding", "topic", "track", "workshop", "about",
]

# Maximum pages to crawl per conference site
MAX_SUBPAGES = 8
# Request timeout (seconds)
REQUEST_TIMEOUT = 20
# Max text length per page (chars) to keep context window small for LLM
MAX_PAGE_TEXT = 12_000
# Minimum chars for trafilatura to be considered successful
_MIN_TRAFILATURA_LEN = 200

# Tags removed in the minimal-fallback path (only truly non-content elements)
_MINIMAL_STRIP_TAGS = {"script", "style", "noscript", "svg", "iframe"}


@dataclass
class PageContent:
    """Cleaned text content of a single page."""
    url: str
    title: str
    text: str


@dataclass
class SiteContent:
    """Aggregated content of a conference website (main page + subpages)."""
    root_url: str
    pages: List[PageContent] = field(default_factory=list)
    raw_html_main: str = ""  # raw HTML of the main page — used for validation

    @property
    def full_text(self) -> str:
        """Concatenated text of all pages, with page separators."""
        parts: List[str] = []
        for p in self.pages:
            parts.append(f"=== PAGE: {p.url} ===\n{p.title}\n{p.text}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Content extraction (two-tier)
# ---------------------------------------------------------------------------

def _table_to_text(table) -> str:
    """Convert an HTML table to a readable text representation."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_with_trafilatura(html: str) -> Optional[str]:
    """Primary extraction via trafilatura (readability-style algorithm)."""
    text = trafilatura.extract(
        html,
        include_tables=True,
        include_links=False,
        favor_recall=True,
    )
    if text and len(text) >= _MIN_TRAFILATURA_LEN:
        return text[:MAX_PAGE_TEXT]
    return None


def _extract_minimal(html: str) -> str:
    """Fallback: strip only script/style/noscript, preserve everything else."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(_MINIMAL_STRIP_TAGS):
        tag.decompose()

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Convert tables to structured text
    for table in soup.find_all("table"):
        table_text = _table_to_text(table)
        if table_text:
            table.replace_with(soup.new_string("\n" + table_text + "\n"))

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:MAX_PAGE_TEXT]


def _extract_content(html: str) -> str:
    """Extract text from HTML: trafilatura first, minimal fallback second."""
    text = _extract_with_trafilatura(html)
    if text is not None:
        return text
    logger.debug("trafilatura returned too little text, using minimal fallback")
    return _extract_minimal(html)


# ---------------------------------------------------------------------------
# HTTP & link helpers
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
    """GET with retries, timeout, and user-agent spoofing."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, url, exc)
    return None


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract page title."""
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return ""


def _same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs share the same domain (or subdomain)."""
    d1 = urlparse(url1).netloc.lower().replace("www.", "")
    d2 = urlparse(url2).netloc.lower().replace("www.", "")
    return d1 == d2


def _is_useful_link(href: str, text: str) -> bool:
    """Heuristic: does this link likely lead to a useful subpage?"""
    combined = (href + " " + text).lower()
    return any(kw in combined for kw in _SUBPAGE_KEYWORDS)


def _discover_subpages(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Find internal links that look like useful conference subpages."""
    seen: Set[str] = set()
    results: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Skip anchors, mailto, javascript, files
        if href.startswith(("#", "mailto:", "javascript:")) or href.endswith((".pdf", ".zip", ".docx")):
            continue
        full = urljoin(base_url, href).split("#")[0].split("?")[0]  # normalize
        if full in seen or not _same_domain(full, base_url):
            continue
        seen.add(full)
        link_text = a.get_text(strip=True)
        if _is_useful_link(href, link_text):
            results.append(full)
        if len(results) >= MAX_SUBPAGES:
            break
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_conference_site(url: str) -> SiteContent:
    """
    Fetch the main conference page and its relevant subpages.

    Returns a SiteContent object with cleaned text for each page and
    the raw HTML of the main page (used later for validation).
    """
    site = SiteContent(root_url=url)

    # 1. Fetch main page
    resp = _get(url)
    if resp is None:
        logger.error("Could not fetch main page: %s", url)
        return site

    site.raw_html_main = resp.text
    soup = BeautifulSoup(resp.text, "lxml")

    main_page = PageContent(
        url=url,
        title=_extract_title(soup),
        text=_extract_content(resp.text),
    )
    site.pages.append(main_page)

    # 2. Discover and fetch subpages
    subpage_urls = _discover_subpages(soup, url)
    logger.info("Discovered %d subpage candidates for %s", len(subpage_urls), url)

    visited: Set[str] = {url}
    for sub_url in subpage_urls:
        if sub_url in visited:
            continue
        visited.add(sub_url)
        sub_resp = _get(sub_url)
        if sub_resp is None:
            continue
        sub_soup = BeautifulSoup(sub_resp.text, "lxml")
        page = PageContent(
            url=sub_url,
            title=_extract_title(sub_soup),
            text=_extract_content(sub_resp.text),
        )
        site.pages.append(page)
        # Also keep raw HTML for validation
        site.raw_html_main += "\n" + sub_resp.text

    logger.info("Fetched %d pages total for %s", len(site.pages), url)
    return site
