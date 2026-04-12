"""
Web scraping module: fetch conference pages, discover subpages, clean HTML → text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

# Subpage keywords that typically contain useful conference metadata
_SUBPAGE_KEYWORDS = [
    "call", "cfp", "submission", "submit", "important", "dates", "deadline",
    "keynote", "invited", "speaker", "program", "schedule", "venue",
    "location", "registration", "committee", "organiz", "publish",
    "proceeding", "topic", "track", "workshop", "about",
]

# Tags/classes/ids that are navigation/chrome — not useful content
_STRIP_TAGS = {"script", "style", "nav", "footer", "noscript", "svg", "iframe"}
# Tags that should only be stripped when they contain a small portion of the page
_CONDITIONAL_STRIP_TAGS = {"header"}
_CONDITIONAL_STRIP_RATIO = 0.3  # strip only if tag text < 30% of total text

_STRIP_CLASS_RE = re.compile(
    r"\b(?:nav(?:bar|igation)?|menu|foot(?:er)?|sidebar|cookie|banner|popup|modal"
    r"|breadcrumb|social[-_]?(?:media|link|icon|share)|share[-_]?(?:bar|button))\b",
    re.I,
)
# Structural elements that must never be removed by class/id stripping
_PROTECTED_TAGS = {"body", "html", "main", "article", "section"}

# Maximum pages to crawl per conference site
MAX_SUBPAGES = 8
# Request timeout (seconds)
REQUEST_TIMEOUT = 20
# Max text length per page (chars) to keep context window small for LLM
MAX_PAGE_TEXT = 12_000


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
# Internal helpers
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


def _table_to_text(table) -> str:
    """Convert an HTML table to a readable text representation."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _clean_html(soup: BeautifulSoup) -> str:
    """Remove boilerplate elements and return cleaned text with structure."""
    # Remove unwanted tags entirely
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    # Conditionally strip tags (header) only when they hold a small portion
    total_text_len = len(soup.get_text(strip=True))
    if total_text_len > 0:
        for tag_name in _CONDITIONAL_STRIP_TAGS:
            for tag in soup.find_all(tag_name):
                tag_text_len = len(tag.get_text(strip=True))
                if tag_text_len / total_text_len < _CONDITIONAL_STRIP_RATIO:
                    tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove elements with navigation/footer-like class or id,
    # but never remove structural/container elements or large content blocks
    remaining_text_len = len(soup.get_text(strip=True))
    for el in soup.find_all(True):
        if el.name in _PROTECTED_TAGS:
            continue
        if el.attrs is None:
            continue
        cls = " ".join(el.get("class", []))
        el_id = el.get("id", "") or ""
        if _STRIP_CLASS_RE.search(cls) or _STRIP_CLASS_RE.search(el_id):
            el_text_len = len(el.get_text(strip=True))
            # Don't remove elements that hold most of the remaining content
            if remaining_text_len > 0 and el_text_len / remaining_text_len > _CONDITIONAL_STRIP_RATIO:
                continue
            remaining_text_len -= el_text_len
            el.decompose()

    # Convert tables to structured text before extracting plain text
    for table in soup.find_all("table"):
        table_text = _table_to_text(table)
        if table_text:
            table.replace_with(soup.new_string("\n" + table_text + "\n"))

    text = soup.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:MAX_PAGE_TEXT]


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
        text=_clean_html(BeautifulSoup(resp.text, "lxml")),  # fresh copy
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
            text=_clean_html(BeautifulSoup(sub_resp.text, "lxml")),
        )
        site.pages.append(page)
        # Also keep raw HTML for validation
        site.raw_html_main += "\n" + sub_resp.text

    logger.info("Fetched %d pages total for %s", len(site.pages), url)
    return site