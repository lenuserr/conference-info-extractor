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

# Tags/classes/ids that are navigation/chrome — not useful content
_STRIP_TAGS = {"script", "style", "nav", "footer", "noscript", "svg", "iframe"}
# Ratio threshold for conditional stripping of elements by class/id.
_CONDITIONAL_STRIP_RATIO = 0.3

_STRIP_CLASS_RE = re.compile(
    r"\b(?:nav(?:bar|igation)?|menu|foot(?:er)?|sidebar|cookie|banner|popup|modal"
    r"|breadcrumb|social[-_]?(?:media|link|icon|share)|share[-_]?(?:bar|button))\b",
    re.I,
)
# Structural elements that must never be removed by class/id stripping
_PROTECTED_TAGS = {"body", "html", "main", "article", "section"}

# Request timeout (seconds)
REQUEST_TIMEOUT = 10
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
    """GET with a single attempt, timeout, and user-agent spoofing."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
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
    # Remove unwanted tags entirely (nav, footer, script, style, etc.)
    # Note: <header> is NOT stripped — after <nav> removal, it typically
    # contains the hero/banner with conference name, dates, and venue,
    # which is exactly the most valuable content.
    for tag in soup.find_all(_STRIP_TAGS):
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


# URL path segments that almost never contain useful content.
# Used as a lightweight negative filter instead of the old keyword allowlist.
_JUNK_PATH_RE = re.compile(
    r"(?:^|/)"
    r"(?:login|logout|signin|signup|register|cart|checkout|shop|buy|donate"
    r"|privacy|cookie|gdpr|terms|tos|legal|disclaimer|imprint|impressum"
    r"|wp-admin|wp-login|wp-content|cgi-bin|assets|static|media|images?"
    r"|feed|rss|atom|sitemap|robots"
    r"|[?&](?:lang|session|token|utm_))"
    r"(?:/|$|[?&#])",
    re.I,
)

# File extensions to skip (binary / non-HTML resources)
_SKIP_EXTENSIONS = (
    ".pdf", ".zip", ".gz", ".tar", ".rar",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv",
    ".css", ".js", ".json", ".xml", ".bib", ".tex",
)


def _is_junk_link(url: str) -> bool:
    """Return True if the URL path looks like a non-content page."""
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return True
    return bool(_JUNK_PATH_RE.search(path))


def _discover_subpages(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Find all internal links on the page, excluding obvious junk.

    Unlike the previous keyword-allowlist approach, this collects *every*
    same-domain link that isn't clearly non-content (login, assets, binary
    files, etc.).  The downstream ``content_selection`` module is
    responsible for ranking pages by relevance to each extraction target —
    duplicating that logic here was both redundant and lossy.
    """
    seen: Set[str] = set()
    results: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Skip anchors, mailto, javascript
        if href.startswith(("#", "mailto:", "javascript:")):
            continue
        full = urljoin(base_url, href).split("#")[0].split("?")[0]  # normalize
        if full in seen or full == base_url:
            continue
        if not _same_domain(full, base_url):
            continue
        seen.add(full)
        if _is_junk_link(full):
            continue
        results.append(full)
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