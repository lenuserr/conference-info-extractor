#!/usr/bin/env python3
"""
Rule-based baseline extractor (no LLM).

Reads prepared contexts from ``prepare_contexts.py`` and extracts
conference information using only regex patterns and heuristics.
Produces output in the same JSON format as ``run_llm.py`` so that
``evaluate.py`` can compare results against the golden dataset.

Usage:
    python baseline.py --input gold_claude_prepared/ --output results/
    python baseline.py -i prepared/ -o results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

COUNTRIES = {
    "afghanistan", "albania", "algeria", "andorra", "angola",
    "argentina", "armenia", "australia", "austria", "azerbaijan",
    "bahrain", "bangladesh", "belarus", "belgium", "benin", "bhutan",
    "bolivia", "bosnia and herzegovina", "botswana", "brazil",
    "brunei", "bulgaria", "burkina faso", "burundi",
    "cambodia", "cameroon", "canada", "chad", "chile", "china",
    "colombia", "congo", "costa rica", "croatia", "cuba", "cyprus",
    "czech republic", "czechia",
    "denmark", "dominican republic",
    "ecuador", "egypt", "el salvador", "estonia", "ethiopia",
    "fiji", "finland", "france",
    "gabon", "gambia", "georgia", "germany", "ghana", "greece",
    "guatemala", "guinea",
    "haiti", "honduras", "hong kong", "hungary",
    "iceland", "india", "indonesia", "iran", "iraq", "ireland",
    "israel", "italy",
    "jamaica", "japan", "jordan",
    "kazakhstan", "kenya", "korea", "south korea", "kuwait",
    "kyrgyzstan",
    "laos", "latvia", "lebanon", "libya", "liechtenstein",
    "lithuania", "luxembourg",
    "macao", "macau", "madagascar", "malawi", "malaysia", "maldives",
    "mali", "malta", "mauritius", "mexico", "moldova", "monaco",
    "mongolia", "montenegro", "morocco", "mozambique", "myanmar",
    "namibia", "nepal", "netherlands", "new zealand", "nicaragua",
    "niger", "nigeria", "north macedonia", "norway",
    "oman",
    "pakistan", "palestine", "panama", "paraguay", "peru",
    "philippines", "poland", "portugal",
    "qatar",
    "romania", "russia", "rwanda",
    "saudi arabia", "senegal", "serbia", "singapore", "slovakia",
    "slovenia", "somalia", "south africa", "spain", "sri lanka",
    "sudan", "sweden", "switzerland", "syria",
    "taiwan", "tajikistan", "tanzania", "thailand", "togo",
    "trinidad and tobago", "tunisia", "turkey", "turkiye",
    "turkmenistan",
    "uganda", "ukraine", "united arab emirates", "uae",
    "united kingdom", "uk", "united states", "usa",
    "united states of america",
    "uruguay", "uzbekistan",
    "venezuela", "vietnam",
    "yemen",
    "zambia", "zimbabwe",
    # Common variants
    "u.s.a.", "u.s.", "u.k.",
}

# Normalize country names for output
_COUNTRY_NORMALIZE = {
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "turkiye": "Turkey",
    "czechia": "Czech Republic",
    "south korea": "South Korea",
    "korea": "South Korea",
    "hong kong": "Hong Kong",
    "uae": "United Arab Emirates",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _combine_contexts(cat_data: Dict) -> str:
    """Combine L1+L2+L3 context strings for a category."""
    parts = []
    for level in ("L1", "L2", "L3"):
        ctx = cat_data.get(level, {})
        if isinstance(ctx, dict):
            ctx = ctx.get("context", "")
        if ctx:
            parts.append(ctx)
    return "\n".join(parts)


def _parse_month(s: str) -> Optional[int]:
    return MONTHS.get(s.strip().lower())


def _fmt_date(year: int, month: int, day: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"


def _normalize_country(raw: str) -> str:
    """Normalize a country name to title case with known corrections."""
    key = raw.strip().lower()
    if key in _COUNTRY_NORMALIZE:
        return _COUNTRY_NORMALIZE[key]
    return raw.strip().title()


def _is_country(s: str) -> bool:
    """Check if a string is a known country name."""
    return s.strip().lower() in COUNTRIES


def _clean_name(name: str) -> str:
    """Strip academic titles from a person's name."""
    name = name.strip()
    # Remove common prefixes
    for prefix in ("Prof. Dr. ", "Prof.Dr. ", "Prof. ", "Professor ",
                   "Dr. ", "Assoc. Prof. ", "Assist. Prof. ",
                   "Asst. Prof. ", "Assoc Prof ", "Assoc.Prof. "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.strip()


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

# Pattern: "Month DD-DD, YYYY" or "Month DD ~ DD, YYYY"
_DATE_RANGE_SAME_MONTH = re.compile(
    r"(\w+)\s+(\d{1,2})\s*[-–~]\s*(\d{1,2})\s*,?\s*(\d{4})",
    re.IGNORECASE,
)

# Pattern: "Month DD - Month DD, YYYY" (cross-month)
_DATE_RANGE_CROSS_MONTH = re.compile(
    r"(\w+)\s+(\d{1,2})\s*[-–~]\s*(\w+)\s+(\d{1,2})\s*,?\s*(\d{4})",
    re.IGNORECASE,
)

# Pattern: "DD-DD Month YYYY"
_DATE_RANGE_EU = re.compile(
    r"(\d{1,2})\s*[-–~]\s*(\d{1,2})\s+(\w+)\s*,?\s*(\d{4})",
    re.IGNORECASE,
)

# Pattern: "DD Month - DD Month YYYY"
_DATE_RANGE_EU_CROSS = re.compile(
    r"(\d{1,2})\s+(\w+)\s*[-–~]\s*(\d{1,2})\s+(\w+)\s*,?\s*(\d{4})",
    re.IGNORECASE,
)

# Single date: "Month DD, YYYY"
_SINGLE_DATE = re.compile(
    r"(\w+)\s+(\d{1,2})\s*,?\s*(\d{4})",
    re.IGNORECASE,
)

# Single date EU: "DD Month YYYY"
_SINGLE_DATE_EU = re.compile(
    r"(\d{1,2})\s+(\w+)\s*,?\s*(\d{4})",
    re.IGNORECASE,
)

# ISO date: YYYY-MM-DD
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _find_date_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Find conference date range in text. Returns (start_date, end_date)."""
    # Try cross-month first (more specific)
    for m in _DATE_RANGE_CROSS_MONTH.finditer(text):
        m1 = _parse_month(m.group(1))
        m2 = _parse_month(m.group(3))
        if m1 and m2:
            d1, d2, year = int(m.group(2)), int(m.group(4)), int(m.group(5))
            if 2020 <= year <= 2030 and 1 <= d1 <= 31 and 1 <= d2 <= 31:
                return _fmt_date(year, m1, d1), _fmt_date(year, m2, d2)

    # Same month range
    for m in _DATE_RANGE_SAME_MONTH.finditer(text):
        month = _parse_month(m.group(1))
        if month:
            d1, d2, year = int(m.group(2)), int(m.group(3)), int(m.group(4))
            if 2020 <= year <= 2030 and 1 <= d1 <= 31 and 1 <= d2 <= 31:
                return _fmt_date(year, month, d1), _fmt_date(year, month, d2)

    # EU cross-month
    for m in _DATE_RANGE_EU_CROSS.finditer(text):
        m1 = _parse_month(m.group(2))
        m2 = _parse_month(m.group(4))
        if m1 and m2:
            d1, d2, year = int(m.group(1)), int(m.group(3)), int(m.group(5))
            if 2020 <= year <= 2030:
                return _fmt_date(year, m1, d1), _fmt_date(year, m2, d2)

    # EU same month
    for m in _DATE_RANGE_EU.finditer(text):
        month = _parse_month(m.group(3))
        if month:
            d1, d2, year = int(m.group(1)), int(m.group(2)), int(m.group(4))
            if 2020 <= year <= 2030:
                return _fmt_date(year, month, d1), _fmt_date(year, month, d2)

    return None, None


def _find_deadline_date(text: str, keywords: List[str]) -> Optional[str]:
    """Find a single date near one of the keywords."""
    text_lower = text.lower()
    for kw in keywords:
        idx = text_lower.find(kw.lower())
        if idx < 0:
            continue
        # Look in a window after the keyword
        window = text[idx:idx + 300]

        # Try ISO
        m = _ISO_DATE.search(window)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        # Try Month DD, YYYY
        m = _SINGLE_DATE.search(window)
        if m:
            month = _parse_month(m.group(1))
            if month:
                day, year = int(m.group(2)), int(m.group(3))
                if 2020 <= year <= 2030 and 1 <= day <= 31:
                    return _fmt_date(year, month, day)

        # Try DD Month YYYY
        m = _SINGLE_DATE_EU.search(window)
        if m:
            month = _parse_month(m.group(2))
            if month:
                day, year = int(m.group(1)), int(m.group(3))
                if 2020 <= year <= 2030 and 1 <= day <= 31:
                    return _fmt_date(year, month, day)

    return None


def extract_dates(text: str) -> Dict[str, Any]:
    """Extract conference dates and deadlines."""
    # Conference dates: look near keywords first, then anywhere
    start, end = None, None

    # Try near conference date keywords
    date_keywords = [
        "conference date", "will be held", "take place",
        "will take place", "held on", "held in",
    ]
    for kw in date_keywords:
        idx = text.lower().find(kw.lower())
        if idx < 0:
            continue
        window = text[idx:idx + 500]
        start, end = _find_date_range(window)
        if start:
            break

    # Fallback: first date range in the text
    if not start:
        start, end = _find_date_range(text[:3000])

    # Deadlines
    submission = _find_deadline_date(text, [
        "submission deadline", "paper submission", "full paper deadline",
        "abstract deadline", "submission due",
    ])
    notification = _find_deadline_date(text, [
        "notification", "acceptance notification", "author notification",
        "notification of acceptance",
    ])
    camera_ready = _find_deadline_date(text, [
        "camera-ready", "camera ready", "final paper", "final version",
    ])

    return {
        "start_date": start,
        "end_date": end,
        "submission_deadline": submission,
        "notification_date": notification,
        "camera_ready_date": camera_ready,
    }


# ---------------------------------------------------------------------------
# Venue extraction
# ---------------------------------------------------------------------------

_VENUE_PATTERNS = [
    # "in City, Country"  (after take place / held / etc.)
    re.compile(
        r"(?:in|venue|location)\s*:?\s*"
        r"([A-Z][a-zA-Zà-ÿ\s\-]+?)\s*,\s*"
        r"([A-Z][a-zA-Zà-ÿ\s\-]+)",
        re.IGNORECASE,
    ),
    # "City, Country" standalone on a line
    re.compile(
        r"^([A-Z][a-zA-Zà-ÿ\s\-]+?)\s*,\s*([A-Z][a-zA-Zà-ÿ\s\-]+)\s*$",
        re.MULTILINE,
    ),
]


def extract_venue(text: str) -> Dict[str, Optional[str]]:
    """Extract city and country."""
    # Try patterns
    for pat in _VENUE_PATTERNS:
        for m in pat.finditer(text[:5000]):
            candidate_city = m.group(1).strip()
            candidate_country = m.group(2).strip()
            # Validate country
            if _is_country(candidate_country):
                return {
                    "city": candidate_city,
                    "country": _normalize_country(candidate_country),
                }

    # Fallback: search for known country names in the first part of text
    text_lower = text[:5000].lower()
    for country in sorted(COUNTRIES, key=len, reverse=True):
        if len(country) < 4:
            continue
        idx = text_lower.find(country)
        if idx > 0:
            # Look for a city-like word before the country
            before = text[max(0, idx - 100):idx]
            # Find last comma-separated or newline-separated capitalized word
            cm = re.search(r"([A-Z][a-zA-Zà-ÿ\s\-]+?)\s*,?\s*$", before)
            if cm:
                city = cm.group(1).strip()
                if len(city) > 2 and city.lower() not in ("the", "and", "for"):
                    return {
                        "city": city,
                        "country": _normalize_country(country),
                    }

    return {"city": None, "country": None}


# ---------------------------------------------------------------------------
# Topics extraction
# ---------------------------------------------------------------------------

_TOPIC_HEADINGS = re.compile(
    r"(?:topics?\s+of\s+interest|topics?\s+include|"
    r"call\s+for\s+papers?|areas?\s+of\s+interest|"
    r"scope|tracks?|themes?|"
    r"topics?\b)",
    re.IGNORECASE,
)

_SECTION_END = re.compile(
    r"^(?:=== PAGE:|important\s+dates?|submission|committee|"
    r"keynote|invited|speaker|registration|publication|"
    r"schedule|program|contact|venue|location|"
    r"paper\s+submission|author\s+guidelines?)\b",
    re.IGNORECASE | re.MULTILINE,
)

_BULLET_RE = re.compile(r"^\s*[-*•·▪◆►●○]\s*")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s*")


def extract_topics(text: str) -> List[str]:
    """Extract research topics from text."""
    topics: List[str] = []

    # Find topic section
    for m in _TOPIC_HEADINGS.finditer(text):
        start = m.end()
        # Find section end
        end_m = _SECTION_END.search(text[start + 50:])
        end = start + 50 + end_m.start() if end_m else min(start + 5000, len(text))
        section = text[start:end]

        # Extract lines that look like topic items
        for line in section.split("\n"):
            line = line.strip()
            if not line or len(line) < 3:
                continue

            # Strip bullet/number prefix
            cleaned = _BULLET_RE.sub("", line)
            cleaned = _NUMBERED_RE.sub("", cleaned)
            cleaned = cleaned.strip()

            if not cleaned or len(cleaned) < 3:
                continue

            # Skip navigation-like items and headings
            if cleaned.lower() in ("home", "contact", "submit", "more details...",
                                   "back", "top", "menu", "skip to main content"):
                continue

            # Skip very long lines (likely paragraphs, not topics)
            if len(cleaned) > 200:
                continue

            # Skip lines that are clearly not topics
            if any(cleaned.lower().startswith(s) for s in
                   ("we invite", "authors are", "please submit",
                    "the conference", "this conference", "papers should",
                    "all papers", "submitted papers", "accepted papers",
                    "paper submissions", "submission", "http", "www.",
                    "email", "copyright", "©")):
                continue

            topics.append(cleaned)

    # Deduplicate (case-insensitive, preserve first occurrence)
    seen = set()
    unique = []
    for t in topics:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique


# ---------------------------------------------------------------------------
# Person extraction (shared by speakers & committee)
# ---------------------------------------------------------------------------

def _extract_persons_from_section(
    section: str,
    current_role: Optional[str] = None,
    include_role: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract person entries from a text section.

    Expected formats:
      - "Name, Affiliation, Country"  (one per line)
      - "Name, Affiliation"           (no country)
      - "Name\\nAffiliation"          (multi-line)

    Returns list of person dicts.
    """
    persons: List[Dict[str, Any]] = []
    lines = section.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or len(line) < 3:
            continue

        # Skip navigation/heading-like lines
        if line.startswith("===") or line.startswith("Link:") or line.startswith("Title:"):
            continue
        if line.lower() in ("skip to main content", "home", "contact",
                            "more details...", "back"):
            continue

        # Detect role headings (e.g., "General Chair", "Program Committee")
        is_heading = False
        line_lower = line.lower().rstrip(":")
        for role_kw in ("chair", "chairs", "co-chair", "co-chairs",
                        "committee", "organiz", "advisory", "board",
                        "reviewer", "editor"):
            if role_kw in line_lower:
                # Looks like a role heading if it's short
                if len(line) < 80:
                    current_role = line.rstrip(":")
                    is_heading = True
                    break
        if is_heading:
            continue

        # Skip lines that are clearly not person entries
        if any(line.lower().startswith(s) for s in
               ("the ", "this ", "we ", "our ", "please ", "for ", "all ",
                "paper", "submission", "accepted", "http", "www.", "email",
                "copyright", "©", "conference", "welcome", "dear ", "call ")):
            continue

        # Try to parse "Name, Affiliation, Country" or "Name, Affiliation"
        parts = [p.strip() for p in line.split(",")]

        # Filter out empty parts
        parts = [p for p in parts if p]

        if not parts:
            continue

        name = _clean_name(parts[0])
        affiliation = None
        country = None

        if not name or len(name) < 2:
            continue

        # Check if name looks like a person name (contains at least one space,
        # or is a short word that could be a single name)
        if not re.search(r"[A-Za-zà-ÿ]", name):
            continue

        if len(parts) >= 3:
            # Last part might be country
            last = parts[-1].strip()
            if _is_country(last):
                country = _normalize_country(last)
                affiliation = ", ".join(parts[1:-1]).strip()
            else:
                affiliation = ", ".join(parts[1:]).strip()
        elif len(parts) == 2:
            candidate = parts[1].strip()
            if _is_country(candidate):
                country = _normalize_country(candidate)
            else:
                affiliation = candidate
        # Single part = just a name

        # Skip if "name" is too long (probably a sentence)
        if len(name) > 60:
            continue

        # Skip if name doesn't look like a person name
        # (should contain letters and typically have spaces for full names)
        if not re.match(r"^[A-Za-zà-ÿ\s\.\-\']+$", name):
            continue

        person: Dict[str, Any] = {
            "name": name,
            "affiliation": affiliation,
            "country": country,
        }
        if include_role:
            person["role"] = current_role

        persons.append(person)

    return persons


# ---------------------------------------------------------------------------
# Speakers extraction
# ---------------------------------------------------------------------------

_SPEAKER_HEADING = re.compile(
    r"(?:keynote|invited\s+speaker|plenary\s+speaker|plenary\s+lecture|"
    r"tutorial\s+speaker|distinguished\s+speaker)",
    re.IGNORECASE,
)

_SPEAKER_SECTION_END = re.compile(
    r"^(?:=== PAGE:|committee|program\s+committee|"
    r"organizing|chairs?|registration|submission|"
    r"call\s+for|important\s+dates?|schedule|"
    r"publication|venue|contact|sponsors?)\b",
    re.IGNORECASE | re.MULTILINE,
)


def extract_speakers(text: str) -> List[Dict[str, Any]]:
    """Extract keynote/invited speakers."""
    # TBA check
    tba_pattern = re.compile(
        r"(?:keynote|speaker).*?(?:to be announced|tba|tbd|coming soon|to be confirmed)",
        re.IGNORECASE | re.DOTALL,
    )
    if tba_pattern.search(text[:5000]):
        return []

    speakers: List[Dict[str, Any]] = []

    for m in _SPEAKER_HEADING.finditer(text):
        start = m.end()
        end_m = _SPEAKER_SECTION_END.search(text[start + 20:])
        end = start + 20 + end_m.start() if end_m else min(start + 3000, len(text))
        section = text[start:end]

        persons = _extract_persons_from_section(section, include_role=False)
        for p in persons:
            # Remove role key if present
            p.pop("role", None)
            speakers.append(p)

    # Deduplicate by name
    seen = set()
    unique = []
    for s in speakers:
        key = s["name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


# ---------------------------------------------------------------------------
# Committee extraction
# ---------------------------------------------------------------------------

_COMMITTEE_HEADING = re.compile(
    r"(?:program\s+committee|scientific\s+committee|"
    r"organizing\s+committee|technical\s+committee|"
    r"conference\s+committee|advisory\s+(?:board|committee)|"
    r"committee\s+members?|"
    r"general\s+chair|program\s+chair|conference\s+chair|"
    r"chairs?\b)",
    re.IGNORECASE,
)

_COMMITTEE_SECTION_END = re.compile(
    r"^(?:=== PAGE:(?!.*committee)|keynote|invited\s+speaker|"
    r"plenary|call\s+for|submission|"
    r"important\s+dates?|schedule|publication|"
    r"venue|contact|sponsors?|registration)\b",
    re.IGNORECASE | re.MULTILINE,
)


def extract_committee(text: str) -> List[Dict[str, Any]]:
    """Extract program committee members."""
    members: List[Dict[str, Any]] = []

    for m in _COMMITTEE_HEADING.finditer(text):
        # Use the heading as initial role
        heading_start = text.rfind("\n", 0, m.start())
        heading_line = text[heading_start + 1:m.end()].strip() if heading_start >= 0 else m.group()

        start = m.end()
        end_m = _COMMITTEE_SECTION_END.search(text[start + 20:])
        end = start + 20 + end_m.start() if end_m else min(start + 10000, len(text))
        section = text[start:end]

        # Determine initial role from heading
        role = heading_line.rstrip(":").strip() if heading_line else None

        persons = _extract_persons_from_section(
            section, current_role=role, include_role=True,
        )
        members.extend(persons)

    # Deduplicate by name (keep first occurrence)
    seen = set()
    unique = []
    for m in members:
        key = m["name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique


# ---------------------------------------------------------------------------
# Conference identity extraction
# ---------------------------------------------------------------------------

_ORDINAL_RE = re.compile(r"(\d+)\s*(?:st|nd|rd|th)\b", re.IGNORECASE)
_ACRONYM_RE = re.compile(r"\(([A-Z][A-Z0-9\-&]+(?:\s*\d{4})?)\)")

_CONF_WORDS = ("conference", "workshop", "symposium", "congress", "forum",
               "summit", "colloquium")


def extract_identity(text: str) -> Dict[str, Any]:
    """Extract conference name, acronym, edition number."""
    first_block = text[:3000]

    # Acronym: uppercase abbreviation in parentheses
    acronym = None
    am = _ACRONYM_RE.search(first_block)
    if am:
        acr = am.group(1).strip()
        # Remove trailing year
        acr = re.sub(r"\s*\d{4}$", "", acr)
        if len(acr) >= 2:
            acronym = acr

    # Full name: look for line containing conference-type keywords
    full_name = None
    for line in first_block.split("\n"):
        line = line.strip()
        if not line or len(line) < 10 or len(line) > 200:
            continue
        line_lower = line.lower()
        if any(w in line_lower for w in _CONF_WORDS):
            # Clean up: remove year, edition, extra whitespace
            candidate = re.sub(r"\s+", " ", line).strip()
            if full_name is None or len(candidate) > len(full_name):
                full_name = candidate
            break

    # Edition number
    edition = None
    em = _ORDINAL_RE.search(first_block)
    if em:
        n = int(em.group(1))
        if 1 <= n <= 100:
            edition = n

    # Publisher
    publisher = None
    publishers = ["Springer", "IEEE", "ACM", "Elsevier", "MDPI", "Wiley"]
    for p in publishers:
        if p.lower() in text[:10000].lower():
            publisher = p
            break

    # Series
    series = None
    series_names = [
        "Lecture Notes in Computer Science",
        "Communications in Computer and Information Science",
        "ACM International Conference Proceeding Series",
        "Lecture Notes in Networks and Systems",
        "Proceedings of Machine Learning Research",
    ]
    for s in series_names:
        if s.lower() in text[:10000].lower():
            series = s
            break

    return {
        "full_name": full_name,
        "acronym": acronym,
        "edition_number": edition,
        "publisher": publisher,
        "series": series,
    }


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_baseline(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """Run full rule-based extraction on a prepared context."""
    url = prepared["url"]

    if prepared.get("error"):
        return _empty_result(url)

    categories = prepared.get("categories", {})

    # Combine contexts per category
    other_text = _combine_contexts(categories.get("other", {}))
    topics_text = _combine_contexts(categories.get("topics", {}))
    speakers_text = _combine_contexts(categories.get("speakers", {}))
    committee_text = _combine_contexts(categories.get("committee", {}))

    # Extract
    dates = extract_dates(other_text)
    venue = extract_venue(other_text)
    identity = extract_identity(other_text)
    topics = extract_topics(topics_text)
    speakers = extract_speakers(speakers_text)
    committee = extract_committee(committee_text)

    return {
        "conference": {
            "full_name": identity.get("full_name") or "",
            "acronym": identity.get("acronym") or "",
            "url": url,
            "edition_number": identity.get("edition_number"),
        },
        "dates": {
            "start_date": dates.get("start_date"),
            "end_date": dates.get("end_date"),
        },
        "venue": {
            "city": venue.get("city"),
            "country": venue.get("country"),
        },
        "deadlines": {
            "submission": dates.get("submission_deadline"),
            "notification": dates.get("notification_date"),
            "camera_ready": dates.get("camera_ready_date"),
        },
        "topics": topics,
        "keynote_speakers": speakers,
        "program_committee": committee,
        "publication": {
            "publisher": identity.get("publisher"),
            "series": identity.get("series"),
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rule-based baseline extractor (no LLM)",
    )
    parser.add_argument(
        "--input", "-i", default="gold_claude_prepared",
        help="Directory with prepared contexts (default: gold_claude_prepared/)",
    )
    parser.add_argument(
        "--output", "-o", default="results",
        help="Output directory (default: results/)",
    )
    args = parser.parse_args()

    # Load prepared contexts
    prepared_files = sorted([
        f for f in os.listdir(args.input) if f.endswith(".json")
    ])
    if not prepared_files:
        print(f"ERROR: no prepared contexts found in {args.input}/")
        return

    out_dir = os.path.join(args.output, "baseline")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Input:  {args.input}")
    print(f"Output: {out_dir}")
    print(f"Sites:  {len(prepared_files)}")
    print()

    start_time = time.time()

    for prep_file in prepared_files:
        prep_path = os.path.join(args.input, prep_file)
        with open(prep_path, encoding="utf-8") as f:
            prepared = json.load(f)

        url = prepared["url"]
        logger.info("Processing %s", url)

        t0 = time.time()
        data = extract_baseline(prepared)
        elapsed = round(time.time() - t0, 3)

        entry = {
            "url": url,
            "model": "baseline",
            "backend": "regex",
            "status": "ok",
            "elapsed_sec": elapsed,
            "pages_fetched": prepared.get("pages_fetched", 0),
            "warnings": [],
            "data": data,
        }

        out_path = os.path.join(out_dir, prep_file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

        print(f"  {url} -> {elapsed}s")

    total = round(time.time() - start_time, 1)
    print(f"\nDone in {total}s. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
