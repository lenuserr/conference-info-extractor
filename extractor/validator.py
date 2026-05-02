"""
Validation layer for extracted conference metadata.

Validates LLM output against the context that was sent to it (not raw HTML).
Called after each LLM call in the pipeline.

1. Source verification — check extracted values appear in the LLM context
2. Date validation — parse all dates from context, normalize, compare
3. Nullification — remove fields not found in context (likely hallucinated)
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from thefuzz import fuzz

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date parsing from free text
# ---------------------------------------------------------------------------

_MONTH_NAMES: List[Tuple[str, str]] = [
    ("January", "Jan"),
    ("February", "Feb"),
    ("March", "Mar"),
    ("April", "Apr"),
    ("May", "May"),
    ("June", "Jun"),
    ("July", "Jul"),
    ("August", "Aug"),
    ("September", "Sep"),
    ("October", "Oct"),
    ("November", "Nov"),
    ("December", "Dec"),
]

_MONTH_LOOKUP: Dict[str, int] = {
    name.lower(): i + 1
    for i, (full_name, abbr) in enumerate(_MONTH_NAMES)
    for name in (full_name, abbr)
}

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_iso_date(s: Optional[str]) -> Optional[date]:
    """Parse a YYYY-MM-DD string into a date object."""
    if s and _ISO_DATE_RE.match(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            pass
    return None


def _extract_dates_from_text(text: str) -> Set[date]:
    """
    Extract all dates from free text in any common format and return
    them as a set of normalized date objects.

    Handles:
    - "August 19, 2026", "Aug 19 2026", "19 August 2026", "19th August 2026"
    - "August 19-21, 2026" → {Aug 19, Aug 20, Aug 21}
    - "19 ~ 22 May, 2026" → {May 19, May 20, May 21, May 22}
    - "2026-08-19", "19/08/2026", "08/19/2026", "2026.08.19"
    - "2026/05/21"
    """
    dates: Set[date] = set()
    if not text:
        return dates

    month_alt = "|".join(sorted(_MONTH_LOOKUP.keys(), key=len, reverse=True))

    # Pattern 1: "Month Day[-–—~]Day[,] Year" — date ranges with month first
    # e.g. "August 19-21, 2026", "May 21 ~ 22, 2026"
    p_range_month_first = re.compile(
        rf"\b({month_alt})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?"
        rf"\s*[-–—~]\s*"
        rf"(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})\b",
        re.IGNORECASE,
    )
    for m in p_range_month_first.finditer(text):
        month_name, d_start, d_end, year = m.groups()
        month_num = _MONTH_LOOKUP.get(month_name.lower().rstrip("."))
        if month_num:
            for day in range(int(d_start), int(d_end) + 1):
                try:
                    dates.add(date(int(year), month_num, day))
                except ValueError:
                    pass

    # Pattern 2: "Day[-–—~]Day Month[,] Year" — date ranges with month last
    # e.g. "19-21 August 2026", "19 ~ 22 May, 2026"
    p_range_month_last = re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?"
        rf"\s*[-–—~]\s*"
        rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_alt})\.?\s*,?\s*(\d{{4}})\b",
        re.IGNORECASE,
    )
    for m in p_range_month_last.finditer(text):
        d_start, d_end, month_name, year = m.groups()
        month_num = _MONTH_LOOKUP.get(month_name.lower().rstrip("."))
        if month_num:
            for day in range(int(d_start), int(d_end) + 1):
                try:
                    dates.add(date(int(year), month_num, day))
                except ValueError:
                    pass

    # Pattern 3: "Month Day[,] Year" — single date with month name first
    # e.g. "August 19, 2026", "Aug 19 2026"
    p_single_month_first = re.compile(
        rf"\b({month_alt})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})\b",
        re.IGNORECASE,
    )
    for m in p_single_month_first.finditer(text):
        month_name, day, year = m.groups()
        month_num = _MONTH_LOOKUP.get(month_name.lower().rstrip("."))
        if month_num:
            try:
                dates.add(date(int(year), month_num, int(day)))
            except ValueError:
                pass

    # Pattern 4: "Day Month[,] Year" — single date with month name last
    # e.g. "19 August 2026", "19th of August, 2026"
    p_single_month_last = re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_alt})\.?\s*,?\s*(\d{{4}})\b",
        re.IGNORECASE,
    )
    for m in p_single_month_last.finditer(text):
        day, month_name, year = m.groups()
        month_num = _MONTH_LOOKUP.get(month_name.lower().rstrip("."))
        if month_num:
            try:
                dates.add(date(int(year), month_num, int(day)))
            except ValueError:
                pass

    # Pattern 5: ISO and numeric formats
    # "2026-08-19", "2026/08/19", "2026.08.19"
    p_iso = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")
    for m in p_iso.finditer(text):
        year, month, day = m.groups()
        try:
            dates.add(date(int(year), int(month), int(day)))
        except ValueError:
            pass

    # "19/08/2026", "08/19/2026", "19.08.2026" — ambiguous, try both
    p_dmy = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b")
    for m in p_dmy.finditer(text):
        a, b, year = m.groups()
        # Try day/month/year
        try:
            dates.add(date(int(year), int(b), int(a)))
        except ValueError:
            pass
        # Try month/day/year
        try:
            dates.add(date(int(year), int(a), int(b)))
        except ValueError:
            pass

    return dates


def _date_found_in_context(iso_date: str, context_dates: Set[date]) -> bool:
    """Check if an ISO date string matches any date extracted from context."""
    dt = _parse_iso_date(iso_date)
    if dt is None:
        return False
    return dt in context_dates


# ---------------------------------------------------------------------------
# String matching
# ---------------------------------------------------------------------------

def _fuzzy_found(value: str, source_text: str, threshold: int = 70) -> bool:
    """Check whether *value* appears in *source_text* via fuzzy matching."""
    if not value or not source_text:
        return False
    value_lower = value.lower().strip()
    source_lower = source_text.lower()

    # Exact substring (fast path)
    if value_lower in source_lower:
        return True

    # For short values (city names, acronyms), use partial ratio
    if len(value_lower) < 40:
        return fuzz.partial_ratio(value_lower, source_lower) >= threshold

    # For longer strings, use token set ratio
    return fuzz.token_set_ratio(value_lower, source_lower) >= threshold


# ---------------------------------------------------------------------------
# Date ordering check
# ---------------------------------------------------------------------------

def _check_date_ordering(data: Dict[str, Any]) -> List[str]:
    """
    Check logical ordering of dates:
      submission < notification < camera_ready < start_date ≤ end_date
    Returns list of warnings.
    """
    warnings: List[str] = []

    start = _parse_iso_date(data.get("start_date"))
    end = _parse_iso_date(data.get("end_date"))
    sub = _parse_iso_date(data.get("submission_deadline"))
    notif = _parse_iso_date(data.get("notification_date"))
    cam = _parse_iso_date(data.get("camera_ready_date"))

    if start and end and start > end:
        warnings.append(f"start_date ({start}) > end_date ({end})")

    ordered: List[Tuple[str, Optional[date]]] = [
        ("submission_deadline", sub),
        ("notification_date", notif),
        ("camera_ready_date", cam),
        ("start_date", start),
    ]

    for i in range(len(ordered) - 1):
        name_a, date_a = ordered[i]
        name_b, date_b = ordered[i + 1]
        if date_a and date_b and date_a > date_b:
            warnings.append(f"{name_a} ({date_a}) > {name_b} ({date_b})")

    return warnings


# ---------------------------------------------------------------------------
# Per-category validation
# ---------------------------------------------------------------------------

def validate_other(
    data: Dict[str, Any],
    context: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate OTHER category LLM output against the context.

    Checks each scalar field against the context text. For dates, parses
    all dates from the context first and compares normalized values.
    Nullifies fields not found in context.

    Returns (cleaned_data, warnings).
    """
    if not data:
        return {}, []

    warnings: List[str] = []
    result = dict(data)

    # Parse all dates from context once
    context_dates = _extract_dates_from_text(context)

    # Date fields
    date_fields = [
        "start_date", "end_date",
        "submission_deadline", "notification_date", "camera_ready_date",
    ]
    for field in date_fields:
        value = result.get(field)
        if value is None:
            continue
        if not _date_found_in_context(value, context_dates):
            logger.warning("Nullifying hallucinated date %s = %r", field, value)
            warnings.append(f"[hallucination] {field} = {value}")
            result[field] = None

    # String fields
    string_fields = {
        "full_name": 70,
        "acronym": 80,
        "city": 75,
        "country": 75,
        "publisher": 70,
        "series": 70,
    }
    for field, threshold in string_fields.items():
        value = result.get(field)
        if value is None or value == "":
            continue
        # "Virtual Conference" is a special value we told the LLM to use
        if field in ("city", "country") and value.lower() == "virtual conference":
            continue
        if not _fuzzy_found(value, context, threshold=threshold):
            logger.warning("Nullifying hallucinated field %s = %r", field, value)
            warnings.append(f"[hallucination] {field} = {value}")
            result[field] = None

    # edition_number: just check that the number appears somewhere in context
    edition = result.get("edition_number")
    if edition is not None:
        if str(edition) not in context:
            logger.warning("Nullifying hallucinated edition_number = %r", edition)
            warnings.append(f"[hallucination] edition_number = {edition}")
            result["edition_number"] = None

    # Date ordering check
    date_warnings = _check_date_ordering(result)
    warnings.extend([f"[date_order] {w}" for w in date_warnings])

    return result, warnings


def validate_topics(
    data: Dict[str, Any],
    context: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate TOPICS category LLM output.

    Checks each topic against the context. Removes topics not found.
    """
    if not data:
        return {}, []

    warnings: List[str] = []
    topics = data.get("topics", [])
    if not isinstance(topics, list):
        return {"topics": []}, ["[schema] topics is not a list"]

    filtered = []
    for topic in topics:
        if not isinstance(topic, str) or not topic.strip():
            continue
        if _fuzzy_found(topic, context, threshold=65):
            filtered.append(topic)
        else:
            logger.warning("Removing hallucinated topic: %r", topic)
            warnings.append(f"[hallucination] topic = {topic}")

    return {"topics": filtered}, warnings


def validate_speakers(
    data: Dict[str, Any],
    context: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate SPEAKERS category LLM output.

    Checks each speaker name against the context. Removes speakers
    whose names are not found.
    """
    if not data:
        return {}, []

    warnings: List[str] = []
    speakers = data.get("keynote_speakers", [])
    if not isinstance(speakers, list):
        return {"keynote_speakers": []}, ["[schema] keynote_speakers is not a list"]

    filtered = []
    context_lower = context.lower()
    for speaker in speakers:
        if not isinstance(speaker, dict):
            continue
        name = speaker.get("name", "")
        if not name:
            continue
        if name.lower() in context_lower or _fuzzy_found(name, context, threshold=75):
            filtered.append(speaker)
        else:
            logger.warning("Removing hallucinated speaker: %r", name)
            warnings.append(f"[hallucination] speaker = {name}")

    return {"keynote_speakers": filtered}, warnings


def validate_committee(
    data: Dict[str, Any],
    context: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate COMMITTEE category LLM output.

    Checks each committee member name against the context. Removes
    members whose names are not found.
    """
    if not data:
        return {}, []

    warnings: List[str] = []
    committee = data.get("program_committee", [])
    if not isinstance(committee, list):
        return {"program_committee": []}, ["[schema] program_committee is not a list"]

    filtered = []
    context_lower = context.lower()
    for member in committee:
        if not isinstance(member, dict):
            continue
        name = member.get("name", "")
        if not name:
            continue
        if name.lower() in context_lower or _fuzzy_found(name, context, threshold=80):
            filtered.append(member)
        else:
            logger.warning("Removing hallucinated committee member: %r", name)
            warnings.append(f"[hallucination] committee_member = {name}")

    return {"program_committee": filtered}, warnings


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Import here to avoid circular imports at module level
from .content_selection import Category

_VALIDATORS: Dict[Category, Any] = {
    Category.OTHER: validate_other,
    Category.TOPICS: validate_topics,
    Category.SPEAKERS: validate_speakers,
    Category.COMMITTEE: validate_committee,
}


def validate_category(
    category: Category,
    data: Dict[str, Any],
    context: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate LLM output for a specific category against the context
    that was sent to the LLM.

    Returns (cleaned_data, warnings).
    """
    validator = _VALIDATORS[category]
    return validator(data, context)
