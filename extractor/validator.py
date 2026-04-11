"""
Validation layer for extracted conference metadata.

1. JSON Schema validation
2. Date logic checks (ordering)
3. Source verification (fuzzy match against original HTML text)
4. Confidence scoring per field
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from thefuzz import fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON Schema (simplified inline — avoids extra dependency complexity)
# ---------------------------------------------------------------------------

CONFERENCE_SCHEMA = {
    "type": "object",
    "required": ["conference", "dates", "venue", "deadlines", "topics", "keynote_speakers", "publication"],
    "properties": {
        "conference": {
            "type": "object",
            "required": ["full_name", "acronym", "url", "edition_number"],
            "properties": {
                "full_name": {"type": "string"},
                "acronym": {"type": "string"},
                "url": {"type": "string"},
                "edition_number": {"type": ["integer", "null"]},
            },
        },
        "dates": {
            "type": "object",
            "properties": {
                "start_date": {"type": ["string", "null"]},
                "end_date": {"type": ["string", "null"]},
            },
        },
        "venue": {
            "type": "object",
            "properties": {
                "city": {"type": ["string", "null"]},
                "country": {"type": ["string", "null"]},
            },
        },
        "deadlines": {
            "type": "object",
            "properties": {
                "submission": {"type": ["string", "null"]},
                "notification": {"type": ["string", "null"]},
                "camera_ready": {"type": ["string", "null"]},
            },
        },
        "topics": {"type": "array", "items": {"type": "string"}},
        "keynote_speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "affiliation": {"type": ["string", "null"]},
                    "country": {"type": ["string", "null"]},
                },
            },
        },
        "publication": {
            "type": "object",
            "properties": {
                "publisher": {"type": ["string", "null"]},
                "series": {"type": ["string", "null"]},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

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


def _parse_date(s: Optional[str]) -> Optional[date]:
    if s and _DATE_RE.match(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            pass
    return None


def _ordinal_suffix(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _generate_date_variants(
    iso_date: str,
) -> Tuple[List[str], List[str], str]:
    """
    Generate surface-form variants of an ISO date (``YYYY-MM-DD``).

    Returns ``(full_variants, partial_variants, year)``:
      - ``full_variants``: lowercased complete date strings (year + day + month)
      - ``partial_variants``: lowercased day+month fragments (e.g. ``"august 19"``),
        used to recognise date-range text like ``"19-21 August 2026"`` where
        neither individual date appears as a full substring
      - ``year``: the year as a 4-digit string, or ``""`` if the input is
        not a valid ISO date
    """
    dt = _parse_date(iso_date)
    if dt is None:
        return [], [], ""

    year = f"{dt.year:04d}"
    month_full, month_abbr = _MONTH_NAMES[dt.month - 1]

    day_int = dt.day
    month_int = dt.month
    day_num = str(day_int)               # "19", "5"
    day_pad = f"{day_int:02d}"           # "19", "05"
    month_num = str(month_int)           # "8"
    month_pad = f"{month_int:02d}"       # "08"
    day_ord = f"{day_int}{_ordinal_suffix(day_int)}"  # "19th", "21st"

    day_forms = {day_num, day_pad, day_ord}

    full: List[str] = []

    # Month-name forms: "August 19, 2026" / "19 August 2026" / "19th of August, 2026"
    for m_name in (month_full, month_abbr):
        for d in day_forms:
            full.append(f"{m_name} {d}, {year}")
            full.append(f"{m_name} {d} {year}")
            full.append(f"{d} {m_name} {year}")
            full.append(f"{d} {m_name}, {year}")
            full.append(f"{d} of {m_name} {year}")
            full.append(f"{d} of {m_name}, {year}")

    # Purely numeric forms in the common separators: "-", "/", "."
    for sep in ("-", "/", "."):
        full.append(f"{year}{sep}{month_pad}{sep}{day_pad}")
        full.append(f"{day_pad}{sep}{month_pad}{sep}{year}")
        full.append(f"{month_pad}{sep}{day_pad}{sep}{year}")
        # Non-zero-padded variants (e.g. "8/19/2026")
        if month_num != month_pad or day_num != day_pad:
            full.append(f"{day_num}{sep}{month_num}{sep}{year}")
            full.append(f"{month_num}{sep}{day_num}{sep}{year}")

    # Day+month fragments — used as a fallback for date ranges
    partial: List[str] = []
    for m_name in (month_full, month_abbr):
        for d in day_forms:
            partial.append(f"{m_name} {d}")
            partial.append(f"{d} {m_name}")
            partial.append(f"{d} of {m_name}")

    return (
        [v.lower() for v in full],
        [p.lower() for p in partial],
        year,
    )


_MONTH_LOOKUP: Dict[str, int] = {
    name.lower(): i + 1
    for i, (full_name, abbr) in enumerate(_MONTH_NAMES)
    for name in (full_name, abbr)
}

_MONTH_ALT = "|".join(sorted(_MONTH_LOOKUP.keys(), key=len, reverse=True))
# Ranges like "August 19-21, 2026" / "August 19 – 21 2026"
_RANGE_MONTH_FIRST_RE = re.compile(
    rf"\b({_MONTH_ALT})\s+(\d{{1,2}})\s*[-–—]\s*(\d{{1,2}})\s*,?\s*(\d{{4}})\b",
    re.IGNORECASE,
)
# Ranges like "19-21 August 2026" / "19 – 21 Aug, 2026"
_RANGE_MONTH_LAST_RE = re.compile(
    rf"\b(\d{{1,2}})\s*[-–—]\s*(\d{{1,2}})\s+({_MONTH_ALT})\s*,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)


def _date_in_text_range(iso_date: str, source_text: str) -> bool:
    """
    True if *iso_date* is contained in a month-name day-range appearing in
    *source_text*, e.g. ``"August 19-21, 2026"`` or ``"19-21 August 2026"``.
    """
    dt = _parse_date(iso_date)
    if dt is None:
        return False

    for m in _RANGE_MONTH_FIRST_RE.finditer(source_text):
        month_name, d_start, d_end, year_str = m.groups()
        month_num = _MONTH_LOOKUP.get(month_name.lower())
        try:
            if (
                month_num == dt.month
                and int(year_str) == dt.year
                and int(d_start) <= dt.day <= int(d_end)
            ):
                return True
        except ValueError:
            continue

    for m in _RANGE_MONTH_LAST_RE.finditer(source_text):
        d_start, d_end, month_name, year_str = m.groups()
        month_num = _MONTH_LOOKUP.get(month_name.lower())
        try:
            if (
                month_num == dt.month
                and int(year_str) == dt.year
                and int(d_start) <= dt.day <= int(d_end)
            ):
                return True
        except ValueError:
            continue

    return False


def _date_found_in_source(iso_date: str, source_text: str) -> bool:
    """
    True if *iso_date* (``YYYY-MM-DD``) appears in *source_text* in any
    common surface form.

    Strategy:
      1. Generate plausible complete representations (``August 19, 2026``,
         ``19/08/2026``, ``2026-08-19``, ...) and look for a substring match.
      2. Look for a month-name day-range that brackets the date
         (``August 19-21, 2026`` / ``19-21 August 2026``).
      3. Fallback: year present AND a ``day + month`` fragment anywhere in
         the text.
    """
    if not iso_date or not source_text:
        return False
    full, partial, year = _generate_date_variants(iso_date)
    if not year:
        return False

    src = source_text.lower()

    for v in full:
        if v in src:
            return True

    if _date_in_text_range(iso_date, source_text):
        return True

    if year in src:
        for p in partial:
            if p in src:
                return True

    return False


def _fuzzy_found_in_source(value: str, source_text: str, threshold: int = 70) -> bool:
    """Check whether *value* can be found in *source_text* via fuzzy matching."""
    if not value or not source_text:
        return False
    value_lower = value.lower().strip()
    source_lower = source_text.lower()

    # Exact substring first (fast path)
    if value_lower in source_lower:
        return True

    # For short values (city names, dates), use partial ratio
    if len(value_lower) < 40:
        return fuzz.partial_ratio(value_lower, source_lower) >= threshold

    # For longer strings, use token set ratio
    return fuzz.token_set_ratio(value_lower, source_lower) >= threshold


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_schema(data: Dict[str, Any]) -> List[str]:
    """
    Validate against the JSON Schema. Returns list of error messages (empty = OK).
    """
    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(CONFERENCE_SCHEMA)
        return [e.message for e in validator.iter_errors(data)]
    except ImportError:
        logger.warning("jsonschema not installed — skipping schema validation")
        return []


# ---------------------------------------------------------------------------
# Date logic checks
# ---------------------------------------------------------------------------

def validate_dates(data: Dict[str, Any]) -> List[str]:
    """
    Check logical ordering of dates:
      submission < notification < camera_ready < start_date < end_date
    Returns list of warnings.
    """
    warnings: List[str] = []

    start = _parse_date(data.get("dates", {}).get("start_date"))
    end = _parse_date(data.get("dates", {}).get("end_date"))
    sub = _parse_date(data.get("deadlines", {}).get("submission"))
    notif = _parse_date(data.get("deadlines", {}).get("notification"))
    cam = _parse_date(data.get("deadlines", {}).get("camera_ready"))

    if start and end and start > end:
        warnings.append(f"start_date ({start}) > end_date ({end})")

    ordered: List[Tuple[str, Optional[date]]] = [
        ("submission", sub),
        ("notification", notif),
        ("camera_ready", cam),
        ("start_date", start),
    ]

    for i in range(len(ordered) - 1):
        name_a, date_a = ordered[i]
        name_b, date_b = ordered[i + 1]
        if date_a and date_b and date_a > date_b:
            warnings.append(f"{name_a} ({date_a}) > {name_b} ({date_b})")

    return warnings


# ---------------------------------------------------------------------------
# Source verification + confidence scoring
# ---------------------------------------------------------------------------

def verify_against_source(
    data: Dict[str, Any],
    source_text: str,
) -> Dict[str, str]:
    """
    For each key field, check if the extracted value appears in the original
    source text. Returns a dict mapping field paths to confidence levels:
    "high", "medium", "low".

    - high: exact substring found
    - medium: fuzzy match found
    - low: not found at all (possible hallucination)
    """
    confidence: Dict[str, str] = {}

    # Fields to check: (json_path, value)
    checks: List[Tuple[str, Optional[str]]] = [
        ("conference.full_name", data.get("conference", {}).get("full_name")),
        ("conference.acronym", data.get("conference", {}).get("acronym")),
        ("dates.start_date", data.get("dates", {}).get("start_date")),
        ("dates.end_date", data.get("dates", {}).get("end_date")),
        ("venue.city", data.get("venue", {}).get("city")),
        ("venue.country", data.get("venue", {}).get("country")),
        ("deadlines.submission", data.get("deadlines", {}).get("submission")),
        ("deadlines.notification", data.get("deadlines", {}).get("notification")),
        ("deadlines.camera_ready", data.get("deadlines", {}).get("camera_ready")),
        ("publication.publisher", data.get("publication", {}).get("publisher")),
    ]

    # Date fields store normalized ISO dates ("YYYY-MM-DD") but the source
    # text rarely uses that exact format, so they need format-aware matching.
    date_paths = {
        "dates.start_date",
        "dates.end_date",
        "deadlines.submission",
        "deadlines.notification",
        "deadlines.camera_ready",
    }

    source_lower = source_text.lower() if source_text else ""

    for path, value in checks:
        if value is None:
            confidence[path] = "high"  # null is a valid "I don't know"
            continue

        if path in date_paths:
            if _date_found_in_source(str(value).strip(), source_text):
                confidence[path] = "high"
            else:
                confidence[path] = "low"
            continue

        val_lower = str(value).lower().strip()
        if val_lower in source_lower:
            confidence[path] = "high"
        elif _fuzzy_found_in_source(str(value), source_text, threshold=70):
            confidence[path] = "medium"
        else:
            confidence[path] = "low"

    # Keynote speakers: check each name
    for i, speaker in enumerate(data.get("keynote_speakers", [])):
        name = speaker.get("name", "")
        path = f"keynote_speakers[{i}].name"
        if name.lower() in source_lower:
            confidence[path] = "high"
        elif _fuzzy_found_in_source(name, source_text, threshold=75):
            confidence[path] = "medium"
        else:
            confidence[path] = "low"

    return confidence


def nullify_low_confidence(
    data: Dict[str, Any],
    confidence: Dict[str, str],
) -> Dict[str, Any]:
    """
    Set fields with 'low' confidence to null (likely hallucinated).
    Mutates and returns data.
    """
    import copy
    data = copy.deepcopy(data)

    mapping = {
        "conference.full_name": ("conference", "full_name"),
        "conference.acronym": ("conference", "acronym"),
        "dates.start_date": ("dates", "start_date"),
        "dates.end_date": ("dates", "end_date"),
        "venue.city": ("venue", "city"),
        "venue.country": ("venue", "country"),
        "deadlines.submission": ("deadlines", "submission"),
        "deadlines.notification": ("deadlines", "notification"),
        "deadlines.camera_ready": ("deadlines", "camera_ready"),
        "publication.publisher": ("publication", "publisher"),
        "publication.series": ("publication", "series"),
    }

    for field_path, conf_level in confidence.items():
        if conf_level == "low" and field_path in mapping:
            section, key = mapping[field_path]
            old_val = data.get(section, {}).get(key)
            if old_val is not None:
                logger.warning("Nullifying hallucinated field %s = %r", field_path, old_val)
                data[section][key] = None

    # Remove keynote speakers whose name is low-confidence
    speakers = data.get("keynote_speakers", [])
    filtered = []
    for i, sp in enumerate(speakers):
        path = f"keynote_speakers[{i}].name"
        if confidence.get(path) != "low":
            filtered.append(sp)
        else:
            logger.warning("Removing hallucinated speaker: %s", sp.get("name"))
    data["keynote_speakers"] = filtered

    return data


# ---------------------------------------------------------------------------
# Aggregate validation
# ---------------------------------------------------------------------------

def full_validate(
    data: Dict[str, Any],
    source_text: str,
) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """
    Run all validations. Returns:
      (cleaned_data, confidence_map, list_of_warnings)
    """
    warnings: List[str] = []

    # 1. Schema
    schema_errors = validate_schema(data)
    warnings.extend([f"[schema] {e}" for e in schema_errors])

    # 2. Date logic
    date_warnings = validate_dates(data)
    warnings.extend([f"[dates] {w}" for w in date_warnings])

    # 3. Source verification
    confidence = verify_against_source(data, source_text)
    low_fields = [k for k, v in confidence.items() if v == "low"]
    if low_fields:
        warnings.extend([f"[source] low confidence: {f}" for f in low_fields])

    # 4. Nullify hallucinated fields
    data = nullify_low_confidence(data, confidence)

    return data, confidence, warnings
