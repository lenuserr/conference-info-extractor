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


def _parse_date(s: Optional[str]) -> Optional[date]:
    if s and _DATE_RE.match(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            pass
    return None


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

    source_lower = source_text.lower() if source_text else ""

    for path, value in checks:
        if value is None:
            confidence[path] = "high"  # null is a valid "I don't know"
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
