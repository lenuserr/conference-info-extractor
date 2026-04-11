"""
Field-level matchers: decide whether a predicted value equals a gold value.

Each matcher returns True/False. Normalization and fuzziness are type-specific:

  - dates        : normalize to YYYY-MM-DD, exact equality
  - acronym      : case-insensitive exact
  - edition      : integer equality
  - generic str  : normalize (lowercase, strip punct/noise), substring or
                   token_set_ratio >= STRING_THRESHOLD
  - country      : ISO alpha-2 via pycountry if available, otherwise a
                   small hardcoded alias map + generic string fallback
  - topic        : normalize + token_sort_ratio >= TOPIC_THRESHOLD
  - speaker name : strip diacritics/titles, token_set_ratio >= NAME_THRESHOLD

For list fields (topics, speakers, program committee) use ``greedy_list_match``
which returns the per-example (tp, fp, fn) tuple via greedy bipartite matching.
Both speakers and committee members match via ``match_speaker`` (name only).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from thefuzz import fuzz

# ---------------------------------------------------------------------------
# Thresholds (tuned conservatively — raise to be stricter)
# ---------------------------------------------------------------------------
STRING_THRESHOLD = 85
TOPIC_THRESHOLD = 78
NAME_THRESHOLD = 88


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Noise phrases commonly seen in conference names, stripped before matching.
_NAME_NOISE = [
    "the ",
    "annual ",
    "international ",
    "conference on ",
    "conference of ",
    "proceedings of ",
    "ieee ",
    "acm ",
    "symposium on ",
    "workshop on ",
]

_TITLE_RE = re.compile(
    r"\b(prof|professor|dr|mr|ms|mrs|sir|phd|ph\.d)\.?\b",
    re.IGNORECASE,
)


def _strip_diacritics(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize_string(s: Optional[str]) -> str:
    """Lowercase, strip diacritics, punctuation, and common noise phrases."""
    if s is None:
        return ""
    s = _strip_diacritics(str(s)).lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    # Strip leading noise phrases repeatedly
    changed = True
    while changed:
        changed = False
        for noise in _NAME_NOISE:
            if s.startswith(noise):
                s = s[len(noise):]
                changed = True
    return s.strip()


def normalize_name(s: Optional[str]) -> str:
    """Normalize a person name (strip titles, diacritics, punctuation)."""
    if s is None:
        return ""
    s = _strip_diacritics(str(s)).lower()
    s = _TITLE_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_date(s: Any) -> Optional[str]:
    """Return ``YYYY-MM-DD`` if parseable, else None."""
    if s is None:
        return None
    s = str(s).strip()
    if _DATE_RE.match(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            return None
    # Last-ditch: try a couple of forgiving formats
    for fmt in ("%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Country canonicalization
# ---------------------------------------------------------------------------

# Minimal alias map used when pycountry is not installed.
_COUNTRY_ALIASES = {
    "usa": "us",
    "u.s.a": "us",
    "u.s.a.": "us",
    "united states": "us",
    "united states of america": "us",
    "america": "us",
    "uk": "gb",
    "u.k.": "gb",
    "united kingdom": "gb",
    "great britain": "gb",
    "england": "gb",
    "south korea": "kr",
    "korea": "kr",
    "republic of korea": "kr",
    "russia": "ru",
    "russian federation": "ru",
    "czech republic": "cz",
    "czechia": "cz",
    "uae": "ae",
    "united arab emirates": "ae",
}

try:
    import pycountry  # type: ignore

    def _country_to_iso(s: str) -> Optional[str]:
        s = s.strip()
        if not s:
            return None
        # Try direct lookup, then name search
        try:
            match = pycountry.countries.lookup(s)
            return match.alpha_2.lower()
        except LookupError:
            pass
        # Fall back to alias map
        return _COUNTRY_ALIASES.get(s.lower())

except ImportError:  # pragma: no cover
    def _country_to_iso(s: str) -> Optional[str]:
        s = s.strip().lower()
        if not s:
            return None
        return _COUNTRY_ALIASES.get(s, s[:2] if len(s) == 2 else None)


# ---------------------------------------------------------------------------
# Scalar matchers
# ---------------------------------------------------------------------------

def match_date(gold: Any, pred: Any) -> bool:
    g = normalize_date(gold)
    p = normalize_date(pred)
    if g is None or p is None:
        return False
    return g == p


def match_acronym(gold: Any, pred: Any) -> bool:
    if gold is None or pred is None:
        return False
    return str(gold).strip().lower() == str(pred).strip().lower()


def match_edition(gold: Any, pred: Any) -> bool:
    if gold is None or pred is None:
        return False
    try:
        return int(gold) == int(pred)
    except (TypeError, ValueError):
        return False


def match_string(
    gold: Any,
    pred: Any,
    aliases: Optional[Sequence[str]] = None,
    threshold: int = STRING_THRESHOLD,
) -> bool:
    """
    Generic fuzzy string match.

    ``aliases`` is a list of alternative acceptable surface forms for the
    gold value (e.g. ``["Vancouver, BC"]`` for gold ``"Vancouver"``).
    """
    if gold is None or pred is None:
        return False
    p_norm = normalize_string(pred)
    if not p_norm:
        return False

    candidates = [gold] + list(aliases or [])
    for cand in candidates:
        g_norm = normalize_string(cand)
        if not g_norm:
            continue
        if g_norm == p_norm:
            return True
        if g_norm in p_norm or p_norm in g_norm:
            return True
        if fuzz.token_set_ratio(g_norm, p_norm) >= threshold:
            return True
    return False


def match_country(
    gold: Any,
    pred: Any,
    aliases: Optional[Sequence[str]] = None,
) -> bool:
    if gold is None or pred is None:
        return False
    g_iso = _country_to_iso(str(gold))
    p_iso = _country_to_iso(str(pred))
    if g_iso and p_iso and g_iso == p_iso:
        return True
    # Fall back to string match (covers obscure regions / pycountry misses)
    return match_string(gold, pred, aliases=aliases)


# ---------------------------------------------------------------------------
# List matchers
# ---------------------------------------------------------------------------

def match_topic(gold: Any, pred: Any) -> bool:
    g = normalize_string(gold)
    p = normalize_string(pred)
    if not g or not p:
        return False
    if g == p or g in p or p in g:
        return True
    return fuzz.token_sort_ratio(g, p) >= TOPIC_THRESHOLD


def match_speaker(gold: Any, pred: Any) -> bool:
    """Match by name only; affiliation is not required."""
    g_name = gold.get("name") if isinstance(gold, dict) else gold
    p_name = pred.get("name") if isinstance(pred, dict) else pred
    g = normalize_name(g_name)
    p = normalize_name(p_name)
    if not g or not p:
        return False
    if g == p:
        return True
    return fuzz.token_set_ratio(g, p) >= NAME_THRESHOLD


def greedy_list_match(
    gold_items: Iterable[Any],
    pred_items: Iterable[Any],
    matcher: Callable[[Any, Any], bool],
) -> Tuple[int, int, int]:
    """
    Greedy bipartite match between gold and predicted items.

    Returns (tp, fp, fn):
      tp = gold items that found a matching prediction
      fp = predicted items that matched no gold
      fn = gold items with no matching prediction
    """
    gold_list: List[Any] = list(gold_items or [])
    pred_list: List[Any] = list(pred_items or [])
    used_pred: List[bool] = [False] * len(pred_list)

    tp = 0
    for g in gold_list:
        for i, p in enumerate(pred_list):
            if used_pred[i]:
                continue
            if matcher(g, p):
                used_pred[i] = True
                tp += 1
                break

    fn = len(gold_list) - tp
    fp = used_pred.count(False)
    return tp, fp, fn
