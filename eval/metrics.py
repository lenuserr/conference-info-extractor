"""
Metrics: turn (gold, prediction) pairs into per-field outcomes,
then aggregate across fields, categories and examples.

Field outcomes (scalar fields):

  TP          gold present, pred present, pred matches gold
  FN          gold present, pred absent                 (miss)
  FP_WRONG    gold present, pred present, pred differs  (wrong value)
  FP_HALLUC   gold absent,  pred present                (hallucination)
  TN          gold absent,  pred absent                 (correct abstention)

List fields (topics, keynote_speakers, program_committee) are scored by
greedy set matching and contribute per-example (tp, fp, fn) counts directly.

Precision, recall and F1 are computed as:

  precision = TP / (TP + FP_WRONG + FP_HALLUC)
  recall    = TP / (TP + FN + FP_WRONG)
  F1        = 2 * P * R / (P + R)

TNs are tracked separately and reported as ``abstention_rate`` — they
never inflate precision/recall because "correctly returning null" isn't
the extraction behaviour we want to reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import matchers as M


class Outcome(str, Enum):
    TP = "tp"
    FN = "fn"
    FP_WRONG = "fp_wrong"
    FP_HALLUC = "fp_halluc"
    TN = "tn"


# ---------------------------------------------------------------------------
# Field registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScalarField:
    path: str                         # "dates.start_date"
    section: str                      # "dates"
    key: str                          # "start_date"
    category: str                     # "dates"
    matcher: Callable[[Any, Any], bool]


@dataclass(frozen=True)
class ListField:
    path: str                         # "topics"
    section: str                      # top-level key
    category: str                     # "topics"
    matcher: Callable[[Any, Any], bool]


SCALAR_FIELDS: List[ScalarField] = [
    ScalarField("conference.full_name", "conference", "full_name", "identity", M.match_string),
    ScalarField("conference.acronym", "conference", "acronym", "identity", M.match_acronym),
    ScalarField("conference.edition_number", "conference", "edition_number", "identity", M.match_edition),
    ScalarField("dates.start_date", "dates", "start_date", "dates", M.match_date),
    ScalarField("dates.end_date", "dates", "end_date", "dates", M.match_date),
    ScalarField("venue.city", "venue", "city", "location", M.match_string),
    ScalarField("venue.country", "venue", "country", "location", M.match_country),
    ScalarField("deadlines.submission", "deadlines", "submission", "dates", M.match_date),
    ScalarField("deadlines.notification", "deadlines", "notification", "dates", M.match_date),
    ScalarField("deadlines.camera_ready", "deadlines", "camera_ready", "dates", M.match_date),
    ScalarField("publication.publisher", "publication", "publisher", "publication", M.match_string),
    ScalarField("publication.series", "publication", "series", "publication", M.match_string),
]

LIST_FIELDS: List[ListField] = [
    ListField("topics", "topics", "topics", M.match_topic),
    ListField("keynote_speakers", "keynote_speakers", "speakers", M.match_speaker),
    ListField("program_committee", "program_committee", "committee", M.match_speaker),
]

CATEGORIES = [
    "dates", "location", "identity", "publication",
    "topics", "speakers", "committee",
]


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _get_scalar(obj: Dict[str, Any], fld: ScalarField) -> Any:
    return (obj.get(fld.section) or {}).get(fld.key)


def _is_absent(v: Any) -> bool:
    """Null / empty string / empty list → absent."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, tuple)) and len(v) == 0:
        return True
    return False


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    """Result for a single (field, example) pair."""
    path: str
    category: str
    # Scalar:
    outcome: Optional[Outcome] = None
    # List:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: bool = False  # True when both gold list and pred list were empty
    # For debugging
    gold: Any = None
    pred: Any = None


@dataclass
class ExampleResult:
    url: str
    fields: List[FieldResult] = field(default_factory=list)


def _aliases_for(aliases: Dict[str, Any], path: str) -> List[str]:
    raw = aliases.get(path, [])
    if isinstance(raw, str):
        return [raw]
    return list(raw or [])


def evaluate_one(
    gold: Dict[str, Any],
    prediction: Dict[str, Any],
    aliases: Optional[Dict[str, Any]] = None,
) -> ExampleResult:
    """Compute field-level outcomes for a single example."""
    aliases = aliases or {}
    ground = gold.get("ground_truth") or gold  # tolerate either shape
    pred = prediction.get("data") or prediction  # benchmark entry or bare dict

    url = gold.get("url") or prediction.get("url") or ""
    result = ExampleResult(url=url)

    # Scalar fields
    for fld in SCALAR_FIELDS:
        g = _get_scalar(ground, fld)
        p = _get_scalar(pred, fld)
        g_absent = _is_absent(g)
        p_absent = _is_absent(p)

        if g_absent and p_absent:
            outcome = Outcome.TN
        elif g_absent and not p_absent:
            outcome = Outcome.FP_HALLUC
        elif not g_absent and p_absent:
            outcome = Outcome.FN
        else:
            # Both present — does it match?
            alias_list = _aliases_for(aliases, fld.path)
            if fld.matcher is M.match_string or fld.matcher is M.match_country:
                ok = fld.matcher(g, p, aliases=alias_list) if alias_list else fld.matcher(g, p)
            else:
                ok = fld.matcher(g, p)
            outcome = Outcome.TP if ok else Outcome.FP_WRONG

        result.fields.append(
            FieldResult(
                path=fld.path, category=fld.category,
                outcome=outcome, gold=g, pred=p,
            )
        )

    # List fields
    for lfld in LIST_FIELDS:
        g_list = ground.get(lfld.section) or []
        p_list = pred.get(lfld.section) or []
        g_absent = len(g_list) == 0
        p_absent = len(p_list) == 0

        if g_absent and p_absent:
            fr = FieldResult(
                path=lfld.path, category=lfld.category,
                tn=True, gold=g_list, pred=p_list,
            )
        else:
            tp, fp, fn = M.greedy_list_match(g_list, p_list, lfld.matcher)
            fr = FieldResult(
                path=lfld.path, category=lfld.category,
                tp=tp, fp=fp, fn=fn,
                gold=g_list, pred=p_list,
            )
        result.fields.append(fr)

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class Counts:
    tp: int = 0
    fn: int = 0
    fp_wrong: int = 0
    fp_halluc: int = 0
    tn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fn += other.fn
        self.fp_wrong += other.fp_wrong
        self.fp_halluc += other.fp_halluc
        self.tn += other.tn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp_wrong + self.fp_halluc
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn + self.fp_wrong
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def support(self) -> int:
        """Number of (field, example) pairs where gold was present."""
        return self.tp + self.fn + self.fp_wrong

    @property
    def hallucination_rate(self) -> float:
        """Share of gold-absent cases where we hallucinated a value."""
        denom = self.fp_halluc + self.tn
        return self.fp_halluc / denom if denom else 0.0


def _counts_from_field(fr: FieldResult) -> Counts:
    c = Counts()
    if fr.outcome is not None:
        # Scalar field
        if fr.outcome == Outcome.TP:
            c.tp = 1
        elif fr.outcome == Outcome.FN:
            c.fn = 1
        elif fr.outcome == Outcome.FP_WRONG:
            c.fp_wrong = 1
        elif fr.outcome == Outcome.FP_HALLUC:
            c.fp_halluc = 1
        elif fr.outcome == Outcome.TN:
            c.tn = 1
    else:
        # List field
        c.tp = fr.tp
        c.fp_wrong = fr.fp  # treat "extra predicted item" as wrong prediction
        c.fn = fr.fn
        if fr.tn:
            c.tn = 1
    return c


@dataclass
class ModelMetrics:
    model: str
    backend: str = ""
    vllm_extra_args: List[str] = field(default_factory=list)
    n_examples: int = 0             # examples actually scored
    overall: Counts = field(default_factory=Counts)
    by_category: Dict[str, Counts] = field(default_factory=dict)
    by_field: Dict[str, Counts] = field(default_factory=dict)
    # Secondary metrics
    avg_latency_sec: float = 0.0
    avg_attempts: float = 0.0
    errors: int = 0                 # runs that raised an exception
    fetch_failures: int = 0         # runs where the scraper got 0 pages —
                                    # the model had no input, so they're
                                    # excluded from precision/recall

    @property
    def macro_f1_categories(self) -> float:
        """Macro-F1 across categories (each category weighted equally)."""
        cats = [c for c in CATEGORIES if c in self.by_category]
        if not cats:
            return 0.0
        return sum(self.by_category[c].f1 for c in cats) / len(cats)


# Warning string emitted by extractor.pipeline when the scraper returns
# zero pages. We detect fetch failures by looking for it in the entry's
# warnings list rather than by status, so that the benchmark.py format
# stays unchanged and old result files are handled the same way.
_FETCH_FAILED_WARNING = "Could not fetch any pages"


def _is_fetch_failure(entry: Dict[str, Any]) -> bool:
    """True if the entry's warnings indicate the scraper got nothing."""
    warnings = entry.get("warnings") or []
    return any(_FETCH_FAILED_WARNING in str(w) for w in warnings)


def aggregate(
    model: str,
    example_results: List[ExampleResult],
    entries: Optional[List[Dict[str, Any]]] = None,
) -> ModelMetrics:
    """
    Collapse per-example field outcomes into overall / per-category / per-field
    counts.  ``entries`` are the original benchmark result dicts (used for
    latency / attempts / backend metadata); pass None to skip secondary stats.

    ``entries`` should be the full set of benchmark runs that had a
    matching gold file (including failed ones), so ``errors`` and
    ``fetch_failures`` are counted accurately. ``example_results`` should
    only cover runs that were actually scored (i.e. excluding fetch
    failures and crashes), so the P/R/F1 numbers aren't polluted by runs
    where the model had no input.
    """
    metrics = ModelMetrics(model=model, n_examples=len(example_results))

    for ex in example_results:
        for fr in ex.fields:
            c = _counts_from_field(fr)
            metrics.overall.add(c)
            metrics.by_category.setdefault(fr.category, Counts()).add(c)
            metrics.by_field.setdefault(fr.path, Counts()).add(c)

    if entries:
        ok_entries = [e for e in entries if e.get("status") == "ok"]
        metrics.errors = len(entries) - len(ok_entries)
        metrics.fetch_failures = sum(1 for e in entries if _is_fetch_failure(e))
        if ok_entries:
            metrics.avg_latency_sec = sum(e.get("elapsed_sec", 0) for e in ok_entries) / len(ok_entries)
            metrics.avg_attempts = sum(e.get("attempts", 0) for e in ok_entries) / len(ok_entries)
        # Backend / extra args (take from first entry)
        metrics.backend = entries[0].get("backend", "")
        metrics.vllm_extra_args = entries[0].get("vllm_extra_args") or []

    return metrics
