#!/usr/bin/env python3
"""
Evaluate model results against a golden dataset.

Compares extracted data on 5 dimensions:
  - dates:             exact match of start_date, end_date
  - venue:             fuzzy match of city, country
  - topics:            precision / recall / F1 over topic lists
  - keynote_speakers:  match by name, then check affiliation & country
  - program_committee: match by name, then check affiliation, country & role

Usage:
    python evaluate.py --golden results/claude/ --pred results/mistral_latest/
    python evaluate.py --golden results/claude/ --pred results/qwen3_4b/ --fuzzy-threshold 75
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

from thefuzz import fuzz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results(directory: str) -> Dict[str, Dict[str, Any]]:
    """Load all JSON result files from a directory. Returns {filename: data}."""
    results = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8") as f:
            entry = json.load(f)
        results[name] = entry.get("data", {})
    return results


def _fuzzy_match(a: str, b: str, threshold: int = 75) -> bool:
    """Check if two strings are a fuzzy match."""
    if not a or not b:
        return False
    a_lower = a.strip().lower()
    b_lower = b.strip().lower()
    if a_lower == b_lower:
        return True
    if len(a_lower) < 30 or len(b_lower) < 30:
        return fuzz.partial_ratio(a_lower, b_lower) >= threshold
    return fuzz.token_set_ratio(a_lower, b_lower) >= threshold


# ---------------------------------------------------------------------------
# Per-category evaluation
# ---------------------------------------------------------------------------

def eval_dates(golden: Dict, pred: Dict) -> Dict[str, Any]:
    """Evaluate dates: exact match for start_date and end_date."""
    g = golden.get("dates", {})
    p = pred.get("dates", {})

    results = {}
    for field in ("start_date", "end_date"):
        gv = g.get(field)
        pv = p.get(field)

        if gv is None and pv is None:
            results[field] = "both_null"
        elif gv is None:
            results[field] = "extra"      # model produced value, golden has none
        elif pv is None:
            results[field] = "missing"    # model missed it
        elif gv == pv:
            results[field] = "correct"
        else:
            results[field] = "wrong"

    correct = sum(1 for v in results.values() if v in ("correct", "both_null"))
    total = len(results)
    return {
        "fields": results,
        "accuracy": correct / total if total > 0 else 1.0,
    }


def eval_venue(golden: Dict, pred: Dict, threshold: int = 75) -> Dict[str, Any]:
    """Evaluate venue: fuzzy match for city and country."""
    g = golden.get("venue", {})
    p = pred.get("venue", {})

    results = {}
    for field in ("city", "country"):
        gv = g.get(field)
        pv = p.get(field)

        if gv is None and pv is None:
            results[field] = "both_null"
        elif gv is None:
            results[field] = "extra"
        elif pv is None:
            results[field] = "missing"
        elif _fuzzy_match(gv, pv, threshold):
            results[field] = "correct"
        else:
            results[field] = "wrong"

    correct = sum(1 for v in results.values() if v in ("correct", "both_null"))
    total = len(results)
    return {
        "fields": results,
        "accuracy": correct / total if total > 0 else 1.0,
    }


def eval_topics(
    golden: Dict, pred: Dict, threshold: int = 70,
) -> Dict[str, Any]:
    """Evaluate topics: precision / recall / F1 with fuzzy matching."""
    g_topics: List[str] = golden.get("topics", [])
    p_topics: List[str] = pred.get("topics", [])

    if not g_topics and not p_topics:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "golden_count": 0, "pred_count": 0, "matched": 0}

    if not g_topics:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0,
                "golden_count": 0, "pred_count": len(p_topics), "matched": 0}

    if not p_topics:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0,
                "golden_count": len(g_topics), "pred_count": 0, "matched": 0}

    # Match predicted topics to golden (greedy, each golden used at most once)
    g_used = [False] * len(g_topics)
    matched = 0

    for pt in p_topics:
        best_idx = -1
        best_score = 0
        for i, gt in enumerate(g_topics):
            if g_used[i]:
                continue
            score = fuzz.token_set_ratio(pt.lower(), gt.lower())
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0 and best_score >= threshold:
            g_used[best_idx] = True
            matched += 1

    precision = matched / len(p_topics) if p_topics else 0.0
    recall = matched / len(g_topics) if g_topics else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "golden_count": len(g_topics),
        "pred_count": len(p_topics),
        "matched": matched,
    }


def _match_persons(
    golden_list: List[Dict], pred_list: List[Dict],
    name_threshold: int = 75,
    field_threshold: int = 75,
    extra_fields: Tuple[str, ...] = ("affiliation", "country"),
) -> Dict[str, Any]:
    """
    Match two lists of person dicts by name (fuzzy).
    For matched pairs, evaluate extra fields.

    Returns precision/recall/F1 for name matching,
    plus per-field accuracy for matched pairs.
    """
    if not golden_list and not pred_list:
        result = {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                  "golden_count": 0, "pred_count": 0, "matched": 0}
        for fld in extra_fields:
            result[f"{fld}_accuracy"] = 1.0
        return result

    if not golden_list:
        result = {"precision": 0.0, "recall": 1.0, "f1": 0.0,
                  "golden_count": 0, "pred_count": len(pred_list), "matched": 0}
        for fld in extra_fields:
            result[f"{fld}_accuracy"] = 0.0
        return result

    if not pred_list:
        result = {"precision": 1.0, "recall": 0.0, "f1": 0.0,
                  "golden_count": len(golden_list), "pred_count": 0, "matched": 0}
        for fld in extra_fields:
            result[f"{fld}_accuracy"] = 0.0
        return result

    # Greedy match by name
    g_used = [False] * len(golden_list)
    matches: List[Tuple[Dict, Dict]] = []  # (golden, pred)

    for pp in pred_list:
        pname = (pp.get("name") or "").strip()
        if not pname:
            continue
        best_idx = -1
        best_score = 0
        for i, gp in enumerate(golden_list):
            if g_used[i]:
                continue
            gname = (gp.get("name") or "").strip()
            if not gname:
                continue
            score = fuzz.token_set_ratio(pname.lower(), gname.lower())
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0 and best_score >= name_threshold:
            g_used[best_idx] = True
            matches.append((golden_list[best_idx], pp))

    matched = len(matches)
    precision = matched / len(pred_list) if pred_list else 0.0
    recall = matched / len(golden_list) if golden_list else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Per-field accuracy for matched pairs
    field_acc = {}
    for fld in extra_fields:
        if not matches:
            field_acc[f"{fld}_accuracy"] = 0.0
            continue
        correct = 0
        for gp, pp in matches:
            gv = (gp.get(fld) or "").strip()
            pv = (pp.get(fld) or "").strip()
            if not gv and not pv:
                correct += 1
            elif gv and pv and _fuzzy_match(gv, pv, field_threshold):
                correct += 1
        field_acc[f"{fld}_accuracy"] = round(correct / len(matches), 4)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "golden_count": len(golden_list),
        "pred_count": len(pred_list),
        "matched": matched,
        **field_acc,
    }


def eval_speakers(golden: Dict, pred: Dict, threshold: int = 75) -> Dict[str, Any]:
    """Evaluate keynote speakers: match by name, check affiliation & country."""
    return _match_persons(
        golden.get("keynote_speakers", []),
        pred.get("keynote_speakers", []),
        name_threshold=threshold,
        field_threshold=threshold,
        extra_fields=("affiliation", "country"),
    )


def eval_committee(golden: Dict, pred: Dict, threshold: int = 75) -> Dict[str, Any]:
    """Evaluate program committee: match by name, check affiliation, country & role."""
    return _match_persons(
        golden.get("program_committee", []),
        pred.get("program_committee", []),
        name_threshold=threshold,
        field_threshold=threshold,
        extra_fields=("affiliation", "country", "role"),
    )


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_site(
    golden: Dict[str, Any],
    pred: Dict[str, Any],
    fuzzy_threshold: int = 75,
) -> Dict[str, Any]:
    """Evaluate all 5 categories for a single site."""
    return {
        "dates": eval_dates(golden, pred),
        "venue": eval_venue(golden, pred, threshold=fuzzy_threshold),
        "topics": eval_topics(golden, pred, threshold=fuzzy_threshold - 5),
        "keynote_speakers": eval_speakers(golden, pred, threshold=fuzzy_threshold),
        "program_committee": eval_committee(golden, pred, threshold=fuzzy_threshold),
    }


def aggregate(site_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-site results into overall metrics."""
    n = len(site_results)
    if n == 0:
        return {}

    # dates & venue: average accuracy
    dates_acc = sum(r["dates"]["accuracy"] for r in site_results.values()) / n
    venue_acc = sum(r["venue"]["accuracy"] for r in site_results.values()) / n

    # topics: macro-average P/R/F1
    topics_p = sum(r["topics"]["precision"] for r in site_results.values()) / n
    topics_r = sum(r["topics"]["recall"] for r in site_results.values()) / n
    topics_f1 = sum(r["topics"]["f1"] for r in site_results.values()) / n

    # speakers: macro-average P/R/F1 + field accuracies
    sp_p = sum(r["keynote_speakers"]["precision"] for r in site_results.values()) / n
    sp_r = sum(r["keynote_speakers"]["recall"] for r in site_results.values()) / n
    sp_f1 = sum(r["keynote_speakers"]["f1"] for r in site_results.values()) / n
    sp_aff = sum(r["keynote_speakers"]["affiliation_accuracy"] for r in site_results.values()) / n
    sp_country = sum(r["keynote_speakers"]["country_accuracy"] for r in site_results.values()) / n

    # committee: macro-average P/R/F1 + field accuracies
    cm_p = sum(r["program_committee"]["precision"] for r in site_results.values()) / n
    cm_r = sum(r["program_committee"]["recall"] for r in site_results.values()) / n
    cm_f1 = sum(r["program_committee"]["f1"] for r in site_results.values()) / n
    cm_aff = sum(r["program_committee"]["affiliation_accuracy"] for r in site_results.values()) / n
    cm_country = sum(r["program_committee"]["country_accuracy"] for r in site_results.values()) / n
    cm_role = sum(r["program_committee"]["role_accuracy"] for r in site_results.values()) / n

    return {
        "sites_evaluated": n,
        "dates": {"accuracy": round(dates_acc, 4)},
        "venue": {"accuracy": round(venue_acc, 4)},
        "topics": {
            "precision": round(topics_p, 4),
            "recall": round(topics_r, 4),
            "f1": round(topics_f1, 4),
        },
        "keynote_speakers": {
            "precision": round(sp_p, 4),
            "recall": round(sp_r, 4),
            "f1": round(sp_f1, 4),
            "affiliation_accuracy": round(sp_aff, 4),
            "country_accuracy": round(sp_country, 4),
        },
        "program_committee": {
            "precision": round(cm_p, 4),
            "recall": round(cm_r, 4),
            "f1": round(cm_f1, 4),
            "affiliation_accuracy": round(cm_aff, 4),
            "country_accuracy": round(cm_country, 4),
            "role_accuracy": round(cm_role, 4),
        },
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(
    site_results: Dict[str, Dict[str, Any]],
    agg: Dict[str, Any],
    golden_dir: str,
    pred_dir: str,
) -> str:
    """Build a human-readable report and return it as a string."""
    lines = []
    lines.append("=" * 80)
    lines.append("EVALUATION REPORT")
    lines.append(f"Golden: {golden_dir}")
    lines.append(f"Pred:   {pred_dir}")
    lines.append(f"Sites:  {agg.get('sites_evaluated', 0)}")
    lines.append("=" * 80)

    # --- Per-site details ---
    lines.append("")
    lines.append("PER-SITE RESULTS")
    lines.append("-" * 80)

    for filename, r in sorted(site_results.items()):
        lines.append(f"\n  {filename}")

        d = r["dates"]["fields"]
        lines.append(f"    dates:      start={d['start_date']:<10s}  end={d['end_date']:<10s}  "
                      f"acc={r['dates']['accuracy']:.0%}")

        v = r["venue"]["fields"]
        lines.append(f"    venue:      city={v['city']:<10s}  country={v['country']:<10s}  "
                      f"acc={r['venue']['accuracy']:.0%}")

        t = r["topics"]
        lines.append(f"    topics:     P={t['precision']:.2f}  R={t['recall']:.2f}  F1={t['f1']:.2f}  "
                      f"({t['matched']}/{t['golden_count']} golden, {t['pred_count']} pred)")

        s = r["keynote_speakers"]
        lines.append(f"    speakers:   P={s['precision']:.2f}  R={s['recall']:.2f}  F1={s['f1']:.2f}  "
                      f"({s['matched']}/{s['golden_count']} golden, {s['pred_count']} pred)  "
                      f"aff={s['affiliation_accuracy']:.0%}  country={s['country_accuracy']:.0%}")

        c = r["program_committee"]
        lines.append(f"    committee:  P={c['precision']:.2f}  R={c['recall']:.2f}  F1={c['f1']:.2f}  "
                      f"({c['matched']}/{c['golden_count']} golden, {c['pred_count']} pred)  "
                      f"aff={c['affiliation_accuracy']:.0%}  country={c['country_accuracy']:.0%}  "
                      f"role={c['role_accuracy']:.0%}")

    # --- Aggregate ---
    lines.append("")
    lines.append("=" * 80)
    lines.append("AGGREGATE (macro-average over sites)")
    lines.append("-" * 80)
    lines.append(f"  dates accuracy:          {agg['dates']['accuracy']:.1%}")
    lines.append(f"  venue accuracy:          {agg['venue']['accuracy']:.1%}")
    lines.append(f"  topics P/R/F1:           {agg['topics']['precision']:.2f} / "
                  f"{agg['topics']['recall']:.2f} / {agg['topics']['f1']:.2f}")
    lines.append(f"  speakers P/R/F1:         {agg['keynote_speakers']['precision']:.2f} / "
                  f"{agg['keynote_speakers']['recall']:.2f} / {agg['keynote_speakers']['f1']:.2f}")
    lines.append(f"    affiliation accuracy:   {agg['keynote_speakers']['affiliation_accuracy']:.1%}")
    lines.append(f"    country accuracy:       {agg['keynote_speakers']['country_accuracy']:.1%}")
    lines.append(f"  committee P/R/F1:        {agg['program_committee']['precision']:.2f} / "
                  f"{agg['program_committee']['recall']:.2f} / {agg['program_committee']['f1']:.2f}")
    lines.append(f"    affiliation accuracy:   {agg['program_committee']['affiliation_accuracy']:.1%}")
    lines.append(f"    country accuracy:       {agg['program_committee']['country_accuracy']:.1%}")
    lines.append(f"    role accuracy:          {agg['program_committee']['role_accuracy']:.1%}")
    lines.append("=" * 80)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model results against a golden dataset",
    )
    parser.add_argument(
        "--golden", "-g", required=True,
        help="Directory with golden (reference) result JSON files",
    )
    parser.add_argument(
        "--pred", "-p", required=True,
        help="Directory with predicted (model) result JSON files",
    )
    parser.add_argument(
        "--fuzzy-threshold", type=int, default=75,
        help="Fuzzy matching threshold (default: 75)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save full report as JSON to this path",
    )
    args = parser.parse_args()

    golden = _load_results(args.golden)
    pred = _load_results(args.pred)

    if not golden:
        print(f"ERROR: no JSON files found in {args.golden}")
        sys.exit(1)

    # Match files
    common = sorted(set(golden.keys()) & set(pred.keys()))
    if not common:
        print(f"ERROR: no matching filenames between golden and pred directories")
        print(f"  Golden files: {sorted(golden.keys())[:5]}")
        print(f"  Pred files:   {sorted(pred.keys())[:5]}")
        sys.exit(1)

    missing = set(golden.keys()) - set(pred.keys())
    if missing:
        logger.warning(
            "%d golden file(s) have no match in pred: %s",
            len(missing), sorted(missing)[:5],
        )

    # Evaluate
    site_results = {}
    for filename in common:
        site_results[filename] = evaluate_site(
            golden[filename], pred[filename],
            fuzzy_threshold=args.fuzzy_threshold,
        )

    agg = aggregate(site_results)

    # Print report
    report = print_report(site_results, agg, args.golden, args.pred)
    print(report)

    # Save JSON
    if args.output:
        full_report = {
            "golden_dir": args.golden,
            "pred_dir": args.pred,
            "fuzzy_threshold": args.fuzzy_threshold,
            "aggregate": agg,
            "per_site": site_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
