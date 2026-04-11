#!/usr/bin/env python3
"""
Evaluate benchmark results against a gold dataset.

Usage:
    # Evaluate a finished benchmark run
    python evaluate.py --results results/ --gold gold/

    # Write a JSON report alongside the text one
    python evaluate.py --results results/ --gold gold/ --json-out eval.json

    # Show per-example field-level diff for debugging
    python evaluate.py --results results/ --gold gold/ --debug

Gold file format (one JSON per conference in ``--gold``):

    {
      "url": "https://neurips.cc",
      "ground_truth": {
        "conference": {"full_name": "...", "acronym": "NeurIPS", "edition_number": 38},
        "dates":       {"start_date": "2024-12-10", "end_date": "2024-12-15"},
        "venue":       {"city": "Vancouver", "country": "Canada"},
        "deadlines":   {"submission": "2024-05-22", "notification": null, "camera_ready": null},
        "topics":            ["machine learning", "deep learning"],
        "keynote_speakers":  [{"name": "Yoshua Bengio", "affiliation": null, "country": null}],
        "program_committee": [{"name": "Jane Doe", "affiliation": "MIT", "country": "United States", "role": "PC Member"}],
        "publication":       {"publisher": null, "series": null}
      },
      "aliases": {
        "conference.full_name": ["Conference on Neural Information Processing Systems"],
        "venue.city": ["Vancouver, BC"]
      },
      "notes": "verified against https://neurips.cc/Conferences/2024 on 2026-04-10"
    }

Semantics:
  - ``null`` / ``[]`` in ground_truth means "this field is not present on
    the site" (verified by the annotator). The eval then distinguishes
    misses from correct abstentions.
  - ``aliases`` are optional extra acceptable surface forms for fuzzy
    string fields.

Results directory layout (produced by benchmark.py):

    results/<sanitized_model>/<sanitized_url>.json

Each file is a benchmark ``entry`` dict with keys ``url``, ``model``,
``status``, ``data``, etc. Evaluation joins entries to gold by URL.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

from eval.metrics import ExampleResult, ModelMetrics, aggregate, evaluate_one
from eval.report import format_summary, model_metrics_to_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    """Canonical form for URL matching (strip scheme, trailing slash, www.)."""
    if not url:
        return ""
    u = url.strip().lower()
    u = u.replace("https://", "").replace("http://", "")
    if u.startswith("www."):
        u = u[4:]
    u = u.rstrip("/")
    return u


def load_gold(gold_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load every ``*.json`` under ``gold_dir``. Key by normalized URL."""
    gold: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(gold_dir):
        raise FileNotFoundError(f"Gold directory not found: {gold_dir}")

    for name in sorted(os.listdir(gold_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(gold_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed gold file %s: %s", path, exc)
            continue
        url = doc.get("url")
        if not url:
            logger.warning("Gold file %s has no 'url' field, skipping", path)
            continue
        gold[_normalize_url(url)] = doc
    return gold


def load_results(results_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Walk ``results/<model_dir>/*.json`` and group the benchmark entries by
    their ``model`` field (not by directory name — directory is sanitized).
    """
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for sub in sorted(os.listdir(results_dir)):
        sub_path = os.path.join(results_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        for name in sorted(os.listdir(sub_path)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(sub_path, name)
            try:
                with open(path, encoding="utf-8") as f:
                    entry = json.load(f)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed result file %s: %s", path, exc)
                continue
            if not isinstance(entry, dict) or "model" not in entry:
                continue
            by_model.setdefault(entry["model"], []).append(entry)
    return by_model


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate_model(
    model: str,
    entries: List[Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
    debug: bool = False,
) -> Tuple[ModelMetrics, List[ExampleResult]]:
    """
    Run ``evaluate_one`` for every entry that has a matching gold file.
    Returns aggregated metrics plus the per-example results.
    """
    per_example: List[ExampleResult] = []
    matched_entries: List[Dict[str, Any]] = []
    missing: List[str] = []

    for entry in entries:
        key = _normalize_url(entry.get("url", ""))
        if key not in gold:
            missing.append(entry.get("url", "?"))
            continue
        gold_doc = gold[key]
        ex = evaluate_one(gold_doc, entry, aliases=gold_doc.get("aliases"))
        per_example.append(ex)
        matched_entries.append(entry)

    metrics = aggregate(model, per_example, entries=matched_entries)

    if debug:
        print(f"\n--- DEBUG: {model} ---")
        for ex in per_example:
            print(f"  {ex.url}")
            for fr in ex.fields:
                if fr.outcome is not None:
                    if fr.outcome.value not in ("tp", "tn"):
                        print(f"    {fr.path:<28} {fr.outcome.value:<10} gold={fr.gold!r}  pred={fr.pred!r}")
                else:
                    if fr.fp or fr.fn:
                        print(
                            f"    {fr.path:<28} list  "
                            f"tp={fr.tp} fp={fr.fp} fn={fr.fn}"
                        )
        if missing:
            print(f"  (skipped {len(missing)} URL(s) with no gold: {missing})")

    if missing:
        logger.info("%s: %d URL(s) had no gold — excluded from metrics", model, len(missing))

    return metrics, per_example


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark results against a gold dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", default="results", help="benchmark results directory")
    parser.add_argument("--gold", default="gold", help="gold dataset directory")
    parser.add_argument("--json-out", default=None, help="write JSON report to this path")
    parser.add_argument("--text-out", default=None, help="write text report to this path (default: stdout only)")
    parser.add_argument("--debug", action="store_true", help="print per-example field-level diffs")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        gold = load_gold(args.gold)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not gold:
        print(f"ERROR: no gold files found in {args.gold}/", file=sys.stderr)
        return 2
    print(f"Loaded {len(gold)} gold example(s) from {args.gold}/")

    try:
        by_model = load_results(args.results)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not by_model:
        print(f"ERROR: no result entries found in {args.results}/", file=sys.stderr)
        return 2
    print(f"Loaded entries for {len(by_model)} model(s) from {args.results}/")

    all_metrics: List[ModelMetrics] = []
    for model in sorted(by_model):
        metrics, _ = evaluate_model(model, by_model[model], gold, debug=args.debug)
        all_metrics.append(metrics)

    summary = format_summary(all_metrics)
    print(summary)

    if args.text_out:
        with open(args.text_out, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Text report: {args.text_out}")

    if args.json_out:
        payload = {
            "gold_dir": args.gold,
            "results_dir": args.results,
            "n_gold": len(gold),
            "models": [model_metrics_to_dict(m) for m in all_metrics],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"JSON report: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
