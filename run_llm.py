#!/usr/bin/env python3
"""
Phase 2: Run LLM extraction on prepared contexts (offline).

Run this on a machine with an LLM backend (Ollama or vLLM) but
no internet needed. Reads the contexts prepared by ``prepare_contexts.py``
and runs the full extraction pipeline: LLM → validate → merge.

Usage:
    python run_llm.py --input prepared/ --output results/
    python run_llm.py --input prepared/ --models mistral:latest qwen3:4b
    python run_llm.py --input prepared/ --backend vllm --models Qwen/Qwen2.5-7B-Instruct
    python run_llm.py --input prepared/ --models-file models.txt --output results/ -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from extractor.content_selection import Category
from extractor.llm import (
    extract_other,
    extract_topics,
    extract_speakers,
    extract_committee,
    DEFAULT_MODEL,
    DEFAULT_BACKEND,
    get_default_url,
)
from extractor.validator import validate_category

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ModelSpec = Tuple[str, List[str]]

DEFAULT_MODELS: List[ModelSpec] = [
    ("mistral:latest", []),
]

# ---------------------------------------------------------------------------
# Category → extraction function mapping
# ---------------------------------------------------------------------------

_EXTRACT_FN: Dict[Category, Callable] = {
    Category.OTHER: extract_other,
    Category.TOPICS: extract_topics,
    Category.SPEAKERS: extract_speakers,
    Category.COMMITTEE: extract_committee,
}


# ---------------------------------------------------------------------------
# Has-data checks (same as pipeline.py)
# ---------------------------------------------------------------------------

def _has_other_data(result: Dict[str, Any]) -> bool:
    if not result:
        return False
    has_name = bool(result.get("full_name"))
    has_dates = bool(result.get("start_date"))
    has_venue = bool(result.get("city"))
    has_deadline = bool(result.get("submission_deadline"))
    return has_name and (has_dates or has_venue or has_deadline)


def _has_list_data(result: Dict[str, Any], key: str) -> bool:
    if not result:
        return False
    val = result.get(key, [])
    return isinstance(val, list) and len(val) > 0


_HAS_DATA: Dict[Category, Callable[[Dict[str, Any]], bool]] = {
    Category.OTHER: _has_other_data,
    Category.TOPICS: lambda r: _has_list_data(r, "topics"),
    Category.SPEAKERS: lambda r: _has_list_data(r, "keynote_speakers"),
    Category.COMMITTEE: lambda r: _has_list_data(r, "program_committee"),
}


# ---------------------------------------------------------------------------
# Merge helpers (same as pipeline.py)
# ---------------------------------------------------------------------------

def _merge_other(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key in (
        "full_name", "acronym", "edition_number",
        "start_date", "end_date",
        "city", "country",
        "submission_deadline", "notification_date", "camera_ready_date",
        "publisher", "series",
    ):
        if not merged.get(key) and extra.get(key):
            merged[key] = extra[key]
    return merged


def _merge_list(base: Dict[str, Any], extra: Dict[str, Any], key: str) -> Dict[str, Any]:
    merged = dict(base)
    if not merged.get(key) and extra.get(key):
        merged[key] = extra[key]
    return merged


def _merge_category(
    category: Category,
    base: Dict[str, Any],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    if not base:
        return extra or {}
    if not extra:
        return base
    if category == Category.OTHER:
        return _merge_other(base, extra)
    elif category == Category.TOPICS:
        return _merge_list(base, extra, "topics")
    elif category == Category.SPEAKERS:
        return _merge_list(base, extra, "keynote_speakers")
    elif category == Category.COMMITTEE:
        return _merge_list(base, extra, "program_committee")
    return base


# ---------------------------------------------------------------------------
# Run extraction + validation for one context
# ---------------------------------------------------------------------------

def _run_extract(
    category: Category,
    context: str,
    *,
    model: str,
    backend: str,
    base_url: str,
    vllm_extra_args: Optional[List[str]],
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    """Call LLM and validate against context."""
    fn = _EXTRACT_FN[category]
    raw = fn(
        context,
        model=model,
        backend=backend,
        base_url=base_url,
        vllm_extra_args=vllm_extra_args,
    )
    if raw is None:
        return None
    validated, val_warnings = validate_category(category, raw, context)
    warnings.extend(val_warnings)
    return validated


# ---------------------------------------------------------------------------
# Process one prepared site
# ---------------------------------------------------------------------------

def _build_output(
    url: str,
    other: Dict[str, Any],
    topics: Dict[str, Any],
    speakers: Dict[str, Any],
    committee: Dict[str, Any],
) -> Dict[str, Any]:
    o = other or {}
    t = topics or {}
    s = speakers or {}
    c = committee or {}
    return {
        "conference": {
            "full_name": o.get("full_name") or "",
            "acronym": o.get("acronym") or "",
            "url": url,
            "edition_number": o.get("edition_number"),
        },
        "dates": {
            "start_date": o.get("start_date"),
            "end_date": o.get("end_date"),
        },
        "venue": {
            "city": o.get("city"),
            "country": o.get("country"),
        },
        "deadlines": {
            "submission": o.get("submission_deadline"),
            "notification": o.get("notification_date"),
            "camera_ready": o.get("camera_ready_date"),
        },
        "topics": t.get("topics", []),
        "keynote_speakers": s.get("keynote_speakers", []),
        "program_committee": c.get("program_committee", []),
        "publication": {
            "publisher": o.get("publisher"),
            "series": o.get("series"),
        },
    }


def process_site(
    prepared: Dict[str, Any],
    *,
    model: str,
    backend: str,
    base_url: str,
    vllm_extra_args: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Run the full extraction pipeline on a prepared site.

    For each category:
      - Algorithm 1: try L1 → L2 → L3 (stop early if sufficient)
      - Algorithm 2: brute-force (all pages)
      - Merge: targeted priority, brute-force fills gaps
    """
    url = prepared["url"]
    warnings: List[str] = []

    if prepared.get("error"):
        return {
            "url": url,
            "model": model,
            "status": "error",
            "error": prepared["error"],
            "data": {},
            "warnings": [prepared["error"]],
        }

    llm_kwargs = dict(
        model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args,
    )

    category_results: Dict[Category, Dict[str, Any]] = {}

    for category in Category:
        cat_key = category.value
        cat_data = prepared["categories"].get(cat_key, {})
        has_data = _HAS_DATA[category]

        # Algorithm 1: targeted (3 levels)
        targeted: Dict[str, Any] = {}

        for level in ("L1", "L2", "L3"):
            level_data = cat_data.get(level, {})
            ctx = level_data.get("context", "")
            if not ctx:
                continue

            logger.info(
                "Algo1 %s [%s]: %d page(s), %d chars",
                level, cat_key,
                level_data.get("page_count", 0), len(ctx),
            )
            r = _run_extract(
                category, ctx, warnings=warnings, **llm_kwargs,
            )
            if r:
                targeted = _merge_category(category, targeted, r)
                if has_data(targeted):
                    logger.info(
                        "Algo1 %s [%s]: sufficient data found",
                        level, cat_key,
                    )
                    break

        # Algorithm 2: brute-force
        bf_data = cat_data.get("bruteforce", {})
        bf_ctx = bf_data.get("context", "")
        bruteforce: Dict[str, Any] = {}
        if bf_ctx:
            logger.info(
                "Algo2 [%s]: %d page(s), %d chars",
                cat_key, bf_data.get("page_count", 0), len(bf_ctx),
            )
            r = _run_extract(
                category, bf_ctx, warnings=warnings, **llm_kwargs,
            )
            if r:
                bruteforce = r

        # Merge: targeted priority
        merged = _merge_category(category, targeted, bruteforce)
        category_results[category] = merged

        if not has_data(merged):
            warnings.append(f"No data found for category: {cat_key}")

    # Build final output
    data = _build_output(
        url,
        other=category_results.get(Category.OTHER, {}),
        topics=category_results.get(Category.TOPICS, {}),
        speakers=category_results.get(Category.SPEAKERS, {}),
        committee=category_results.get(Category.COMMITTEE, {}),
    )

    # Cross-validation: speakers vs committee overlap
    speaker_names = {
        s.get("name", "").strip().lower()
        for s in data.get("keynote_speakers", [])
        if s.get("name")
    }
    committee_names = {
        c.get("name", "").strip().lower()
        for c in data.get("program_committee", [])
        if c.get("name")
    }
    overlap = speaker_names & committee_names
    if overlap:
        warnings.append(
            f"Speaker/committee overlap ({len(overlap)}): {', '.join(sorted(overlap))}"
        )
        data["program_committee"] = [
            c for c in data["program_committee"]
            if c.get("name", "").strip().lower() not in overlap
        ]

    return {
        "url": url,
        "model": model,
        "backend": backend,
        "status": "ok",
        "error": None,
        "pages_fetched": prepared.get("pages_fetched", 0),
        "warnings": warnings,
        "data": data,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_name(s: str) -> str:
    s = s.replace("https://", "").replace("http://", "")
    s = re.sub(r"[/:.\-]+", "_", s)
    s = s.strip("_")
    return s[:80]


def _read_lines(path: str) -> List[str]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def _read_models_file(path: str) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for line in _read_lines(path):
        parts = shlex.split(line, comments=True)
        if not parts:
            continue
        name, *extra = parts
        specs.append((name, extra))
    return specs


def _count_fields(data: Dict[str, Any]) -> Tuple[int, int]:
    total = 0
    filled = 0
    for section_name in ["conference", "dates", "venue", "deadlines", "publication"]:
        section = data.get(section_name, {})
        for val in section.values():
            total += 1
            if val is not None and val != "":
                filled += 1
    total += 3
    if data.get("topics"):
        filled += 1
    if data.get("keynote_speakers"):
        filled += 1
    if data.get("program_committee"):
        filled += 1
    return filled, total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM extraction on prepared contexts (Phase 2)",
    )
    parser.add_argument(
        "--input", "-i", default="prepared",
        help="Directory with prepared contexts from prepare_contexts.py",
    )
    parser.add_argument(
        "--output", "-o", default="results",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--backend", "-b", default=DEFAULT_BACKEND, choices=["ollama", "vllm"],
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
    )
    parser.add_argument(
        "--models-file", default=None,
    )
    parser.add_argument(
        "--base-url", default=None,
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve models
    models: List[ModelSpec]
    if args.models_file:
        models = _read_models_file(args.models_file)
    elif args.models:
        models = []
        for raw in args.models:
            parts = shlex.split(raw)
            name, *extra = parts
            models.append((name, extra))
    else:
        models = DEFAULT_MODELS

    base_url = args.base_url or get_default_url(args.backend)

    # Load prepared contexts
    prepared_files = sorted([
        f for f in os.listdir(args.input) if f.endswith(".json")
    ])
    if not prepared_files:
        print(f"ERROR: no prepared contexts found in {args.input}/")
        return

    print(f"Backend:  {args.backend}")
    print(f"Base URL: {base_url}")
    print(f"Models:   {len(models)}")
    for name, extra in models:
        label = name if not extra else f"{name} [{' '.join(extra)}]"
        print(f"  - {label}")
    print(f"Sites:    {len(prepared_files)}")
    print(f"Total:    {len(models) * len(prepared_files)} extraction runs")
    print()

    os.makedirs(args.output, exist_ok=True)

    for model, extra_args in models:
        model_dir = os.path.join(args.output, _sanitize_name(model))
        os.makedirs(model_dir, exist_ok=True)

        # Resume: check what's already done for this model
        already_done = {
            f for f in os.listdir(model_dir) if f.endswith(".json")
        }

        remaining_files = [
            f for f in prepared_files if f not in already_done
        ]

        display_model = model if not extra_args else f"{model} {' '.join(extra_args)}"
        logger.info(
            "Model %s: %d total, %d done, %d remaining",
            display_model, len(prepared_files),
            len(already_done), len(remaining_files),
        )

        if not remaining_files:
            logger.info("Nothing to do for %s", display_model)
            continue

        start_time = time.time()
        done = len(already_done)
        errors = 0

        for prep_file in remaining_files:
            prep_path = os.path.join(args.input, prep_file)
            with open(prep_path, encoding="utf-8") as f:
                prepared = json.load(f)

            url = prepared["url"]
            logger.info("Processing %s with %s", url, display_model)

            t0 = time.time()
            try:
                result = process_site(
                    prepared,
                    model=model,
                    backend=args.backend,
                    base_url=base_url,
                    vllm_extra_args=extra_args or None,
                )
            except Exception as exc:
                logger.exception("Failed: %s", url)
                result = {
                    "url": url, "model": model, "backend": args.backend,
                    "status": "error", "error": str(exc),
                    "data": {}, "warnings": [str(exc)],
                }
                errors += 1
            elapsed = round(time.time() - t0, 1)
            result["elapsed_sec"] = elapsed

            # Add fill stats
            if result.get("data"):
                filled, total = _count_fields(result["data"])
                result["fields_filled"] = filled
                result["fields_total"] = total
                result["fill_pct"] = round(filled / total * 100, 1) if total > 0 else 0.0

            # Save
            out_path = os.path.join(model_dir, prep_file)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            done += 1
            if done % 10 == 0 or done == len(prepared_files):
                total_elapsed = time.time() - start_time
                processed = done - len(already_done)
                rate = processed / total_elapsed if total_elapsed > 0 else 0
                left = len(prepared_files) - done
                eta = left / rate if rate > 0 else 0
                logger.info(
                    "[%s] %d/%d (%.0f%%) | errors: %d | %.2f url/s | ETA: %.0fs",
                    display_model,
                    done, len(prepared_files),
                    100 * done / len(prepared_files),
                    errors, rate, eta,
                )

            if result.get("status") == "ok":
                halluc = len([w for w in result.get("warnings", []) if "[hallucination]" in w])
                print(
                    f"  {url} -> {result.get('fill_pct', 0)}% filled, "
                    f"{halluc} hallucinations, {elapsed}s"
                )
            else:
                print(f"  {url} -> ERROR: {result.get('error')} ({elapsed}s)")

    print("\nDone. Results saved to:", args.output)


if __name__ == "__main__":
    main()
