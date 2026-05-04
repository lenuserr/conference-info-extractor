#!/usr/bin/env python3
"""
Phase 2: Run LLM extraction on prepared contexts (offline).

Run this on a machine with an LLM backend (Ollama or vLLM) but
no internet needed. Reads the contexts prepared by ``prepare_contexts.py``
and runs the full extraction pipeline via ``extract_from_prepared``.

Uses the exact same extraction logic as ``extract_conference`` (live mode)
and ``benchmark.py`` — only the input source differs.

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
from typing import Any, Dict, List, Tuple

from extractor.pipeline import extract_from_prepared
from extractor.llm import DEFAULT_BACKEND, get_default_url

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
        "--backend", "-b", default=DEFAULT_BACKEND, choices=["ollama", "vllm", "claude"],
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--models-file", default=None)
    parser.add_argument("--base-url", default=None)
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

        # Resume: check what's already done
        already_done = {f for f in os.listdir(model_dir) if f.endswith(".json")}
        remaining_files = [f for f in prepared_files if f not in already_done]

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
                result = extract_from_prepared(
                    prepared,
                    model=model,
                    backend=args.backend,
                    base_url=base_url,
                    vllm_extra_args=extra_args or None,
                )
                status = "ok"
            except Exception as exc:
                logger.exception("Failed: %s", url)
                result = {
                    "data": {},
                    "warnings": [str(exc)],
                    "meta": {},
                }
                status = "error"
                errors += 1

            elapsed = round(time.time() - t0, 1)

            # Build output entry
            entry = {
                "url": url,
                "model": model,
                "vllm_extra_args": extra_args,
                "backend": args.backend,
                "status": status,
                "elapsed_sec": elapsed,
                "pages_fetched": result.get("meta", {}).get("pages_fetched", 0),
                "warnings": result.get("warnings", []),
                "data": result.get("data", {}),
                "sources": result.get("sources", {}),
            }

            # Add fill stats
            data = entry["data"]
            if data:
                filled, total = _count_fields(data)
                entry["fields_filled"] = filled
                entry["fields_total"] = total
                entry["fill_pct"] = round(filled / total * 100, 1) if total > 0 else 0.0
                entry["hallucinations_caught"] = len([
                    w for w in entry["warnings"] if "[hallucination]" in w
                ])

            # Save result
            out_path = os.path.join(model_dir, prep_file)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)


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

            if status == "ok":
                halluc = entry.get("hallucinations_caught", 0)
                fill = entry.get("fill_pct", 0)
                print(f"  {url} -> {fill}% filled, {halluc} hallucinations, {elapsed}s")
            else:
                print(f"  {url} -> ERROR ({elapsed}s)")

    print("\nDone. Results saved to:", args.output)


if __name__ == "__main__":
    main()
