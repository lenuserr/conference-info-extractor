#!/usr/bin/env python3
"""
Extract conference metadata from wikicfp_conferences.json using local LLM.

Reads conference entries, scrapes each website_url, extracts structured info
via the existing pipeline (ollama / vLLM), and saves results to a JSON file
that preserves all original wikicfp fields alongside the extracted data.

Supports:
  - Multiple models (run sequentially, each producing its own output)
  - Resume: skips URLs already present in the output file
  - Configurable limit / offset for partial runs

Usage:
    # Ollama (default)
    python run_wikicfp.py
    python run_wikicfp.py --models mistral:latest qwen3:4b --limit 50

    # vLLM
    python run_wikicfp.py --backend vllm --models Qwen/Qwen2.5-7B-Instruct

    # Resume a previous run (output file already has some results)
    python run_wikicfp.py --output results_wikicfp.json

    # Process only entries 100-200
    python run_wikicfp.py --offset 100 --limit 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from extractor.pipeline import extract_conference
from extractor.llm import DEFAULT_MODEL, DEFAULT_BACKEND

logger = logging.getLogger(__name__)

ModelSpec = Tuple[str, List[str]]  # (model_name, extra_vllm_args)

DEFAULT_INPUT = "wikicfp_conferences.json"
DEFAULT_OUTPUT = "wikicfp_results.json"


def _load_input(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_existing_results(path: str) -> List[Dict[str, Any]]:
    """Load previously saved results for resume support."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Could not parse existing output file %s, starting fresh", path)
        return []


def _save_results(results: List[Dict[str, Any]], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _done_keys(results: List[Dict[str, Any]], model: str) -> set:
    """Build set of website_urls already processed for a given model."""
    return {
        r["wikicfp"]["website_url"]
        for r in results
        if r.get("meta", {}).get("model") == model
        and r.get("wikicfp", {}).get("website_url")
    }


def run_extraction(
    conferences: List[Dict[str, Any]],
    model: str,
    backend: str,
    base_url: Optional[str],
    extra_args: List[str],
    output_path: str,
    save_every: int = 5,
) -> None:
    """Run extraction for all conferences with one model, saving incrementally."""
    results = _load_existing_results(output_path)
    done = _done_keys(results, model)

    to_process = [c for c in conferences if c["website_url"] not in done]
    total = len(to_process)
    skipped = len(conferences) - total

    if skipped:
        print(f"  Resuming: {skipped} already done, {total} remaining")
    if total == 0:
        print("  Nothing to do for this model.")
        return

    display_model = model if not extra_args else f"{model} {' '.join(extra_args)}"

    for i, conf in enumerate(to_process, 1):
        url = conf["website_url"]
        print(f"\n  [{i}/{total}] model={display_model}  url={url}")

        t0 = time.time()
        try:
            result = extract_conference(
                url,
                model=model,
                backend=backend,
                base_url=base_url,
                vllm_extra_args=extra_args or None,
            )
            status = "ok"
            error = None
        except Exception as exc:
            logger.exception("Extraction failed for %s", url)
            result = {
                "data": {},
                "confidence": {},
                "warnings": [str(exc)],
                "meta": {"model": model, "backend": backend},
            }
            status = "error"
            error = str(exc)
        elapsed = round(time.time() - t0, 1)

        entry = {
            "wikicfp": conf,
            "extracted": result["data"],
            "confidence": result.get("confidence", {}),
            "warnings": result.get("warnings", []),
            "meta": {
                **result.get("meta", {}),
                "model": model,
                "backend": backend,
                "status": status,
                "error": error,
                "elapsed_sec": elapsed,
                "extracted_at": datetime.now().isoformat(),
            },
        }
        results.append(entry)

        if status == "ok":
            print(f"    -> done in {elapsed}s")
        else:
            print(f"    -> ERROR: {error} ({elapsed}s)")

        # Save periodically
        if i % save_every == 0 or i == total:
            _save_results(results, output_path)
            logger.debug("Saved %d results to %s", len(results), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract conference info from wikicfp_conferences.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", default=DEFAULT_INPUT,
        help=f"Input JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--backend", "-b", default=DEFAULT_BACKEND, choices=["ollama", "vllm"],
        help=f"Inference backend (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model name(s). For vLLM, quote extra args: \"Qwen/Qwen3-32B --max-model-len 131072\"",
    )
    parser.add_argument(
        "--models-file", default=None,
        help="File with model specs, one per line",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="Server URL override",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip first N conferences (default: 0)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N conferences (default: all)",
    )
    parser.add_argument(
        "--save-every", type=int, default=5,
        help="Save results to disk every N extractions (default: 5)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load input
    conferences = _load_input(args.input)
    print(f"Loaded {len(conferences)} conferences from {args.input}")

    # Apply offset/limit
    conferences = conferences[args.offset:]
    if args.limit is not None:
        conferences = conferences[:args.limit]
    print(f"Processing {len(conferences)} conferences (offset={args.offset}, limit={args.limit})")

    # Resolve models
    models: List[ModelSpec]
    if args.models_file:
        models = []
        with open(args.models_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = shlex.split(line)
                    name, *extra = parts
                    models.append((name, extra))
    elif args.models:
        models = []
        for raw in args.models:
            parts = shlex.split(raw)
            name, *extra = parts
            models.append((name, extra))
    else:
        models = [(DEFAULT_MODEL, [])]

    print(f"Backend: {args.backend}")
    print("Models:")
    for name, extra in models:
        label = name if not extra else f"{name} [{' '.join(extra)}]"
        print(f"  - {label}")
    print(f"Output:  {args.output}")
    print()

    for model, extra_args in models:
        display = model if not extra_args else f"{model} [{' '.join(extra_args)}]"
        print(f"=== Model: {display} ===")
        run_extraction(
            conferences,
            model=model,
            backend=args.backend,
            base_url=args.base_url,
            extra_args=extra_args,
            output_path=args.output,
            save_every=args.save_every,
        )

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
