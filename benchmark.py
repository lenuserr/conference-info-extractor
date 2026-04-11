#!/usr/bin/env python3
"""
Benchmark: run extraction across multiple models x multiple URLs.

Results are saved into a structured directory:
    results/
    ├── benchmark_20260405_184500.json   # full aggregated report
    ├── mistral_latest/
    │   ├── neurips_cc.json
    │   ├── acit_tech.json
    │   └── ...
    ├── Qwen__Qwen2_5_7B_Instruct/
    │   └── ...
    └── summary.txt                      # human-readable comparison table

Usage:
    # Ollama (default)
    python benchmark.py
    python benchmark.py --models mistral:latest qwen3:4b

    # vLLM
    python benchmark.py --backend vllm --models Qwen/Qwen2.5-7B-Instruct

    # From files
    python benchmark.py --models-file models.txt --urls-file urls.txt
    python benchmark.py --outdir my_results -v

models.txt format:
    # One model per line. Lines starting with "#" and blank lines are ignored.
    # For the vLLM backend, any tokens after the model name are forwarded
    # verbatim to ``vllm serve`` as CLI flags.
    mistral:latest
    Qwen/Qwen3-32B --max-model-len 131072
    Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.9 --dtype bfloat16
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
from typing import Any, Dict, List, Tuple

from extractor.pipeline import extract_conference
from extractor.llm import DEFAULT_MODEL, DEFAULT_BACKEND

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ModelSpec = Tuple[str, List[str]]  # (model_name, extra_vllm_args)

DEFAULT_MODELS: List[ModelSpec] = [
    ("mistral:latest", []),
    ("qwen3:4b", []),
    ("deepseek-r1:7b", []),
]

DEFAULT_URLS = [
    "https://neurips.cc",
    "https://2024.gecon-conference.org",
    "https://2024.aclweb.org",
    "https://acit.tech",
    "https://aaai.org/conference/aaai/aaai-25/",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_name(s: str) -> str:
    """Turn a model name or URL into a safe directory/file name."""
    s = s.replace("https://", "").replace("http://", "")
    s = re.sub(r"[/:.\-]+", "_", s)
    s = s.strip("_")
    return s[:80]


def _count_fields(data: Dict[str, Any]) -> tuple:
    """Return (filled, total) scalar fields count."""
    total = 0
    filled = 0
    for section_name in ["conference", "dates", "venue", "deadlines", "publication"]:
        section = data.get(section_name, {})
        for val in section.values():
            total += 1
            if val is not None and val != "":
                filled += 1
    total += 3  # topics + keynote_speakers + program_committee
    if data.get("topics"):
        filled += 1
    if data.get("keynote_speakers"):
        filled += 1
    if data.get("program_committee"):
        filled += 1
    return filled, total


def _read_lines(path: str) -> List[str]:
    """Read non-empty, non-comment lines from a file."""
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def _read_models_file(path: str) -> List[ModelSpec]:
    """
    Read a models file where each non-empty, non-comment line has the form:

        <model_name> [vllm serve flag ...]

    Tokens after the model name are forwarded to ``vllm serve`` (ignored for
    the ollama backend). Parsing uses shell rules (``shlex``), so quoted
    values are supported.
    """
    specs: List[ModelSpec] = []
    for line in _read_lines(path):
        parts = shlex.split(line, comments=True)
        if not parts:
            continue
        name, *extra = parts
        specs.append((name, extra))
    return specs


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_benchmark(
    models: List[ModelSpec],
    urls: List[str],
    outdir: str,
    backend: str,
    base_url: str | None,
) -> Dict[str, Any]:
    """
    Run extraction for every (model, url) pair.
    Save per-model/per-url JSON files and return the full report.
    """
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report: Dict[str, Any] = {
        "timestamp": timestamp,
        "backend": backend,
        "models": [
            {"name": name, "vllm_extra_args": extra} for name, extra in models
        ],
        "urls": urls,
        "results": [],
    }

    total_combos = len(models) * len(urls)
    current = 0

    for model, extra_args in models:
        model_dir = os.path.join(outdir, _sanitize_name(model))
        os.makedirs(model_dir, exist_ok=True)

        display_model = model if not extra_args else f"{model} {' '.join(extra_args)}"

        for url in urls:
            current += 1
            print(f"\n[{current}/{total_combos}] backend={backend}  model={display_model}  url={url}")
            print("-" * 60)

            t0 = time.time()
            try:
                result = extract_conference(
                    url, model=model, backend=backend, base_url=base_url,
                    vllm_extra_args=extra_args or None,
                )
                status = "ok"
                error = None
            except Exception as exc:
                logger.exception(
                    "Extraction failed for url=%s model=%s backend=%s",
                    url, display_model, backend,
                )
                result = {"data": {}, "confidence": {}, "warnings": [str(exc)], "meta": {}}
                status = "error"
                error = str(exc)
            elapsed = round(time.time() - t0, 1)

            data = result["data"]
            confidence = result["confidence"]
            warnings = result["warnings"]
            meta = result.get("meta", {})

            filled, total = _count_fields(data)
            fill_pct = round(filled / total * 100, 1) if total > 0 else 0.0
            low_fields = [k for k, v in confidence.items() if v == "low"]
            high_fields = [k for k, v in confidence.items() if v == "high"]

            entry = {
                "model": model,
                "vllm_extra_args": extra_args,
                "backend": backend,
                "url": url,
                "status": status,
                "error": error,
                "elapsed_sec": elapsed,
                "pages_fetched": meta.get("pages_fetched", 0),
                "attempts": meta.get("attempts", 0),
                "fields_filled": filled,
                "fields_total": total,
                "fill_pct": fill_pct,
                "high_confidence_count": len(high_fields),
                "low_confidence_fields": low_fields,
                "warnings": warnings,
                "data": data,
                "confidence": confidence,
            }
            report["results"].append(entry)

            # Save individual result
            url_name = _sanitize_name(url)
            result_path = os.path.join(model_dir, f"{url_name}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

            # Print progress
            if status == "ok":
                print(f"  -> {fill_pct}% filled, {len(low_fields)} hallucinated, {elapsed}s")
            else:
                print(f"  -> ERROR: {error} ({elapsed}s)")

    # Save full report
    report_path = os.path.join(outdir, f"benchmark_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Generate summary
    _write_summary(report, outdir)

    return report


def _write_summary(report: Dict[str, Any], outdir: str) -> None:
    """Write a human-readable summary table."""
    # report["models"] is a list of {"name": ..., "vllm_extra_args": [...]}.
    # Summaries are keyed by model name only — extra args are shown once below.
    models: List[str] = [m["name"] for m in report["models"]]
    urls = report["urls"]
    results = report["results"]
    backend = report.get("backend", "?")

    # Build lookup: (model, url) -> entry
    lookup: Dict[tuple, Dict] = {}
    for r in results:
        lookup[(r["model"], r["url"])] = r

    lines: List[str] = []
    lines.append(f"Benchmark — {report['timestamp']}  (backend: {backend})")
    lines.append("=" * 100)
    lines.append("")

    # --- Per-URL comparison table ---
    url_col_w = 40
    model_col_w = 20
    header = f"{'URL':<{url_col_w}}"
    for m in models:
        short = m[:model_col_w - 1]
        header += f" | {short:^{model_col_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    for url in urls:
        short_url = url.replace("https://", "")[:url_col_w - 1]
        row = f"{short_url:<{url_col_w}}"
        for m in models:
            e = lookup.get((m, url))
            if e is None:
                cell = "-"
            elif e["status"] != "ok":
                cell = "ERR"
            else:
                cell = f"{e['fill_pct']}% h:{len(e['low_confidence_fields'])}"
            row += f" | {cell:^{model_col_w}}"
        lines.append(row)

    lines.append("")
    lines.append("(fill% h:hallucinated_fields)")
    lines.append("")

    # --- Per-model averages ---
    extra_by_name = {m["name"]: m.get("vllm_extra_args") or [] for m in report["models"]}
    lines.append("Model averages:")
    lines.append("-" * 60)
    for m in models:
        entries = [lookup[(m, u)] for u in urls if (m, u) in lookup and lookup[(m, u)]["status"] == "ok"]
        extra = extra_by_name.get(m) or []
        label = m if not extra else f"{m} [{' '.join(extra)}]"
        if not entries:
            lines.append(f"  {label}: no successful runs")
            continue
        avg_fill = sum(e["fill_pct"] for e in entries) / len(entries)
        avg_time = sum(e["elapsed_sec"] for e in entries) / len(entries)
        total_halluc = sum(len(e["low_confidence_fields"]) for e in entries)
        lines.append(
            f"  {label}: avg_fill={avg_fill:.1f}%  avg_time={avg_time:.0f}s  "
            f"total_hallucinations={total_halluc}  sites={len(entries)}/{len(urls)}"
        )

    lines.append("")

    summary_text = "\n".join(lines)
    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n\n{'=' * 70}")
    print(summary_text)
    print(f"{'=' * 70}")
    print(f"Full report:  {os.path.join(outdir, 'benchmark_' + report['timestamp'] + '.json')}")
    print(f"Summary:      {summary_path}")
    print(f"Per-model:    {outdir}/<model>/<url>.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark conference extraction across models and URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ollama (default)
  python benchmark.py
  python benchmark.py --models mistral:latest qwen3:4b

  # vLLM
  python benchmark.py --backend vllm --models Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct
  python benchmark.py --backend vllm --base-url http://gpu-server:8000

  # From files
  python benchmark.py --models-file models.txt --urls-file urls.txt
  python benchmark.py --outdir my_results -v
        """,
    )
    parser.add_argument(
        "--backend", "-b", default=DEFAULT_BACKEND, choices=["ollama", "vllm"],
        help=f"Inference backend (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="List of model names",
    )
    parser.add_argument(
        "--models-file", default=None,
        help="File with model names, one per line",
    )
    parser.add_argument(
        "--urls", nargs="+", default=None,
        help="List of conference URLs",
    )
    parser.add_argument(
        "--urls-file", default=None,
        help="File with URLs, one per line",
    )
    parser.add_argument(
        "--outdir", default="results",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="Server URL (default: auto per backend — :11434 for ollama, :8000 for vllm)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve models
    models: List[ModelSpec]
    if args.models_file:
        models = _read_models_file(args.models_file)
    elif args.models:
        # Each --models entry is parsed with shlex so users can pass
        # things like --models "Qwen/Qwen3-32B --max-model-len 131072"
        models = []
        for raw in args.models:
            parts = shlex.split(raw)
            if not parts:
                continue
            name, *extra = parts
            models.append((name, extra))
    else:
        models = DEFAULT_MODELS

    # Resolve URLs
    if args.urls_file:
        urls = _read_lines(args.urls_file)
    elif args.urls:
        urls = args.urls
    else:
        urls = DEFAULT_URLS

    print(f"Backend: {args.backend}")
    print("Models:")
    for name, extra in models:
        if extra:
            print(f"  - {name}  [vllm args: {' '.join(extra)}]")
        else:
            print(f"  - {name}")
    print(f"URLs:    {len(urls)} sites")
    print(f"Total:   {len(models) * len(urls)} extraction runs")
    print(f"Output:  {args.outdir}/")

    run_benchmark(models, urls, args.outdir, args.backend, args.base_url)


if __name__ == "__main__":
    main()
