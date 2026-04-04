#!/usr/bin/env python3
"""
Test extraction on 5+ conference sites of different types.

Runs the full pipeline for each URL and writes a report with:
- model used
- % of fields successfully extracted
- which fields were hallucinated (low confidence)
- full extracted JSON

Usage:
    python test_extraction.py [--model MODEL] [--ollama-url URL]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List

from extractor.pipeline import extract_conference
from extractor.llm import DEFAULT_MODEL, DEFAULT_OLLAMA_URL

# --- Test URLs ---
# Mix of large / medium / small / workshop conferences
TEST_URLS = [
    # Large flagship conference
    "https://neurips.cc",
    # Medium European conference
    "https://2024.gecon-conference.org",
    # ACL conference (NLP, large)
    "https://2024.aclweb.org",
    # Small/medium conference
    "https://acit.tech",
    # AAAI conference
    "https://aaai.org/conference/aaai/aaai-25/",
]


def _count_fields(data: Dict[str, Any]) -> tuple:
    """Return (filled, total) scalar fields count."""
    total = 0
    filled = 0

    for section_name in ["conference", "dates", "venue", "deadlines", "publication"]:
        section = data.get(section_name, {})
        for key, val in section.items():
            total += 1
            if val is not None and val != "":
                filled += 1

    # Arrays count as 1 field each
    total += 2  # topics + keynote_speakers
    if data.get("topics"):
        filled += 1
    if data.get("keynote_speakers"):
        filled += 1

    return filled, total


def run_tests(model: str, ollama_url: str) -> None:
    """Run extraction on all test URLs and print a report."""
    report: List[Dict[str, Any]] = []

    for url in TEST_URLS:
        print(f"\n{'='*70}")
        print(f"  Testing: {url}")
        print(f"{'='*70}")

        try:
            result = extract_conference(url, model=model, ollama_url=ollama_url)
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            report.append({
                "url": url,
                "status": "error",
                "error": str(exc),
            })
            continue

        data = result["data"]
        confidence = result["confidence"]
        warnings = result["warnings"]
        meta = result["meta"]

        filled, total = _count_fields(data)
        pct = (filled / total * 100) if total > 0 else 0

        low_confidence_fields = [k for k, v in confidence.items() if v == "low"]
        high_confidence_fields = [k for k, v in confidence.items() if v == "high"]

        print(f"  Model: {meta.get('model', model)}")
        print(f"  Pages fetched: {meta.get('pages_fetched', '?')}")
        print(f"  Attempts: {meta.get('attempts', '?')}")
        print(f"  Fields filled: {filled}/{total} ({pct:.0f}%)")
        print(f"  High confidence: {len(high_confidence_fields)}")
        print(f"  Low confidence (hallucinated?): {low_confidence_fields or 'none'}")
        if warnings:
            print(f"  Warnings:")
            for w in warnings:
                print(f"    - {w}")

        print(f"\n  Extracted data:")
        print(json.dumps(data, indent=4, ensure_ascii=False))

        report.append({
            "url": url,
            "status": "ok",
            "model": meta.get("model", model),
            "pages_fetched": meta.get("pages_fetched"),
            "attempts": meta.get("attempts"),
            "fields_filled": filled,
            "fields_total": total,
            "fill_pct": round(pct, 1),
            "low_confidence_fields": low_confidence_fields,
            "warnings": warnings,
            "data": data,
        })

    # Write full report
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for r in report:
        status = r.get("status")
        url = r["url"]
        if status == "ok":
            print(f"  ✓ {url}  —  {r['fill_pct']}% filled, {len(r['low_confidence_fields'])} hallucinated")
        else:
            print(f"  ✗ {url}  —  ERROR: {r.get('error', 'unknown')}")
    print(f"\n  Full report saved to: {report_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test conference extraction on multiple sites")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API URL")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_tests(args.model, args.ollama_url)


if __name__ == "__main__":
    main()
