#!/usr/bin/env python3
"""
CLI entry point for conference metadata extraction.

Usage:
    python extract.py <URL> [--model MODEL] [--ollama-url URL] [--output FILE]

Examples:
    python extract.py https://neurips.cc
    python extract.py https://neurips.cc --model qwen2.5:7b --output neurips.json
    python extract.py https://neurips.cc --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from extractor.pipeline import extract_conference
from extractor.llm import DEFAULT_MODEL, DEFAULT_OLLAMA_URL


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured metadata from a conference website → JSON",
    )
    parser.add_argument("url", help="Conference website URL")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-url", default=DEFAULT_OLLAMA_URL,
        help=f"Ollama API base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = extract_conference(
        url=args.url,
        model=args.model,
        ollama_url=args.ollama_url,
    )

    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"✓ Result saved to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
