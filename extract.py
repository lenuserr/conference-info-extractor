#!/usr/bin/env python3
"""
CLI entry point for conference metadata extraction.

Usage:
    python extract.py <URL> [--backend ollama|vllm] [--model MODEL] [--base-url URL] [--output FILE]

Examples:
    # Ollama (default)
    python extract.py https://neurips.cc
    python extract.py https://neurips.cc --model mistral:latest -o neurips.json

    # vLLM
    python extract.py https://neurips.cc --backend vllm --model Qwen/Qwen2.5-7B-Instruct
    python extract.py https://neurips.cc --backend vllm --base-url http://gpu-server:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from extractor.pipeline import extract_conference
from extractor.llm import DEFAULT_MODEL, DEFAULT_BACKEND


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured metadata from a conference website -> JSON",
    )
    parser.add_argument("url", help="Conference website URL")
    parser.add_argument(
        "--backend", "-b", default=DEFAULT_BACKEND, choices=["ollama", "vllm"],
        help=f"Inference backend (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="Server URL (default: auto per backend — :11434 for ollama, :8000 for vllm)",
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
        backend=args.backend,
        base_url=args.base_url,
    )

    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Result saved to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
