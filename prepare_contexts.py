#!/usr/bin/env python3
"""
Phase 1: Scrape conference sites and prepare LLM contexts.

Run this on a machine with internet access. It scrapes each site,
runs page selection for all 4 categories × 3 levels + brute-force,
and saves the ready-to-use contexts to disk.

The output directory can then be copied to an offline machine with
an LLM backend, where ``run_llm.py`` processes the contexts.

Usage:
    python prepare_contexts.py
    python prepare_contexts.py --input data/wikicfp_conferences_filtered_stage2.json --output prepared/
    python prepare_contexts.py --urls https://neurips.cc https://aaai.org/conference/aaai/aaai-25/
    python prepare_contexts.py --urls-file urls.txt --output prepared/ --workers 10
"""

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from extractor.scraper import SiteContent, fetch_conference_site
from extractor.content_selection import Category, PageSelector, build_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _sanitize_name(s: str) -> str:
    """Turn a URL into a safe filename."""
    s = s.replace("https://", "").replace("http://", "")
    s = re.sub(r"[/:.\-]+", "_", s)
    s = s.strip("_")
    return s[:80]


def _prepare_one_site(url: str) -> Dict[str, Any]:
    """
    Scrape a site and prepare all contexts for all categories.

    Returns a dict with all contexts needed by the LLM phase.
    """
    result: Dict[str, Any] = {
        "url": url,
        "error": None,
        "pages_fetched": 0,
        "categories": {},
    }

    site = fetch_conference_site(url)
    if not site.pages:
        result["error"] = "Could not fetch any pages"
        return result

    result["pages_fetched"] = len(site.pages)

    # For each category, prepare contexts for all 3 levels + brute-force
    for category in Category:
        selector = PageSelector(site, category)
        cat_data: Dict[str, Any] = {}

        # Level 1: navigation
        pages_1 = selector.select_by_navigation()
        cat_data["L1"] = {
            "context": build_context(pages_1) if pages_1 else "",
            "page_count": len(pages_1),
            "urls": [p.url for p in pages_1],
        }

        # Level 2: content
        pages_2 = selector.select_by_content()
        cat_data["L2"] = {
            "context": build_context(pages_2) if pages_2 else "",
            "page_count": len(pages_2),
            "urls": [p.url for p in pages_2],
        }

        # Level 3: remaining
        pages_3 = selector.select_remaining()
        cat_data["L3"] = {
            "context": build_context(pages_3) if pages_3 else "",
            "page_count": len(pages_3),
            "urls": [p.url for p in pages_3],
        }

        # Brute-force: all pages
        all_pages = list(site.pages)
        cat_data["bruteforce"] = {
            "context": build_context(all_pages),
            "page_count": len(all_pages),
            "urls": [p.url for p in all_pages],
        }

        result["categories"][category.value] = cat_data

    return result


def _read_lines(path: str) -> List[str]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Scrape sites and prepare LLM contexts (Phase 1)",
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="Input JSON file with conference entries (must have 'website_url' field)",
    )
    parser.add_argument(
        "--urls", nargs="+", default=None,
        help="List of URLs to process",
    )
    parser.add_argument(
        "--urls-file", default=None,
        help="File with URLs, one per line",
    )
    parser.add_argument(
        "--output", "-o", default="prepared",
        help="Output directory (default: prepared/)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=10,
        help="Number of parallel workers for scraping (default: 10)",
    )
    args = parser.parse_args()

    # Resolve URLs
    urls: List[str] = []
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            entries = json.load(f)
        urls = [e["website_url"] for e in entries if e.get("website_url")]
    elif args.urls_file:
        urls = _read_lines(args.urls_file)
    elif args.urls:
        urls = args.urls
    else:
        print("ERROR: provide --input, --urls, or --urls-file")
        return

    os.makedirs(args.output, exist_ok=True)

    # Resume: check what's already done
    already_done = set()
    for name in os.listdir(args.output):
        if name.endswith(".json"):
            already_done.add(name)
    remaining = [
        u for u in urls
        if f"{_sanitize_name(u)}.json" not in already_done
    ]

    logger.info(
        "Total: %d, already done: %d, remaining: %d",
        len(urls), len(already_done), len(remaining),
    )

    if not remaining:
        logger.info("Nothing to do.")
        return

    done = len(already_done)
    errors = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_prepare_one_site, url): url
            for url in remaining
        }

        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("Unexpected error for %s: %s", url, exc)
                result = {"url": url, "error": str(exc), "pages_fetched": 0, "categories": {}}

            # Save individual result
            filename = f"{_sanitize_name(url)}.json"
            path = os.path.join(args.output, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)

            done += 1
            if result.get("error"):
                errors += 1

            if done % 50 == 0 or done == len(urls):
                elapsed = time.time() - start_time
                processed = done - len(already_done)
                rate = processed / elapsed if elapsed > 0 else 0
                left = len(urls) - done
                eta = left / rate if rate > 0 else 0
                logger.info(
                    "%d/%d (%.0f%%) | errors: %d | %.1f url/s | ETA: %.0fs",
                    done, len(urls), 100 * done / len(urls),
                    errors, rate, eta,
                )

    elapsed = time.time() - start_time
    logger.info("Done in %.1fs. Processed: %d, errors: %d", elapsed, done, errors)
    logger.info("Output: %s/", args.output)


if __name__ == "__main__":
    main()
