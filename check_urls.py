#!/usr/bin/env python3
"""
Check all URLs from wikicfp_conferences_filtered.json:
fetch each URL and record the final URL after redirects, status, etc.

Usage:
    python check_urls.py
    python check_urls.py --input data/wikicfp_conferences_filtered.json --output data/url_check_results.json
    python check_urls.py --workers 20
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from extractor.scraper import _get

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_url(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch a single URL and return the result."""
    url = entry["website_url"]
    result = {
        **entry,
        "final_url": None,
        "status_code": None,
        "redirected": False,
        "error": None,
    }

    resp = _get(url)
    if resp:
        result["final_url"] = resp.url
        result["status_code"] = resp.status_code
        result["redirected"] = resp.url != url
    else:
        result["error"] = "Failed to fetch"

    return result


def main():
    parser = argparse.ArgumentParser(description="Check all conference URLs")
    parser.add_argument(
        "--input", "-i",
        default="data/wikicfp_conferences_filtered.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/url_check_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        entries = json.load(f)

    # Resume: load already processed results and skip those URLs
    results = []
    already_done = set()
    try:
        with open(args.output, encoding="utf-8") as f:
            results = json.load(f)
            already_done = {r["website_url"] for r in results}
            logger.info("Resuming: loaded %d already processed URLs from %s", len(already_done), args.output)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    remaining = [e for e in entries if e["website_url"] not in already_done]
    total = len(entries)
    logger.info("Total: %d, already done: %d, remaining: %d", total, len(already_done), len(remaining))

    if not remaining:
        logger.info("Nothing to do, all URLs already processed.")
        return

    done = len(already_done)
    errors = sum(1 for r in results if r.get("error"))
    redirects = sum(1 for r in results if r.get("redirected"))
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_url, entry): entry for entry in remaining}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)


            done += 1
            if result["error"]:
                errors += 1
            if result["redirected"]:
                redirects += 1

            if done % 100 == 0 or done == total:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d (%.0f%%) | errors: %d | redirects: %d | %.1f url/s | ETA: %.0fs",
                    done, total, 100 * done / total,
                    errors, redirects, rate, eta,
                )

                url_order = {e["website_url"]: i for i, e in enumerate(entries)}
                results.sort(key=lambda r: url_order.get(r["website_url"], 0))

                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time

    # Sort results in the same order as input
    url_order = {e["website_url"]: i for i, e in enumerate(entries)}
    results.sort(key=lambda r: url_order.get(r["website_url"], 0))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in results if r["status_code"] and r["status_code"] < 400)
    logger.info("Done in %.1fs", elapsed)
    logger.info("Results: %d ok, %d errors, %d redirects", ok, errors, redirects)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
