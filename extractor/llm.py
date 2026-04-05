"""
LLM-based extraction via local inference backends (Ollama or vLLM).
No paid APIs — everything runs locally.

Uses chain-of-extraction: split into two passes to reduce hallucinations
in small models (7-10B parameters).

Pass 1 — "basic": name, acronym, dates, venue, deadlines, publisher
Pass 2 — "details": topics, keynote speakers

Supported backends:
  - ollama  (default): Ollama API at http://localhost:11434
  - vllm:              vLLM OpenAI-compatible server at http://localhost:8000
"""

from __future__ import annotations

import atexit
import json
import logging
import re
import subprocess
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "mistral:latest"
DEFAULT_BACKEND = "ollama"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_VLLM_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# vLLM server auto-management
# ---------------------------------------------------------------------------

_vllm_process: Optional[subprocess.Popen] = None
_vllm_current_model: Optional[str] = None  # model the running server is serving


def _is_server_up(base_url: str) -> bool:
    """Check if a vLLM server is responding."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _stop_vllm_server() -> None:
    """Terminate the vLLM server process if we started one."""
    global _vllm_process, _vllm_current_model
    if _vllm_process is not None:
        logger.info("Stopping vLLM server (pid %d, model %s)...", _vllm_process.pid, _vllm_current_model)
        _vllm_process.terminate()
        try:
            _vllm_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _vllm_process.kill()
        _vllm_process = None
        _vllm_current_model = None


def ensure_vllm_server(model: str, base_url: str) -> None:
    """
    If no vLLM server is reachable at *base_url* **with the right model**,
    launch ``vllm serve <model>`` as a background process and wait until
    it's ready.  If the server is already running with a different model,
    restart it.
    """
    global _vllm_process, _vllm_current_model

    # Server we started is running with the correct model — nothing to do
    if _vllm_process is not None and _vllm_current_model == model and _is_server_up(base_url):
        logger.debug("vLLM server already running with model %s", model)
        return

    # External server (not started by us) is running — use it as-is
    if _vllm_process is None and _is_server_up(base_url):
        logger.debug("External vLLM server detected at %s, using as-is", base_url)
        return

    # Need to (re)start: either different model or server is down
    if _vllm_process is not None:
        logger.info("Restarting vLLM server: %s -> %s", _vllm_current_model, model)
        _stop_vllm_server()

    # Parse host/port from base_url
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    host = parsed.hostname or "0.0.0.0"
    port = str(parsed.port or 8000)

    cmd = [
        "vllm", "serve", model,
        "--host", host,
        "--port", port,
    ]
    logger.info("Starting vLLM server: %s", " ".join(cmd))
    print(f"Starting vLLM server for model '{model}' on {base_url} ...")

    _vllm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _vllm_current_model = model
    atexit.register(_stop_vllm_server)

    # Wait for the server to become ready
    max_wait = 300  # seconds — large models can take a while to load
    poll_interval = 3
    waited = 0
    while waited < max_wait:
        # Check if process crashed
        ret = _vllm_process.poll()
        if ret is not None:
            stderr_output = _vllm_process.stderr.read().decode(errors="replace")[-2000:]
            _vllm_process = None
            raise RuntimeError(
                f"vLLM server exited with code {ret}.\n"
                f"stderr (last 2000 chars):\n{stderr_output}"
            )
        if _is_server_up(base_url):
            logger.info("vLLM server ready after %ds", waited)
            print(f"vLLM server ready (took {waited}s)")
            return
        time.sleep(poll_interval)
        waited += poll_interval
        if waited % 30 == 0:
            print(f"  Still waiting for vLLM server... ({waited}s)")

    # Timeout — kill and raise
    _stop_vllm_server()
    raise RuntimeError(
        f"vLLM server did not become ready within {max_wait}s. "
        f"Make sure the model '{model}' is available and you have enough GPU memory."
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BASIC_PROMPT = """\
You are a precise data extraction assistant. Extract conference metadata from the text below.
Return ONLY a valid JSON object — no markdown, no explanation, no extra text.

If a field cannot be determined from the text, use null for scalars and [] for arrays.
Dates must be in YYYY-MM-DD format. Country must be the full English name (e.g. "Spain", not "ES").
edition_number is the numeric edition (e.g. 38 for "38th Annual Conference"), or null.

Required JSON structure:
{{
  "full_name": "string",
  "acronym": "string",
  "edition_number": int or null,
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
  "city": "string or null",
  "country": "string or null",
  "submission_deadline": "YYYY-MM-DD or null",
  "notification_date": "YYYY-MM-DD or null",
  "camera_ready_date": "YYYY-MM-DD or null",
  "publisher": "string or null",
  "series": "string or null"
}}

--- TEXT START ---
{text}
--- TEXT END ---

JSON:"""

_DETAILS_PROMPT = """\
You are a precise data extraction assistant. From the conference text below,
extract ONLY the following two things. Return ONLY valid JSON — no markdown, no explanation.

1. "topics" — a list of research topics/themes/tracks of this conference.
   Return [] if none are found.
2. "keynote_speakers" — a list of keynote/invited speakers. Each entry:
   {{"name": "string", "affiliation": "string or null", "country": "string or null"}}
   Return [] if none are found.

Required JSON structure:
{{
  "topics": ["string", ...],
  "keynote_speakers": [
    {{"name": "...", "affiliation": "...", "country": "..."}}
  ]
}}

--- TEXT START ---
{text}
--- TEXT END ---

JSON:"""


# ---------------------------------------------------------------------------
# Backend: Ollama
# ---------------------------------------------------------------------------

def _call_ollama(
    prompt: str,
    model: str,
    base_url: str,
    temperature: float = 0.1,
) -> Optional[str]:
    """Send a prompt to Ollama and return the raw text response."""
    # Try the ollama Python library first
    try:
        import ollama as _ollama_lib
        client = _ollama_lib.Client(host=base_url)
        response = client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": 2048},
        )
        return response.get("response", "")
    except ImportError:
        logger.debug("ollama library not installed, falling back to HTTP")
    except Exception as exc:
        logger.warning("ollama library call failed: %s — falling back to HTTP", exc)

    # Fallback: direct HTTP
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 2048},
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as exc:
        logger.error("Ollama HTTP call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Backend: vLLM (OpenAI-compatible /v1/completions)
# ---------------------------------------------------------------------------

def _call_vllm(
    prompt: str,
    model: str,
    base_url: str,
    temperature: float = 0.1,
) -> Optional[str]:
    """Send a prompt to vLLM's OpenAI-compatible completions endpoint.
    Automatically starts a vLLM server if one is not already running."""
    ensure_vllm_server(model, base_url)
    try:
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 2048,
                "temperature": temperature,
                "stop": ["\n\n\n"],
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("text", "")
        return None
    except Exception as exc:
        logger.error("vLLM call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Unified call dispatcher
# ---------------------------------------------------------------------------

def _call_llm(
    prompt: str,
    model: str,
    backend: str,
    base_url: str,
    temperature: float = 0.1,
) -> Optional[str]:
    """Route to the appropriate backend."""
    if backend == "ollama":
        return _call_ollama(prompt, model, base_url, temperature)
    elif backend == "vllm":
        return _call_vllm(prompt, model, base_url, temperature)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'ollama' or 'vllm'.")


# ---------------------------------------------------------------------------
# JSON parsing from LLM output
# ---------------------------------------------------------------------------

def _parse_json_from_response(raw: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response, tolerating markdown fences etc."""
    if not raw:
        return None

    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Find first balanced { ... } block
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_default_url(backend: str) -> str:
    """Return the default server URL for a given backend."""
    if backend == "vllm":
        return DEFAULT_VLLM_URL
    return DEFAULT_OLLAMA_URL


def extract_basic(
    text: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pass 1: extract basic metadata (name, dates, venue, deadlines, publisher).
    """
    if base_url is None:
        base_url = get_default_url(backend)
    prompt = _BASIC_PROMPT.format(text=text[:8000])
    raw = _call_llm(prompt, model=model, backend=backend, base_url=base_url)
    result = _parse_json_from_response(raw)
    if result is None:
        logger.warning("Pass 1 (basic) failed to produce valid JSON")
    return result


def extract_details(
    text: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pass 2: extract topics and keynote speakers.
    """
    if base_url is None:
        base_url = get_default_url(backend)
    prompt = _DETAILS_PROMPT.format(text=text[:8000])
    raw = _call_llm(prompt, model=model, backend=backend, base_url=base_url)
    result = _parse_json_from_response(raw)
    if result is None:
        logger.warning("Pass 2 (details) failed to produce valid JSON")
    return result
