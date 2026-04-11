"""
LLM-based extraction via local inference backends (Ollama or vLLM).
No paid APIs — everything runs locally.

Uses chain-of-extraction: split into two passes to reduce hallucinations
in small models (7-10B parameters).

Pass 1 — "basic": name, acronym, dates, venue, deadlines, publisher
Pass 2 — "details": topics, keynote speakers, program committee members

Supported backends:
  - ollama  (default): Ollama API at http://localhost:11434
  - vllm:              vLLM OpenAI-compatible server at http://localhost:8000
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional

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
_vllm_current_args: tuple = ()  # extra CLI args the running server was started with

# Grace period between SIGTERM and SIGKILL escalation. vLLM's shutdown path
# (releasing CUDA contexts, tearing down distributed workers, freeing IPC
# shared memory) can easily take 15-30s for large models, so give it some room.
_VLLM_SIGTERM_GRACE_SEC = 30
_VLLM_SIGKILL_GRACE_SEC = 10

_POSIX = hasattr(os, "killpg") and hasattr(os, "getpgid")


def _is_server_up(base_url: str) -> bool:
    """Check if a vLLM server is responding."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _check_gpu_free() -> None:
    """
    Best-effort sanity check that GPU memory dropped after stopping vLLM.

    Calls ``nvidia-smi`` and logs a warning if any GPU still reports more
    than 1 GiB of used memory. Silently skipped if nvidia-smi is absent
    (non-GPU / non-NVIDIA dev boxes).
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return  # no nvidia-smi, or it misbehaved — nothing to do

    try:
        used_mib = [int(x.strip()) for x in out.decode().splitlines() if x.strip()]
    except ValueError:
        return
    leftover = [m for m in used_mib if m > 1024]
    if leftover:
        logger.warning(
            "After stopping vLLM, GPU memory still in use: %s MiB per GPU. "
            "A worker process may have survived — check `nvidia-smi` and kill "
            "leftover python processes manually if needed.",
            used_mib,
        )


def _stop_vllm_server() -> None:
    """
    Terminate the vLLM server process if we started one.

    vLLM forks worker subprocesses (engine core, API server, one per
    tensor-parallel rank, …). Signalling only the launcher leaves those
    workers orphaned and holding CUDA contexts. We start the server with
    ``start_new_session=True`` so every child lives in a fresh POSIX
    session / process group, then SIGTERM (and escalate to SIGKILL) the
    whole group here so that *all* GPU-holding processes actually exit.
    """
    global _vllm_process, _vllm_current_model, _vllm_current_args
    if _vllm_process is None:
        return

    pid = _vllm_process.pid
    logger.info("Stopping vLLM server (pid %d, model %s)...", pid, _vllm_current_model)

    pgid: Optional[int] = None
    if _POSIX:
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            pgid = None

    def _signal_group(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except ProcessLookupError:
                return
        # Windows / fallback: signal only the launcher
        try:
            if sig == signal.SIGKILL:
                _vllm_process.kill()
            else:
                _vllm_process.terminate()
        except ProcessLookupError:
            pass

    # Phase 1: polite SIGTERM to the whole group → let vLLM release CUDA cleanly
    _signal_group(signal.SIGTERM)
    try:
        _vllm_process.wait(timeout=_VLLM_SIGTERM_GRACE_SEC)
    except subprocess.TimeoutExpired:
        logger.warning(
            "vLLM did not exit within %ds of SIGTERM; escalating to SIGKILL",
            _VLLM_SIGTERM_GRACE_SEC,
        )
        # Phase 2: force-kill the group
        _signal_group(signal.SIGKILL)
        try:
            _vllm_process.wait(timeout=_VLLM_SIGKILL_GRACE_SEC)
        except subprocess.TimeoutExpired:
            logger.error(
                "vLLM server (pid %d) still alive after SIGKILL — "
                "manual intervention required",
                pid,
            )

    _vllm_process = None
    _vllm_current_model = None
    _vllm_current_args = ()

    # Give the NVIDIA driver a beat to reap the CUDA contexts of the dead
    # processes, then sanity-check that memory is actually free.
    time.sleep(2)
    _check_gpu_free()


def ensure_vllm_server(
    model: str,
    base_url: str,
    extra_args: Optional[List[str]] = None,
) -> None:
    """
    If no vLLM server is reachable at *base_url* **with the right model
    and CLI args**, launch ``vllm serve <model> [extra_args...]`` as a
    background process and wait until it's ready.  If the server is
    already running with a different model or different args, restart it.
    """
    global _vllm_process, _vllm_current_model, _vllm_current_args

    args_tuple = tuple(extra_args or [])

    # Server we started is running with the correct model + args — nothing to do
    if (
        _vllm_process is not None
        and _vllm_current_model == model
        and _vllm_current_args == args_tuple
        and _is_server_up(base_url)
    ):
        logger.debug("vLLM server already running with model %s and args %s", model, args_tuple)
        return

    # External server (not started by us) is running — use it as-is
    if _vllm_process is None and _is_server_up(base_url):
        logger.debug("External vLLM server detected at %s, using as-is", base_url)
        return

    # Need to (re)start: model/args changed or server is down
    if _vllm_process is not None:
        logger.info(
            "Restarting vLLM server: %s %s -> %s %s",
            _vllm_current_model, list(_vllm_current_args), model, list(args_tuple),
        )
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
        *args_tuple,
    ]
    logger.info("Starting vLLM server: %s", " ".join(cmd))
    print(f"Starting vLLM server: {' '.join(cmd)}")

    # start_new_session=True puts the launcher — and every worker it forks —
    # into a fresh POSIX session/process group, so _stop_vllm_server can
    # signal the whole group at once and reliably free GPU memory when we
    # switch models between benchmark entries. (No-op on Windows.)
    _vllm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    _vllm_current_model = model
    _vllm_current_args = args_tuple
    atexit.register(_stop_vllm_server)

    # Wait for the server to become ready
    max_wait = 900  # seconds — large models can take a while to load
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
extract ONLY the following three things. Return ONLY valid JSON — no markdown, no explanation.

1. "topics" — a list of research topics/themes/tracks of this conference.
   Return [] if none are found.
2. "keynote_speakers" — a list of keynote/invited/plenary speakers. Each entry:
   {{"name": "string", "affiliation": "string or null", "country": "string or null"}}
   Return [] if none are found.
3. "program_committee" — a list of members of the Program Committee (PC),
   Technical Program Committee (TPC), Organizing Committee, General Chairs,
   Program Chairs, Track Chairs, Reviewers, and similar roles. Each entry:
   {{"name": "string", "affiliation": "string or null", "country": "string or null", "role": "string or null"}}
   "role" is the person's role (e.g. "General Chair", "Program Chair", "PC Member").
   Do NOT duplicate keynote speakers here unless the site also explicitly lists
   them as part of the committee.
   Return [] if no committee members are found.

Required JSON structure:
{{
  "topics": ["string", ...],
  "keynote_speakers": [
    {{"name": "...", "affiliation": "...", "country": "..."}}
  ],
  "program_committee": [
    {{"name": "...", "affiliation": "...", "country": "...", "role": "..."}}
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
    vllm_extra_args: Optional[List[str]] = None,
) -> Optional[str]:
    """Send a prompt to vLLM's OpenAI-compatible completions endpoint.
    Automatically starts a vLLM server if one is not already running."""
    ensure_vllm_server(model, base_url, extra_args=vllm_extra_args)
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
    vllm_extra_args: Optional[List[str]] = None,
) -> Optional[str]:
    """Route to the appropriate backend."""
    if backend == "ollama":
        return _call_ollama(prompt, model, base_url, temperature)
    elif backend == "vllm":
        return _call_vllm(prompt, model, base_url, temperature, vllm_extra_args=vllm_extra_args)
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
    vllm_extra_args: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pass 1: extract basic metadata (name, dates, venue, deadlines, publisher).
    """
    if base_url is None:
        base_url = get_default_url(backend)
    prompt = _BASIC_PROMPT.format(text=text)
    raw = _call_llm(
        prompt, model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args,
    )
    result = _parse_json_from_response(raw)
    if result is None:
        logger.warning("Pass 1 (basic) failed to produce valid JSON")
    return result


def extract_details(
    text: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    base_url: Optional[str] = None,
    vllm_extra_args: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pass 2: extract topics, keynote speakers, and program committee members.
    """
    if base_url is None:
        base_url = get_default_url(backend)
    prompt = _DETAILS_PROMPT.format(text=text)
    raw = _call_llm(
        prompt, model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args,
    )
    result = _parse_json_from_response(raw)
    if result is None:
        logger.warning("Pass 2 (details) failed to produce valid JSON")
    return result
