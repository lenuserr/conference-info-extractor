"""
LLM-based extraction via local or cloud inference backends.

Uses chain-of-extraction: split into four focused passes to reduce
hallucinations. Each pass receives a category-specific page context
built by ``extractor.content_selection`` rather than the whole scraped
site, so the LLM only sees text that's actually relevant to the task.

Pass 1 — "other":     identity, dates, venue, deadlines, publication
Pass 2 — "topics":    research topics / themes / tracks / scope
Pass 3 — "speakers":  keynote / invited / plenary speakers
Pass 4 — "committee": program committee / chairs / organizers

Supported backends:
  - ollama  (default): Ollama API at http://localhost:11434
  - vllm:              vLLM OpenAI-compatible server at http://localhost:8000
  - claude:            Anthropic Claude API (requires ANTHROPIC_API_KEY env var)
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
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"

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

_OTHER_PROMPT = """\
You are a precise data extraction assistant. Extract conference metadata from the text below.
Return ONLY a valid JSON object — no markdown, no explanation, no extra text.

If a field cannot be determined from the text, use null. Only extract information that is \
EXPLICITLY stated in the text. Do not guess or infer values that are not clearly written.

Field descriptions:

- "full_name": The complete official name of the conference, e.g. "International Conference \
on Machine Learning", "IEEE Symposium on Security and Privacy". Do not include the year or \
edition number in this field.
- "acronym": The short abbreviation, e.g. "ICML", "IEEE S&P", "NeurIPS". Sometimes includes \
the year like "ICML 2026" — include only the abbreviation without the year.
- "edition_number": The numeric edition of the conference, e.g. 38 for "38th Annual \
Conference", 15 for "15th International Workshop". Use null if not mentioned.
- "start_date": The first day of the conference (not workshops or tutorials). Format: YYYY-MM-DD.
- "end_date": The last day of the conference. Format: YYYY-MM-DD. If the conference is a \
single day, end_date equals start_date.
- "city": The city where the conference takes place, e.g. "Vancouver", "Tokyo". If the \
conference is virtual/online with no physical location, use "Virtual Conference".
- "country": The country where the conference takes place, full English name, e.g. "Canada", \
"Japan", not "CA" or "JP". If virtual, use "Virtual Conference".
- "submission_deadline": The deadline for paper submissions (often called "paper submission \
deadline", "full paper deadline", "abstract deadline"). Format: YYYY-MM-DD. This is NOT the \
conference start date.
- "notification_date": The date when authors are notified of acceptance/rejection (often \
called "notification of acceptance", "author notification"). Format: YYYY-MM-DD.
- "camera_ready_date": The deadline for submitting the final camera-ready version of accepted \
papers (often called "camera-ready deadline", "final paper submission"). Format: YYYY-MM-DD.
- "publisher": The publisher of the conference proceedings, e.g. "Springer", "IEEE", "ACM", \
"Elsevier". Use null if not mentioned.
- "series": The publication series the proceedings appear in, e.g. "Lecture Notes in Computer \
Science", "ACM International Conference Proceeding Series", "Communications in Computer and \
Information Science". Use null if not mentioned.

Important date parsing rules:
- Conference websites use many different date formats. Examples: "May 21 ~ 22, 2026" (tilde \
instead of dash), "21-22 May 2026", "May 21st-22nd, 2026", "2026/05/21", "June 15, 2026 \
(extended)". Parse them all into YYYY-MM-DD.
- Do NOT confuse submission deadlines with conference dates — they are different things.

Required JSON structure:
{{
  "full_name": "string or null",
  "acronym": "string or null",
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

_TOPICS_PROMPT = """\
You are a precise data extraction assistant. From the conference text below,
extract ONLY the list of research topics, themes, tracks, or scope areas.
Return ONLY valid JSON — no markdown, no explanation, no extra text.

These are the subject areas that the conference calls for papers on, often
listed under headings like "Topics", "Scope", "Tracks", "Areas of Interest",
"Call for Papers", etc.

Each topic should be a short phrase (e.g. "machine learning", "network security",
"natural language processing"). Do not include full sentences or descriptions.

Only extract topics that are EXPLICITLY listed in the text. Do not invent or
generalize topics.

Return [] if no topics are listed in the text.

Required JSON structure:
{{
  "topics": ["string", ...]
}}

--- TEXT START ---
{text}
--- TEXT END ---

JSON:"""

_SPEAKERS_PROMPT = """\
You are a precise data extraction assistant. From the conference text below,
extract ONLY the list of keynote / invited / plenary speakers. Return ONLY
valid JSON — no markdown, no explanation, no extra text.

Field descriptions:

- "name": The speaker's full name as written on the website, e.g. "Yoshua Bengio", \
"Prof. Dr. Maria Garcia". Include titles (Prof., Dr.) only if prominently displayed.
- "affiliation": The speaker's organization — university, company, or research lab, \
e.g. "MIT", "Google DeepMind", "Max Planck Institute". This is the institution they \
are associated with, NOT their country. Use null if not mentioned.
- "country": The country of the speaker's affiliation (not nationality), e.g. "United States", \
"Germany", "Japan". Use full English country names. Use null if not mentioned on the page.

CRITICAL RULES:
1. Do NOT include program committee members, general chairs, track chairs,
   organizers, or reviewers here. Speakers and committee members are ALWAYS listed
   in different sections of the website. If someone is listed under "Committee",
   "Organizing", "Chairs", or similar headings — they are NOT a speaker.
2. Only extract people who are EXPLICITLY listed as keynote speakers, invited speakers,
   plenary speakers, or tutorial presenters WITH THEIR NAMES clearly stated.
3. If speakers are announced but names are not yet listed (e.g. "Keynote speakers
   to be announced", "To be announced soon", "TBA", "Coming soon", "To be confirmed"),
   return an EMPTY list [].
4. If the text does not mention any speakers at all, return an EMPTY list [].
5. It is MUCH BETTER to return an empty list than to guess or pick random names
   from the text. An empty list is a valid and correct answer.

Required JSON structure:
{{
  "keynote_speakers": [
    {{"name": "...", "affiliation": "...", "country": "..."}}
  ]
}}

--- TEXT START ---
{text}
--- TEXT END ---

JSON:"""

_COMMITTEE_PROMPT = """\
You are a precise data extraction assistant. From the conference text below,
extract ONLY the Program Committee / Organizing Committee / Chairs and related
roles. Return ONLY valid JSON — no markdown, no explanation, no extra text.

Field descriptions:

- "name": The person's full name as written on the website, e.g. "John Smith", \
"Prof. Maria Garcia".
- "affiliation": The person's organization — university, company, or research lab, \
e.g. "Stanford University", "Microsoft Research", "INRIA". This is the institution \
they are associated with, NOT their country. Use null if not mentioned.
- "country": The country of the person's affiliation (not nationality), e.g. "United States", \
"France", "China". Use full English country names. Use null if not mentioned.
- "role": The person's explicit role on the committee. Examples: "General Chair", \
"General Co-Chair", "Program Chair", "Program Co-Chair", "TPC Chair", "Track Chair", \
"Workshop Chair", "Publicity Chair", "Finance Chair", "Web Chair", "Local Arrangement Chair", \
"Steering Committee Member", "Advisory Board Member", "PC Member", "Reviewer". \
Use the role exactly as written on the website. Use null if the person is listed under \
a committee section but no specific role is given (e.g. just a list of names under \
"Program Committee").

Roles that belong here: Program Committee (PC) members, Technical Program
Committee (TPC) members, Organizing Committee members, General Chairs,
Program Chairs, Track Chairs, Workshop Chairs, Publicity Chairs, Reviewers,
Steering Committee, Advisory Board, and similar organizational roles.

CRITICAL: Do NOT include keynote / invited / plenary speakers here. Speakers
and committee members are ALWAYS listed in different sections of the website.
If someone is listed under "Keynote", "Invited Speakers", "Plenary", or
similar headings — they are NOT a committee member.

Return [] if no committee members are listed in the text.

Required JSON structure:
{{
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
            options={"temperature": temperature, "num_predict": 8192},
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
                "options": {"temperature": temperature, "num_predict": 8192},
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
                "max_tokens": 8192,
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
# Backend: Claude (Anthropic API)
# ---------------------------------------------------------------------------

def _call_claude(
    prompt: str,
    model: str,
    temperature: float = 0.1,
) -> Optional[str]:
    """Send a prompt to Claude via the Anthropic API.

    Requires the ``anthropic`` package and the ``ANTHROPIC_API_KEY``
    environment variable.  The *prompt* is sent as a single user message.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for the claude backend. "
            "Install it with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        # Extract text from the response
        text_parts = [
            block.text for block in message.content
            if hasattr(block, "text")
        ]
        return "\n".join(text_parts) if text_parts else None
    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
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
    elif backend == "claude":
        return _call_claude(prompt, model, temperature)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'ollama', 'vllm', or 'claude'.")


# ---------------------------------------------------------------------------
# JSON parsing from LLM output
# ---------------------------------------------------------------------------

def _parse_json_from_response(raw: str) -> Optional[Any]:
    """Extract a JSON value from an LLM response, tolerating markdown fences.

    Returns whatever ``json.loads`` produced — typically a dict, but a bare
    list is also possible when the model emits a raw array. Callers that
    require a specific shape must normalize the result themselves.
    """
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
    if backend == "claude":
        return "https://api.anthropic.com"  # not used directly, but kept for consistency
    return DEFAULT_OLLAMA_URL


def _run_pass(
    prompt_template: str,
    text: str,
    *,
    pass_name: str,
    model: str,
    backend: str,
    base_url: Optional[str],
    vllm_extra_args: Optional[List[str]],
    list_wrap_key: Optional[str] = None,
    prompts_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Format *prompt_template* with *text*, call the LLM, parse JSON.

    Normalizes the parsed result to a dict. Some models ignore the
    documented ``{"keynote_speakers": [...]}`` wrapper and emit a bare JSON
    array instead; when ``list_wrap_key`` is set, such an array is salvaged
    by wrapping it as ``{list_wrap_key: [...]}`` rather than being
    discarded. Anything else (None, non-dict, non-list) yields None.

    When ``prompts_dir`` is set, the fully rendered prompt and the raw LLM
    response are saved to that directory for debugging.
    """
    if base_url is None:
        base_url = get_default_url(backend)
    prompt = prompt_template.format(text=text)

    if prompts_dir:
        import os
        os.makedirs(prompts_dir, exist_ok=True)
        # Sanitize pass_name for filename: "1 (basic)" → "1_basic"
        safe_name = pass_name.replace(" ", "_").replace("(", "").replace(")", "")
        prompt_path = os.path.join(prompts_dir, f"{safe_name}_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        logger.debug("Saved prompt to %s", prompt_path)

    raw = _call_llm(
        prompt, model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args,
    )

    if prompts_dir and raw:
        response_path = os.path.join(prompts_dir, f"{safe_name}_response.txt")
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(raw)
        logger.debug("Saved LLM response to %s", response_path)

    result = _parse_json_from_response(raw)
    if result is None:
        logger.warning("Pass %s failed to produce valid JSON", pass_name)
        return None
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        if list_wrap_key is None:
            logger.warning(
                "Pass %s returned a bare JSON list but no wrap key is set — discarding",
                pass_name,
            )
            return None
        logger.warning(
            "Pass %s returned a bare JSON list; wrapping as %r (%d items)",
            pass_name, list_wrap_key, len(result),
        )
        return {list_wrap_key: result}
    logger.warning(
        "Pass %s returned unexpected JSON type %s — discarding",
        pass_name, type(result).__name__,
    )
    return None


def extract_other(
    text: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    base_url: Optional[str] = None,
    vllm_extra_args: Optional[List[str]] = None,
    prompts_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pass 1: identity, dates, venue, deadlines, publication.

    ``text`` should be the OTHER-category context built by
    ``PageSelector(site, Category.OTHER)``.
    """
    return _run_pass(
        _OTHER_PROMPT, text,
        pass_name="1 (other)",
        model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args,
        prompts_dir=prompts_dir,
    )


def extract_topics(
    text: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    base_url: Optional[str] = None,
    vllm_extra_args: Optional[List[str]] = None,
    prompts_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pass 2: research topics / themes / tracks / scope.

    ``text`` should be the TOPICS-category context built by
    ``PageSelector(site, Category.TOPICS)``.
    """
    return _run_pass(
        _TOPICS_PROMPT, text,
        pass_name="2 (topics)",
        model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args,
        prompts_dir=prompts_dir,
    )


def extract_speakers(
    text: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    base_url: Optional[str] = None,
    vllm_extra_args: Optional[List[str]] = None,
    prompts_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pass 3: keynote / invited / plenary speakers only.

    ``text`` should be the SPEAKERS-category context built by
    ``PageSelector(site, Category.SPEAKERS)``.
    """
    return _run_pass(
        _SPEAKERS_PROMPT, text,
        pass_name="3 (speakers)",
        model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args,
        list_wrap_key="keynote_speakers",
        prompts_dir=prompts_dir,
    )


def extract_committee(
    text: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    base_url: Optional[str] = None,
    vllm_extra_args: Optional[List[str]] = None,
    prompts_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pass 4: program committee / chairs / organizing committee.

    ``text`` should be the COMMITTEE-category context built by
    ``PageSelector(site, Category.COMMITTEE)``.
    """
    return _run_pass(
        _COMMITTEE_PROMPT, text,
        pass_name="4 (committee)",
        model=model, backend=backend, base_url=base_url,
        vllm_extra_args=vllm_extra_args,
        list_wrap_key="program_committee",
        prompts_dir=prompts_dir,
    )
