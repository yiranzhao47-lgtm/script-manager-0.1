"""
LLM API client with layered tenacity retry strategy.

Retry policy (per exception type):
  RateLimitError (429)     → exponential backoff, up to max_attempts
  APIConnectionError       → exponential backoff (transient network fault)
  APIStatusError  >= 500   → exponential backoff, up to max_attempts
  AuthenticationError(401) → immediate LLMCallError — never retry
  BadRequestError (400)    → immediate LLMCallError — never retry
  Any code in no_retry_codes → immediate LLMCallError — never retry

Usage
─────
    from src.utils.llm_client import LLMClient, extract_json

    client = LLMClient(cfg)
    raw = client.complete(system="...", user="...")
    data = extract_json(raw)
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
from typing import Optional

from tenacity import (
    RetryError,
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Exceptions
# ══════════════════════════════════════════════════════════════════════════════


class LLMCallError(RuntimeError):
    """
    Raised when an LLM call fails permanently — either a non-retriable HTTP
    error (4xx) or exhausted retry budget after transient errors.
    """


# ══════════════════════════════════════════════════════════════════════════════
#  JSON extraction helper (used by both metadata and execution layers)
# ══════════════════════════════════════════════════════════════════════════════


def extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from an LLM response string.

    Handles in order:
      1. Plain JSON response (most common for low-temperature calls)
      2. JSON wrapped in markdown code fences  (```json … ``` or ``` … ```)
      3. JSON object embedded in surrounding prose (outermost brace pair)

    Raises ValueError if no valid JSON object can be recovered.
    """
    text = text.strip()

    # ── 1. Direct parse ───────────────────────────────────────────────────
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ── 2. Strip markdown fences ──────────────────────────────────────────
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # ── 3. Locate outermost JSON object by brace matching ─────────────────
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1  # keep scanning for next candidate object

    raise ValueError(
        f"No valid JSON object found in LLM response.  "
        f"First 400 chars: {text[:400]!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LLMClient
# ══════════════════════════════════════════════════════════════════════════════


class LLMClient:
    """
    Thin OpenAI-compatible chat client with structured retry logic.

    Reads all configuration from the pipeline cfg dict; no global state.
    The underlying openai.OpenAI instance is lazy-initialised on first call.
    """

    def __init__(self, cfg: dict) -> None:
        llm_cfg = cfg.get("execution", {}).get("llm", {})
        self._model: str = llm_cfg.get("model", "gpt-4o")
        self._base_url: Optional[str] = llm_cfg.get("base_url") or None
        self._default_max_tokens: int = int(llm_cfg.get("max_tokens", 8000))

        retry_cfg = cfg.get("execution", {}).get("retry", {})
        self._max_attempts: int = int(retry_cfg.get("max_attempts", 5))
        self._wait_min: float = float(retry_cfg.get("wait_min_sec", 1.0))
        self._wait_max: float = float(retry_cfg.get("wait_max_sec", 60.0))
        self._no_retry_codes: frozenset[int] = frozenset(
            retry_cfg.get("no_retry_codes", [400, 401])
        )

        api_key_env: str = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
        self._api_key: str = os.environ.get(api_key_env, "")
        if not self._api_key:
            logger.warning(
                "LLMClient: env var '%s' is empty — API calls will fail with 401",
                api_key_env,
            )

        self._openai_client = None  # lazy-init on first complete() call

        # FinOps usage ledger — updated after every successful API call
        self._ledger: dict = {
            "total":     {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            "by_module": {},
        }
        self._ledger_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        module_name: str = "default",
    ) -> str:
        """
        Send one chat completion and return the response text.

        Applies the configured retry policy automatically.
        Raises LLMCallError on permanent failure (non-retriable error or
        exhausted retry budget).

        Parameters
        ----------
        json_mode:
            When True, passes ``response_format={"type": "json_object"}`` to
            the API.  Requires the model to support JSON mode (e.g. DeepSeek,
            GPT-4o).  The system prompt must instruct the model to return JSON.
        module_name:
            Logical caller tag used to break down token costs per pipeline
            stage in the FinOps ledger (e.g. "Subtitle_Refine", "Map_Extract").
        """
        limit = max_tokens or self._default_max_tokens

        try:
            for attempt in Retrying(
                stop=stop_after_attempt(self._max_attempts),
                wait=wait_exponential(
                    multiplier=2,
                    min=self._wait_min,
                    max=self._wait_max,
                ),
                retry=retry_if_exception(self._should_retry),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    return self._call_once(
                        system, user, limit,
                        json_mode=json_mode,
                        module_name=module_name,
                    )
        except RetryError as exc:
            raise LLMCallError(
                f"LLM call failed after {self._max_attempts} attempts. "
                f"Last error: {exc}"
            ) from exc

        # unreachable, but satisfies type checkers
        raise LLMCallError("Unexpected exit from retry loop")  # pragma: no cover

    def get_ledger_data(self) -> dict:
        """
        Return a deep-copy snapshot of the current FinOps usage ledger.

        Schema::

            {
              "total": {"input_tokens": int, "output_tokens": int, "calls": int},
              "by_module": {
                "<module_name>": {"input_tokens": int, "output_tokens": int, "calls": int},
                ...
              }
            }
        """
        return copy.deepcopy(self._ledger)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _call_once(
        self,
        system: str,
        user: str,
        max_tokens: int,
        json_mode: bool = False,
        module_name: str = "default",
    ) -> str:
        client = self._get_client()
        kwargs: dict = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.1,  # low temp for deterministic structured output
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""

        # FinOps: record actual usage from the API response
        usage = getattr(resp, "usage", None)
        in_tok  = int(getattr(usage, "prompt_tokens",     0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        self._record_usage(module_name, in_tok, out_tok)

        logger.debug(
            "LLM response — model=%s  module=%s  in=%d  out=%d  chars=%d",
            self._model, module_name, in_tok, out_tok, len(content),
        )
        return content

    def _record_usage(self, module_name: str, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token counts into the FinOps ledger (thread-safe)."""
        with self._ledger_lock:
            t = self._ledger["total"]
            t["input_tokens"]  += input_tokens
            t["output_tokens"] += output_tokens
            t["calls"]         += 1

            m = self._ledger["by_module"].setdefault(
                module_name,
                {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            )
            m["input_tokens"]  += input_tokens
            m["output_tokens"] += output_tokens
            m["calls"]         += 1

    def _get_client(self):
        if self._openai_client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "openai package required: pip install 'openai>=1.30.0'"
                ) from exc
            kwargs: dict = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._openai_client = openai.OpenAI(**kwargs)
        return self._openai_client

    def _should_retry(self, exc: BaseException) -> bool:
        """
        Return True if the exception is transient and worth retrying.
        Raise LLMCallError directly for permanent errors (bypasses retry
        budget entirely — tenacity propagates the new exception immediately).
        """
        try:
            import openai
        except ImportError:
            return False

        if isinstance(exc, openai.RateLimitError):
            logger.warning("429 Rate limit — backing off before retry")
            return True

        if isinstance(exc, openai.APIConnectionError):
            return True

        if isinstance(exc, openai.APIStatusError):
            code = exc.status_code
            if code in self._no_retry_codes:
                # Permanent error (auth failure, malformed request) — fail fast
                raise LLMCallError(
                    f"Non-retriable API error {code}: {exc.message}"
                ) from exc
            return code >= 500

        return False
