from __future__ import annotations

"""Groq LLM client wrapper with retry, model fallback, and robust JSON parsing."""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import groq
import httpx
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    Groq,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

import config

logger = logging.getLogger("crimevision.llm")


class LLMError(Exception):
    """Exception raised for errors in LLM execution, formatting, or connectivity."""


_client_lock = threading.Lock()
_client: Optional[Groq] = None


def is_configured() -> bool:
    """Return True if a non-empty GROQ_API_KEY is configured in environment or config."""
    key = os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY or ""
    return bool(key.strip())


def get_groq_client() -> Groq:
    """Retrieve or lazily initialize the Groq client singleton in a thread-safe manner."""
    global _client
    if not is_configured():
        raise LLMError("GROQ_API_KEY is not configured. LLM operations cannot proceed.")

    if _client is None:
        with _client_lock:
            if _client is None:
                key = (os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY or "").strip()
                try:
                    # Provide an explicit httpx.Client to ensure compatibility across httpx versions
                    _client = Groq(api_key=key, http_client=httpx.Client())
                except Exception as exc:
                    logger.error("Failed to initialize Groq client: %s", exc)
                    raise LLMError(f"Failed to initialize Groq client: {exc}") from exc
    return _client


def _execute_with_retry(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    response_format: Optional[Dict[str, str]] = None,
) -> str:
    """Execute chat completion with retries and fallback model."""
    client = get_groq_client()
    primary_model = model or os.getenv("GROQ_MODEL") or config.GROQ_MODEL
    fallback_model = config.GROQ_FALLBACK_MODEL

    models_to_attempt = [primary_model]
    if fallback_model and fallback_model != primary_model:
        models_to_attempt.append(fallback_model)

    last_error: Optional[Exception] = None

    for m_idx, current_model in enumerate(models_to_attempt):
        max_attempts = 3  # Initial try + 2 retries
        for attempt in range(max_attempts):
            try:
                kwargs: Dict[str, Any] = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content is None:
                    raise LLMError("Groq returned empty response content.")
                return content.strip()

            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
                last_error = exc
                if attempt < max_attempts - 1:
                    sleep_seconds = 2 ** attempt  # 1s on attempt 0, 2s on attempt 1
                    logger.warning(
                        "Transient error contacting Groq (model %s, attempt %d/%d): %s. Retrying in %ds...",
                        current_model,
                        attempt + 1,
                        max_attempts,
                        exc,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
                else:
                    logger.error("Max retries exceeded for model %s: %s", current_model, exc)

            except (NotFoundError, APIStatusError) as exc:
                last_error = exc
                # If 4xx or not found on primary model, jump to fallback model
                status_code = getattr(exc, "status_code", None)
                if status_code in (400, 404) or isinstance(exc, NotFoundError):
                    logger.warning(
                        "Model '%s' failed with error (%s). Trying fallback model if available...",
                        current_model,
                        exc,
                    )
                    break
                logger.error("Groq API error on model %s: %s", current_model, exc)
                raise LLMError(f"Groq API error: {exc}") from exc

            except Exception as exc:
                last_error = exc
                logger.error("Unexpected error invoking Groq model %s: %s", current_model, exc)
                raise LLMError(f"Unexpected Groq client error: {exc}") from exc

    raise LLMError(f"All Groq call attempts failed. Last error: {last_error}")


def chat_text(
    messages: List[Dict[str, str]],
    temperature: float = config.LLM_TEMPERATURE,
    max_tokens: int = 900,
    model: Optional[str] = None,
) -> str:
    """Send chat messages to Groq and return plain text response."""
    return _execute_with_retry(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=None,
    )


def chat_json(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 1200,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Send chat messages to Groq expecting JSON and robustly parse the output."""
    raw_text = _execute_with_retry(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    # Strip code block fences if present (e.g., ```json ... ```)
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Find boundaries of first '{' and last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise LLMError(f"Expected JSON object, got {type(parsed).__name__}")
        return parsed
    except Exception as exc:
        logger.error("Failed to parse JSON from Groq response: %s\nRaw output: %s", exc, raw_text)
        raise LLMError(f"Failed to parse JSON response: {exc}") from exc
