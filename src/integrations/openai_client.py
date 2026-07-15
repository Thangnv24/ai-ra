"""OpenAI-compatible chat-completions API client."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

from core.config import Settings

logger = logging.getLogger("ai_race.llm")


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    raw: str | None = None


class MaxTokensExceededError(Exception):
    """Raised when LLM hits max_tokens limit and returns empty content."""


class ApiLLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.llm_enabled

    def available(self) -> bool:
        if not self.enabled:
            return False
        result = self.chat_json(
            "Return JSON only.",
            "Return {\"decisions\": []}.",
            max_tokens=32,
        )
        return result.ok

    def chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> LLMResult:
        if not self.enabled:
            logger.error("llm_call_blocked reason=disabled")
            return LLMResult(ok=False, error="LLM disabled")
        start = time.perf_counter()
        effective_max_tokens = max_tokens or self.settings.llm_max_tokens
        logger.info(
            "llm_call_start base_url=%s model=%s max_tokens=%s prompt_chars=%s",
            self.settings.llm_base_url,
            self.settings.llm_model,
            effective_max_tokens,
            len(system_prompt) + len(user_prompt),
        )
        text: str | None = None
        last_exc: Exception | None = None
        for attempt in range(3):
            current_max_tokens = (
                effective_max_tokens * (2**attempt)
                if attempt > 0
                else effective_max_tokens
            )
            if attempt > 0:
                logger.info("llm_retry attempt=%d max_tokens=%s", attempt, current_max_tokens)
            try:
                text = self._chat_with_openai_package(
                    system_prompt,
                    user_prompt,
                    current_max_tokens,
                )
            except MaxTokensExceededError:
                last_exc = MaxTokensExceededError(
                    f"max_tokens={current_max_tokens} still insufficient"
                )
                continue
            except Exception as package_exc:  # noqa: BLE001
                logger.warning(
                    "llm_openai_package_failed base_url=%s model=%s seconds=%.6f error=%s",
                    self.settings.llm_base_url,
                    self.settings.llm_model,
                    time.perf_counter() - start,
                    package_exc,
                )
                try:
                    text = self._chat_with_http(
                        system_prompt,
                        user_prompt,
                        current_max_tokens,
                    )
                except MaxTokensExceededError:
                    last_exc = MaxTokensExceededError(
                        f"max_tokens={current_max_tokens} still insufficient"
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    break
            last_exc = None
            break
        if last_exc is not None or text is None:
            error = last_exc or RuntimeError("LLM request failed without response text")
            logger.error(
                "llm_call_failed base_url=%s model=%s seconds=%.6f error=%s",
                self.settings.llm_base_url,
                self.settings.llm_model,
                time.perf_counter() - start,
                error,
            )
            return LLMResult(ok=False, error=str(error))
        try:
            data = _load_json_object(text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "llm_invalid_json base_url=%s model=%s seconds=%.6f raw_chars=%s error=%s",
                self.settings.llm_base_url,
                self.settings.llm_model,
                time.perf_counter() - start,
                len(text),
                exc,
            )
            return LLMResult(ok=False, error=f"invalid JSON from LLM: {exc}", raw=text)
        logger.info(
            "llm_call_complete base_url=%s model=%s seconds=%.6f raw_chars=%s",
            self.settings.llm_base_url,
            self.settings.llm_model,
            time.perf_counter() - start,
            len(text),
        )
        return LLMResult(ok=True, data=data, raw=text)

    def _chat_with_openai_package(self, system_prompt: str, user_prompt: str, max_tokens: int | None) -> str:
        import httpx
        from openai import OpenAI  # type: ignore

        last_error: Exception | None = None
        for use_json_mode in (True, False):
            start = time.perf_counter()
            try:
                with httpx.Client(
                    trust_env=False,
                    verify=False,
                    timeout=httpx.Timeout(self.settings.llm_timeout),
                ) as http_client:
                    client = OpenAI(
                        base_url=self.settings.llm_base_url,
                        api_key=self.settings.llm_api_key or "dummy",
                        http_client=http_client,
                    )
                    kwargs = _chat_payload(self.settings, system_prompt, user_prompt, max_tokens)
                    if use_json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    response = client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                msg = choice.message
                finish_reason = choice.finish_reason
                logger.info(
                    "llm_response_detail content_type=%s content_len=%s finish_reason=%s refusal=%s",
                    type(msg.content).__name__,
                    len(msg.content or ""),
                    finish_reason,
                    getattr(msg, "refusal", None),
                )
                if finish_reason == "length" and not msg.content:
                    raise MaxTokensExceededError(
                        f"finish_reason=length with empty content at max_tokens={max_tokens}"
                    )
                logger.info(
                    "llm_transport_ok transport=openai_package json_mode=%s seconds=%.6f",
                    use_json_mode,
                    time.perf_counter() - start,
                )
                return msg.content or "{}"
            except MaxTokensExceededError as exc:
                logger.warning(
                    "llm_max_tokens_exceeded transport=openai_package json_mode=%s seconds=%.6f error=%s",
                    use_json_mode,
                    time.perf_counter() - start,
                    exc,
                )
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "llm_transport_failed transport=openai_package json_mode=%s seconds=%.6f error=%s",
                    use_json_mode,
                    time.perf_counter() - start,
                    exc,
                )
                last_error = exc
        raise last_error or RuntimeError("OpenAI-compatible request failed")

    def _chat_with_http(self, system_prompt: str, user_prompt: str, max_tokens: int | None) -> str:
        last_error: Exception | None = None
        for use_json_mode in (True, False):
            start = time.perf_counter()
            try:
                payload = _chat_payload(self.settings, system_prompt, user_prompt, max_tokens)
                if use_json_mode:
                    payload["response_format"] = {"type": "json_object"}
                text = self._post_chat_payload(payload)
                logger.info(
                    "llm_transport_ok transport=urllib json_mode=%s seconds=%.6f",
                    use_json_mode,
                    time.perf_counter() - start,
                )
                return text
            except MaxTokensExceededError as exc:
                logger.warning(
                    "llm_max_tokens_exceeded transport=urllib json_mode=%s seconds=%.6f error=%s",
                    use_json_mode,
                    time.perf_counter() - start,
                    exc,
                )
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "llm_transport_failed transport=urllib json_mode=%s seconds=%.6f error=%s",
                    use_json_mode,
                    time.perf_counter() - start,
                    exc,
                )
                last_error = exc
        raise last_error or RuntimeError("HTTP LLM request failed")

    def _post_chat_payload(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.settings.llm_base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.settings.llm_api_key or 'dummy'}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=self.settings.llm_timeout) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content")
        finish_reason = choice.get("finish_reason")
        logger.info(
            "llm_response_detail content_type=%s content_len=%s finish_reason=%s refusal=%s",
            type(content).__name__,
            len(content or ""),
            finish_reason,
            message.get("refusal"),
        )
        if finish_reason == "length" and not content:
            raise MaxTokensExceededError(
                f"finish_reason=length with empty content at max_tokens={payload.get('max_tokens')}"
            )
        return content or "{}"


def _chat_payload(settings: Settings, system_prompt: str, user_prompt: str, max_tokens: int | None) -> dict[str, Any]:
    return {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": settings.llm_temperature,
        "max_tokens": max_tokens or settings.llm_max_tokens,
    }


def _load_json_object(text: str) -> dict[str, Any]:
    candidates = []
    stripped = text.strip()
    candidates.append(stripped)
    if stripped.startswith("```"):
        candidates.append(_strip_code_fence(stripped))
    embedded = _extract_balanced_object(stripped)
    if embedded:
        candidates.append(embedded)

    seen: set[str] = set()
    errors: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(data, dict):
            raise ValueError("LLM JSON root must be an object")
        return data
    raise ValueError(errors[0] if errors else "no JSON object found")


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].lstrip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _extract_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
