"""OpenAI-compatible local chat-completions client with fail-open behavior."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from medkg.config import Settings


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    raw: str | None = None


class LocalLLMClient:
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
            return LLMResult(ok=False, error="LLM disabled")
        try:
            text = self._chat_with_openai_package(system_prompt, user_prompt, max_tokens)
        except Exception:
            try:
                text = self._chat_with_http(system_prompt, user_prompt, max_tokens)
            except Exception as exc:  # noqa: BLE001
                return LLMResult(ok=False, error=str(exc))
        try:
            return LLMResult(ok=True, data=json.loads(text), raw=text)
        except json.JSONDecodeError as exc:
            return LLMResult(ok=False, error=f"invalid JSON from LLM: {exc}", raw=text)

    def _chat_with_openai_package(self, system_prompt: str, user_prompt: str, max_tokens: int | None) -> str:
        from openai import OpenAI  # type: ignore

        client = OpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key or "dummy",
        )
        response = client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.settings.llm_temperature,
            max_tokens=max_tokens or self.settings.llm_max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"

    def _chat_with_http(self, system_prompt: str, user_prompt: str, max_tokens: int | None) -> str:
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "response_format": {"type": "json_object"},
        }
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
        with urllib.request.urlopen(request, timeout=self.settings.llm_timeout) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"] or "{}"
