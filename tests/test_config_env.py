from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import get_settings


ENV_KEYS = (
    "AI_RACE_API_KEY",
    "AI_RACE_MODE",
    "AI_RACE_USE_LLM",
    "AI_RACE_BASE_URL",
    "AI_RACE_MODEL",
    "AI_RACE_TEMPERATURE",
    "AI_RACE_MAX_TOKENS",
    "AI_RACE_TIMEOUT",
    "AI_RACE_FAIL_OPEN",
)


class ConfigEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old = {key: os.environ.get(key) for key in ENV_KEYS}
        for key in ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in self._old.items():
            if value is not None:
                os.environ[key] = value

    def test_new_api_key_enables_llm_defaults(self) -> None:
        os.environ["AI_RACE_API_KEY"] = "test-key"
        with tempfile.TemporaryDirectory() as tmp:
            settings = get_settings(Path(tmp))
        self.assertEqual(settings.llm_api_key, "test-key")
        self.assertTrue(settings.llm_enabled)
        self.assertEqual(settings.mode, "llm_full_doc")
        self.assertEqual(settings.llm_model, "gpt-4.1")

    def test_ai_race_env_overrides_defaults(self) -> None:
        os.environ["AI_RACE_API_KEY"] = "test-key"
        os.environ["AI_RACE_MODE"] = "hybrid"
        os.environ["AI_RACE_USE_LLM"] = "false"
        os.environ["AI_RACE_MODEL"] = "custom-model"
        os.environ["AI_RACE_BASE_URL"] = "https://example.test/v1"
        os.environ["AI_RACE_MAX_TOKENS"] = "1234"
        os.environ["AI_RACE_TIMEOUT"] = "77"
        with tempfile.TemporaryDirectory() as tmp:
            settings = get_settings(Path(tmp))
        self.assertEqual(settings.mode, "hybrid")
        self.assertFalse(settings.llm_enabled)
        self.assertEqual(settings.llm_model, "custom-model")
        self.assertEqual(settings.llm_base_url, "https://example.test/v1")
        self.assertEqual(settings.llm_max_tokens, 1234)
        self.assertEqual(settings.llm_timeout, 77)


if __name__ == "__main__":
    unittest.main()
