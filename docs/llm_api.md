# Local LLM API

The project supports GPT-style chat completions through a local OpenAI-compatible server.

Environment variables:

```text
MEDKG_MODE=hybrid
MEDKG_LLM_ENABLED=true
MEDKG_LLM_BACKEND=openai_compatible
MEDKG_LLM_BASE_URL=http://localhost:8000/v1
MEDKG_LLM_MODEL=your-local-9b-model
MEDKG_LLM_API_KEY=dummy
MEDKG_LLM_TEMPERATURE=0
MEDKG_LLM_MAX_TOKENS=2048
MEDKG_LLM_TIMEOUT=60
MEDKG_LLM_FAIL_OPEN=true
```

The client posts to:

```text
{MEDKG_LLM_BASE_URL}/chat/completions
```

The LLM receives deterministic mention proposals, local context, rule assertions, and retrieved candidate codes. It may return decisions such as keep/drop, final type, assertions, and selected candidates.

Safety rules:

- selected candidates must be a subset of retrieved candidates;
- invalid JSON is ignored;
- unavailable local LLM falls back to baseline;
- final output is always validated deterministically;
- API `meta` is never written into competition JSON files.

Example local vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server ^
  --model YOUR_LOCAL_9B_MODEL ^
  --host 0.0.0.0 ^
  --port 8000
```
