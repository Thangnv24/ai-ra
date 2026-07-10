# GPT/OpenAI-Compatible API

The project supports GPT-style chat completions through the OpenAI API or another remote OpenAI-compatible API endpoint. The documented flow only covers API calls.

Environment variables:

```text
AI_RACE_MODE=llm_full_doc
AI_RACE_USE_LLM=true
AI_RACE_BASE_URL=https://api.openai.com/v1
AI_RACE_MODEL=gpt-4.1
AI_RACE_API_KEY=your_api_key
AI_RACE_TEMPERATURE=0
AI_RACE_MAX_TOKENS=4096
AI_RACE_TIMEOUT=120
AI_RACE_FAIL_OPEN=true
```

The client posts to:

```text
{AI_RACE_BASE_URL}/chat/completions
```

The LLM receives deterministic mention proposals, local context, rule assertions, and retrieved candidate codes. It may return decisions such as keep/drop, final type, assertions, and selected candidates.

Safety rules:

- selected candidates must be a subset of retrieved candidates;
- invalid JSON is ignored;
- unavailable API falls back to baseline;
- final output is always validated deterministically;
- API `meta` is never written into competition JSON files.
