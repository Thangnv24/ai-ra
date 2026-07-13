# AGENTS.md

## Project goal

Build a reliable Python system for the competition **Ontological Reasoning in Medical Knowledge Retrieval**.

The system processes Vietnamese/English clinical text, extracts medical concepts, classifies concept types, detects assertion/context, maps diagnoses to ICD-10 and drugs to RxNorm, validates offsets/schema, and writes the exact JSON files required by `problem/`.

## Current strategy

Use an LLM-required Track A pipeline:

1. Deterministic rule/lab extraction for high-precision proposals.
2. Required LLM entity proposal over document chunks in `llm_full_doc`.
3. Context detection for `isNegated`, `isFamily`, and `isHistorical`.
4. Local candidate retrieval from seed ontology plus `data/candidates`.
5. Required OpenAI-compatible LLM rerank/classification with strict JSON.
6. Deterministic schema and offset validation before writing files.

Do not build a slow open-ended ReAct agent for final inference. LLM output must always be parsed, constrained, and validated. Do not add local fail-open fallback: if LLM/API calls fail, the request must fail and log the error.

## Important files

- `.env`: primary runtime config for the internal LLM endpoint.
- `problem/statement1.md`: phase-1 statement and submission/scoring notes.
- `problem/statement2.md`: full task overview, labels, input/output definition, data and schedule.
- `problem/scoring.md`: scoring rules.
- `problem/sample_input_1.txt`, `problem/sample_input_2.txt`: official examples.
- `input/`: official `.txt` files.
- `output/`: generated `.json` predictions. Do not recreate legacy `outputs/`.
- `data/candidates/`: slim runtime ICD-10/RxNorm candidate KB.
- `scripts/build_slim_candidate_kb.py`: builds slim candidate artifacts from prepared raw sources.
- `main.py`: FastAPI server entry point.
- `tests/test.py`: E2E runner that sends input files to the local server in parallel. Default output is `output/out_put_DDMMYYYY/`.
- `src/services/pipeline.py`: end-to-end orchestration.
- `src/extraction/`: rule NER, lab extractor, sectioning, LLM entity proposal.
- `src/knowledge/`: ontology seed, slim candidate index, candidate retrieval, reasoning.
- `src/integrations/`: OpenAI-compatible client and strict JSON prompts.
- `src/core/schema.py`: output concept model and validation.

## Runtime config

Primary environment variables use unprefixed names:

- `API_KEY`
- `BASE_URL=http://10.221.58.70:8402`
- `MODEL=thangnv108`
- `TEMPERATURE=0`
- `MAX_TOKENS=4096`
- `TIMEOUT=120`
- `LOG_LEVEL=INFO`

Only these unprefixed variables are supported. Do not add alternate legacy environment-variable prefixes. Runtime mode, LLM enable/disable, and fail-open controls are intentionally not supported because the runtime has only the LLM-required flow.

## API contract

`POST /predict` accepts only:

```json
{"text": "..."}
```

Do not add request-level `id`, `mode`, `validate`, `model`, `max_tokens`, or other runtime controls. Runtime controls belong in `.env`.

## Manual commands

- Install:
  `pip install -r requirements.txt`
  `pip install -e .`
- Run API server:
  `python main.py`
- Run full input folder through server with dated output:
  `python tests/test.py`
- Run one file through server:
  `python tests/test.py input/1.txt`
- Run one file with custom output:
  `python tests/test.py input/1.txt --output-dir output/single_run`
- Run full input folder through server with explicit workers:
  `python tests/test.py input --workers 8`
- Create editable manual-gold draft from a prediction folder:
  `python scripts/gold_workflow.py init --prediction-dir output/out_put_DDMMYYYY --gold-dir data/gold_manual/review_YYYYMMDD_HHMMSS --pretty`
- Score predictions against edited manual gold:
  `python scripts/gold_workflow.py score --prediction-dir output/out_put_DDMMYYYY --gold-dir data/gold_manual/review_YYYYMMDD_HHMMSS`
- Build Docker image:
  `docker build -t ai-race-medical-kg:latest .`
- Run API with Docker Compose:
  `docker compose up --build ai-race-api`
- Run batch inference in Docker:
  `docker compose --profile run run --rm ai-race-runner`
- Create a manual review output folder:
  `output/review_YYYYMMDD_HHMMSS/`

For manual output optimization/review, create a new timestamped folder under `output/` and report that folder path. The user packages zip files manually.

## Source of truth

- Treat `problem/statement1.md`, `problem/statement2.md`, and `problem/scoring.md` as the source of truth for labels, fields, candidates, assertions, offsets, and allowed model/API behavior.
- Official inputs are Vietnamese free-form clinical snippets with English drug names, mixed lab abbreviations, numeric results, and ICD-10/RxNorm candidate requirements.
- Generated outputs are reproducible artifacts, not source-of-truth data.
- Keep offsets zero-based and end-exclusive. Every `position` must match `text[start:end]`.

## Data-source rules

- Runtime candidate retrieval uses local files under `data/candidates/`.
- Do not re-add raw full ICD-10/RxNorm dumps unless the user explicitly asks or a preparation task requires it.
- Downloader/build scripts belong in `scripts/` and are preparation-only.
- Before enabling a new knowledge source, verify source, license, version, row counts, checksums, and provenance.
- The OpenAI-compatible API path uses the configured internal LLM endpoint for Track A experimentation unless confirmed compliant for final submission.

## Runtime flow

File/folder server-runner flow:

```text
input/*.txt
  -> core.io.read_text
  -> tests/test.py
  -> FastAPI /predict on local server
  -> services.pipeline.MedicalKGPipeline
  -> extraction.MedicalNER
  -> extraction.labs.extract_lab_spans
  -> extraction.LLMEntityExtractor
  -> extraction.ContextDetector
  -> knowledge.CandidateRetriever/OntologyIndex/SlimCandidateIndex
  -> integrations.ApiLLMClient decision pass
  -> core.schema.validate_output
  -> output/*.json
```

Server flow:

```text
main.py
  -> api.server.app
  -> singleton MedicalKGPipeline
  -> /predict or /predict_batch
  -> tests/test.py writes and validates JSON
```

Data preparation flow:

```text
scripts/download_* -> data/raw
scripts/build_knowledge_base.py -> data/processed
scripts/build_indexes.py -> data/indexes
scripts/build_slim_candidate_kb.py -> data/candidates
```

Do not assume a KB exists because builder scripts exist. Check the actual files and manifest.

## Engineering rules

- Prioritize correctness, valid schema, exact offsets, and candidate quality.
- Keep changes compatible with the schema in `problem/`.
- Prefer existing package structure and local helper APIs.
- Keep edits scoped; do not refactor unrelated files.
- Cache expensive lookup/index work where appropriate.
- Do not emit free-form LLM output to final files.
- Keep runner validation focused on schema, exact offsets, and end-to-end behavior.
- If a user explicitly says not to change code, do not edit inference logic.
- If using temporary test outputs, delete them after reporting.

## Change workflow guide

Before changing inference logic, write down the flow being touched:

1. Request boundary: documentation, scripts/tests, data preparation, or inference logic.
2. Source of truth: relevant `problem/` docs and expected labels/fields.
3. Input-to-output map: `tests/test.py`, FastAPI endpoint, output path.
4. Runtime path: modules called from input read to schema validation.
5. Data/knowledge path: whether the change uses seed ontology, `data/candidates`, or build scripts.
6. Validation plan: smallest relevant unit/E2E test.

Report changed files, validation results, API/LLM availability, deleted temporary outputs, and remaining risks.
