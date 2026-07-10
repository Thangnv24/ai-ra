# Current Project Audit

## Structure

The repository contains a hybrid extraction/retrieval pipeline under `src/`, operational scripts under `scripts`, official inputs under `input`, organizer statements under `problem`, generated outputs under `output`, and submission artifacts under `submission`.

## Current Pipeline

The baseline reads UTF-8 clinical text, extracts medical spans with rule-based NER, detects assertions with context triggers, retrieves local ICD-10/RxNorm candidates, validates exact offsets, and writes competition JSON.

Key modules:

- `src/extraction/ner.py`: rule-based mention extraction.
- `src/extraction/context.py`: assertion detection.
- `src/knowledge/ontology.py`: seed ontology index loading.
- `src/knowledge/candidates.py`: slim local candidate KB loading.
- `src/knowledge/retrieval.py`: cached candidate lookup.
- `src/services/pipeline.py`: end-to-end orchestration.
- `src/core/schema.py`: output validation.

## Server Entrypoint

`src/api/server.py` exposes FastAPI endpoints. `main.py` starts the app. The server keeps one pipeline singleton in memory.

## Client Behavior

Root `test.py` is the manual inference client, not a pytest file. It supports one file or a directory, server or direct mode, and writes final competition JSON lists to the chosen output directory.

## Data and Index Files

Current durable artifacts:

- `data/candidates/icd10_candidates.jsonl`
- `data/candidates/rxnorm_candidates.jsonl`
- `data/candidates/candidate_aliases.jsonl`
- `data/candidates/candidate_manifest.json`

## Output Format

Final `{id}.json` files contain only a JSON list of concept dictionaries. API responses may include `meta`, but `test.py` writes only the list under `concepts`.

## Added or Needed

The upgrade adds best-effort public data download, Vietnamese alias preparation, KB/index builders, data inventory reporting, GPT/OpenAI-compatible API client, hybrid mode metadata, `/predict_batch`, and a manual output validator.

## Must Not Be Broken

- Deterministic baseline mode.
- Exact offset validation.
- Offline final inference fallback.
- Output schema compatibility.
- `submission/output.zip` structure.
- No repository skills or automatic test-running requirements.
