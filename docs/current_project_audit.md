# Current Project Audit

## Structure

The repository already contains a deterministic baseline under `src/medkg`, operational scripts under `scripts`, official inputs under `input`, organizer statements under `problem`, generated outputs under `outputs`, and submission artifacts under `submission`.

## Current Pipeline

The baseline reads UTF-8 clinical text, extracts medical spans with rule-based NER, detects assertions with context triggers, retrieves local ICD-10/RxNorm candidates, validates exact offsets, and writes competition JSON.

Key modules:

- `src/medkg/ner.py`: rule-based mention extraction.
- `src/medkg/context.py`: assertion detection.
- `src/medkg/ontology.py`: seed and local ontology index loading.
- `src/medkg/retrieval.py`: cached candidate lookup.
- `src/medkg/pipeline.py`: end-to-end orchestration.
- `src/medkg/schema.py`: output validation.

## Server Entrypoint

`src/medkg/server.py` exposes FastAPI endpoints. `main.py` and `scripts/run_server.py` both start the same app. The server keeps one pipeline singleton in memory.

## Client Behavior

Root `test.py` is the manual inference client, not a pytest file. It supports one file or a directory, server or direct mode, and writes final competition JSON lists to the chosen output directory.

## Data and Index Files

Current durable artifacts:

- `data/external/vietnamese_clinical_lexicon.csv`
- `data/processed/*` after downloader/KB scripts run
- `data/indexes/ontology_index.json`
- `data/indexes/alias_exact.json`
- `data/indexes/alias_norm.json`
- `data/indexes/kb_manifest.json`

## Output Format

Final `{id}.json` files contain only a JSON list of concept dictionaries. API responses may include `meta`, but `test.py` writes only the list under `concepts`.

## Added or Needed

The upgrade adds best-effort public data download, Vietnamese alias preparation, KB/index builders, data inventory reporting, local OpenAI-compatible LLM client, hybrid mode metadata, `/predict_batch`, and a manual output validator.

## Must Not Be Broken

- Deterministic baseline mode.
- Exact offset validation.
- Offline final inference fallback.
- Output schema compatibility.
- `submission/output.zip` structure.
- No repository skills or automatic test-running requirements.

