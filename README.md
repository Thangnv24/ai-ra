# Ontological Reasoning in Medical Knowledge Retrieval

Runnable project flow for AI Race 2026 medical ontology retrieval. The system reads Vietnamese free-text clinical notes, extracts medical concepts, assigns competition labels, detects context assertions, maps diagnoses to ICD-10 and drugs to RxNorm/RxCUI when local evidence exists, and writes the official JSON submission format.

Final inference is deterministic by default and can optionally use a local self-hosted <=9B OpenAI-compatible LLM for safe reranking/disambiguation. If the local LLM is unavailable, the pipeline falls back to the deterministic baseline.

## Project Structure

- `input/`: official `.txt` input files.
- `output/`: recommended output folder for newly generated competition JSON files.
- `outputs/`: previous/generated baseline output folder kept for compatibility.
- `submission/`: final `output.zip`.
- `problem/`: organizer statements, schema notes, and sample inputs.
- `data/raw/`: downloaded raw public data.
- `data/external/`: curated local resources such as Vietnamese lexicons.
- `data/processed/`: normalized KB tables and alias JSONL files.
- `data/indexes/`: runtime ontology and alias indexes.
- `src/medkg/`: core Python package.
- `scripts/`: download, build, server, validation, and packaging commands.
- `main.py`: root FastAPI entrypoint.
- `test.py`: manual inference client for one file or a folder.
- `.env.example`: local LLM configuration template.

## Install

```bash
pip install -e .
```

## Download Data

```bash
python scripts/download_all_data.py
```

This orchestrates:

```bash
python scripts/download_icd10.py
python scripts/download_rxnorm.py
python scripts/download_public_corpora.py
python scripts/prepare_vi_medical_aliases.py
python scripts/inspect_data_status.py
```

Downloads are best-effort. If ICD-10/RxNorm/public corpora are unavailable, the scripts keep curated fallback data and document status in `docs/data_inventory.md`.

## Build KB And Indexes

```bash
python scripts/build_knowledge_base.py
python scripts/build_indexes.py
python scripts/inspect_data_status.py
```

Generated artifacts:

- `data/processed/concepts.jsonl`
- `data/processed/aliases.jsonl`
- `data/processed/drug_aliases.jsonl`
- `data/processed/disease_aliases.jsonl`
- `data/processed/lab_aliases.jsonl`
- `data/processed/symptom_aliases.jsonl`
- `data/indexes/ontology_index.json`
- `data/indexes/alias_exact.json`
- `data/indexes/alias_norm.json`
- `data/indexes/kb_manifest.json`

## Run FastAPI

```bash
python scripts/run_server.py
```

Equivalent root entrypoint:

```bash
python main.py
```

Server URL: `http://127.0.0.1:8000`

Endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict_batch`
- `POST /predict_file`

## Run Client

One file through the server:

```bash
python test.py --file problem/sample_input_5.txt --out output
python test.py --file problem/sample_input_8.txt --out output
```

Folder through the server:

```bash
python test.py --input-dir input --out output
```

Direct mode without server:

```bash
python test.py --direct --file problem/sample_input_5.txt --out output
python test.py --direct --input-dir input --out output
```

Useful options:

```text
--mode baseline|hybrid|llm_full_doc
--limit N
--pretty
--url http://127.0.0.1:8000
```

## Validate And Package

```bash
python scripts/validate_outputs.py --output-dir output --input-dir input
python scripts/package_submission.py --output-dir output --submission-dir submission
```

The package command creates:

```text
submission/output.zip
```

Zip contents:

```text
output/1.json
output/2.json
...
output/100.json
```

## Output Format

Each output file is a JSON list. Each item may contain only:

- `text`
- `position`
- `type`
- `assertions`
- `candidates`

`position` is `[start, end]`, zero-based and end-exclusive over the original Python string. The validator enforces `input_text[start:end] == text`.

`candidates` is only for:

- `CHẨN_ĐOÁN`: ICD-10 codes
- `THUỐC`: RxNorm/RxCUI codes

## Techniques Used

The baseline uses rule-based medical NER with Vietnamese/English lexicons and regex patterns. It preserves original text for offsets while using normalized lowercase/no-accent forms for matching. Drug spans are expanded to include dose, route, and frequency, then candidate retrieval strips modifiers to match the base drug name.

Context detection uses scoped triggers for `isNegated`, `isFamily`, and `isHistorical`. Candidate retrieval uses local ICD-10/RxNorm aliases, exact normalized lookup, fuzzy matching, and LRU caching.

The KB build flow merges public/fallback ICD-10, public/fallback RxNorm, and Vietnamese clinical aliases into JSONL tables. Runtime indexes are loaded once at startup.

Hybrid mode sends deterministic mention proposals plus retrieved candidates to a local OpenAI-compatible chat-completions server. The LLM is only allowed to rerank candidates, adjust type/assertions, or keep/drop proposals. It cannot invent unsupported codes. Invalid or unavailable LLM responses are ignored safely.

## Local LLM Configuration

```bash
cp .env.example .env
```

Example `.env` values:

```text
MEDKG_MODE=hybrid
MEDKG_LLM_ENABLED=true
MEDKG_LLM_BASE_URL=http://localhost:8000/v1
MEDKG_LLM_MODEL=your-local-9b-model
MEDKG_LLM_API_KEY=dummy
MEDKG_LLM_FAIL_OPEN=true
```

Example local vLLM OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model YOUR_LOCAL_9B_MODEL \
  --host 0.0.0.0 \
  --port 8000
```

## Known Limitations

- Public corpora are optional development resources and are not required for inference.
- UMLS and MIMIC-style resources require manual credentials/licenses and are not downloaded automatically.
- The deterministic extractor is lexicon/rule-heavy, so recall improves most by expanding Vietnamese aliases from official inputs.
- LLM mode is fail-open and optional; final JSON remains schema-validated deterministic output.
