# Architecture

## Source Modules

`src/` contains separate installable packages by responsibility:

- `api`: FastAPI endpoints and process-level pipeline singleton.
- `core`: configuration, I/O, text normalization, and output schema.
- `extraction`: NER, context assertions, and clinical section detection.
- `knowledge`: ontology lookup, candidate retrieval, and relation inference.
- `integrations`: OpenAI-compatible API client and constrained prompts.
- `services`: end-to-end pipeline orchestration.
- `submission`: archive packaging.

The former all-in-one package and its compatibility wrappers were removed. The
distribution name is now `ai-race-medical-kg` in `pyproject.toml`, and runtime
imports follow package responsibility, for example
`from services.pipeline import MedicalKGPipeline`.

## Runtime Pipeline

1. Discover `.txt` files under the input directory.
2. Load an ontology index if a verified one exists; otherwise use the small built-in seed dictionary.
3. Extract concept spans with rule-based NER:
   - curated Vietnamese and English phrase lexicons for diagnoses, symptoms, tests, and common drugs;
   - drug span extension for dose, route, and frequency tokens;
   - lab-pair extraction for `TEST:VALUE` patterns.
4. Detect assertions with local context windows for negation, family experiencer, and historical mentions.
5. Retrieve candidates from the local ontology index for diagnoses and drugs.
6. Validate each output against the strict JSON schema.
7. Write one output JSON file per input and report timing.

## Performance Choices

- Ontology indexes are loaded once per process.
- Candidate lookup is cached with an LRU cache.
- FastAPI keeps one pipeline instance alive for all requests, so regexes and indexes are not reloaded per request.
- Hybrid mode can call a GPT/OpenAI-compatible API for reranking/disambiguation. If the API is unavailable, the pipeline currently falls back to baseline.

## Assumptions

- The public phase-1 format only accepts concept dictionaries; relation extraction remains internal because no relation output field is shown in the submission examples.
- Character offsets are Python string offsets over decoded UTF-8 text and are end-exclusive.
- `candidates` is required for `CHẨN_ĐOÁN` and `THUỐC`, and omitted for other types.
- If local ICD-10/RxNorm files are unavailable, the built-in seed index is used.

## Extension Points

- Define licensed, versioned ICD-10/RxNorm/UMLS/HPO acquisition before recreating `data/`.
- Expand curated phrase lists in `src/extraction/ner.py` for higher recall.
- Add a deterministic reranker only after preserving strict JSON validation and end-to-end schema checks.

## Build Flow

```bash
python main.py
python tests/test_end_to_end.py input/1.txt
python tests/test_end_to_end.py input
python test.py --input-dir input --out output
ai-race-submit --output-dir output --submission-dir submission
```
