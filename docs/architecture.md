# Architecture

## Runtime Pipeline

1. Discover `.txt` files under the input directory.
2. Load the ontology index once from `data/indexes/ontology_index.json` if present, otherwise use deterministic seed dictionaries plus any local CSV/JSON/TXT resources found in `data/raw` and `data/external`.
3. Extract concept spans with rule-based NER:
   - optional Vietnamese clinical lexicon loaded from `data/external/vietnamese_clinical_lexicon.csv`;
   - curated Vietnamese and English phrase lexicons for diagnoses, symptoms, tests, and common drugs;
   - drug span extension for dose, route, and frequency tokens;
   - lab-pair extraction for `TEST:VALUE` patterns.
4. Detect assertions with local context windows for negation, family experiencer, and historical mentions.
5. Retrieve candidates from the local ontology index for diagnoses and drugs.
6. Validate each output against the strict JSON schema.
7. Write one output JSON file per input and report timing.

## Performance Choices

- The final inference path is stdlib-only and offline.
- Ontology indexes are loaded once per process.
- Candidate lookup is cached with an LRU cache.
- No model or agent loop runs during inference.
- FastAPI keeps one pipeline instance alive for all requests, so regexes and indexes are not reloaded per request.
- Hybrid mode can call a local OpenAI-compatible LLM for safe reranking/disambiguation. If the LLM is unavailable, the pipeline falls back to baseline.

## Assumptions

- The public phase-1 format only accepts concept dictionaries; relation extraction remains internal because no relation output field is shown in the submission examples.
- Character offsets are Python string offsets over decoded UTF-8 text and are end-exclusive.
- `candidates` is required for `CHẨN_ĐOÁN` and `THUỐC`, and omitted for other types.
- The repository did not include the official 100-file test set, so `test_inputs/1.txt` is a synthetic smoke input from the public examples in `problem/statement1.md` and `problem/statement2.md`.
- If local ICD-10/RxNorm files are unavailable, the built-in seed index is used.

## Extension Points

- Add large ICD-10/RxNorm/UMLS/HPO resources to `data/external` and run `python scripts/build_ontology_index.py`.
- Expand curated phrase lists in `src/medkg/ner.py` for higher recall.
- Add a deterministic local reranker only after preserving strict JSON validation and latency benchmarks.

## Build Flow

```bash
python scripts/download_all_data.py
python scripts/build_knowledge_base.py
python scripts/build_indexes.py
python scripts/inspect_data_status.py
python scripts/run_server.py
python test.py --input-dir input --out output
python scripts/package_submission.py --output-dir output --submission-dir submission
```
