# AGENTS.md

## Project goal

Build a fast, deterministic Python system for the competition "Ontological Reasoning in Medical Knowledge Retrieval".

The system processes free-text medical inputs such as doctor notes, discharge summaries, lab results, and EHR snippets. It must detect and normalize medical concepts, classify concept types, map diseases to ICD-10 and drugs to RxNorm, infer context such as negation/family/history, infer relations between concepts, and produce the exact submission format described in `problem/`.

## Important files

- `docs/research_notes.md`: prior research notes and recommended datasets/ontologies.
- `problem/statement1.md`: phase-1 submission/scoring statement from the organizer.
- `problem/statement2.md`: full problem overview, input/output definition, data description, and phase schedule.
- `problem/input_output_spec.md`: input/output schema.
- `problem/scoring.md`: benchmark and scoring rules.
- `problem/submission_format.md`: final submission requirements.
- `input/`: official test input files when unpacked from the organizer ZIP.
- `outputs/`: generated output files.
- `submission/`: final packaged submission.

## Engineering rules

- Prioritize correctness and speed.
- Do not use web/API calls during final inference unless the problem statement explicitly allows it.
- Build all ontology indexes offline before running test inference.
- Keep runtime deterministic and reproducible.
- Validate every output against the required schema.
- Avoid free-form LLM output in final files; always parse/validate into structured JSON.
- Cache expensive computations.
- Add benchmark timing for end-to-end runtime and per-file latency.

## Preferred architecture

Use a hybrid pipeline:

1. Rule/model-based entity extraction.
2. Context/assertion detection for negation, family member, historical mention, hypothetical mention.
3. Candidate retrieval from local ICD-10/RxNorm/UMLS/HPO indexes.
4. Optional 9B model reranking/classification with strict JSON schema.
5. Ontology reasoning and rule-based correction.
6. Submission validation.

Do not build a slow open-ended ReAct agent for final inference.

## Manual commands

- Install: `pip install -e .`
- Download public/fallback resources: `python scripts/download_all_data.py`
- Build knowledge base: `python scripts/build_knowledge_base.py`
- Build indexes: `python scripts/build_indexes.py`
- Inspect data status: `python scripts/inspect_data_status.py`
- Run API server: `python scripts/run_server.py`
- Run one file via server: `python test.py --file problem/sample_input_5.txt --out output`
- Run folder via server: `python test.py --input-dir input --out output`
- Run direct fallback mode: `python test.py --direct --input-dir input --out output`
- Validate generated output manually: `python scripts/validate_outputs.py --output-dir output --input-dir input`
- Package: `python scripts/package_submission.py --output-dir output --submission-dir submission`

## Data-source rules

- Treat `problem/statement1.md` and `problem/statement2.md` as the source of truth for labels, JSON fields, scoring, and allowed external model/API behavior.
- Official inputs are Vietnamese free-form clinical snippets with English drug names, mixed Vietnamese lab names, abbreviations, numeric lab results, and RxNorm/ICD-10 candidate requirements.
- Keep downloader/build logic in `scripts/`; keep `data/` for downloaded raw files, curated external dictionaries, and generated indexes only.
- Final inference must use local files only. Network access is allowed only in explicit preparation scripts such as ICD-10/RxNorm downloaders.

## Change guidance

- Keep changes deterministic and compatible with the schema in `problem/`.
- Do not add repository skills or automatic test generation/running requirements to this file.
- Document assumptions in `docs/architecture.md` when they affect inference behavior.

## README writing guidance

When asked to create or update `README.md`, write it as a complete project guide. The README should include:

1. Project overview
   - Explain the competition task: medical concept extraction, normalization, context/assertion detection, ICD-10/RxNorm candidate mapping, and valid JSON submission generation.
   - State that final inference is deterministic and offline.

2. Project structure
   - Describe the purpose of major directories and entry points:
     - `input/`: official `.txt` files.
     - `outputs/`: generated `.json` predictions.
     - `submission/`: final `output.zip`.
     - `problem/`: official problem statements and schema notes.
     - `data/raw`, `data/external`, `data/indexes`: downloaded resources, curated dictionaries, generated ontology indexes.
     - `src/medkg/`: core pipeline package.
     - `scripts/`: operational scripts.
     - `main.py`: FastAPI server entry point.
     - `test.py`: manual file/folder inference client.
     - `requirements.txt`: runtime dependencies.

3. Techniques used
   - Rule-based medical NER using Vietnamese/English lexicons and regex patterns.
   - Vietnamese text normalization: lowercasing, accent stripping, whitespace cleanup, fuzzy matching.
   - Context/assertion detection for `isNegated`, `isFamily`, and `isHistorical`.
   - Local ontology retrieval from ICD-10/RxNorm seed dictionaries and optional downloaded indexes.
   - LRU caching for repeated normalization/candidate lookup.
   - Strict output schema validation before writing JSON.
   - FastAPI singleton pipeline loading index/regex once per process.

4. Runtime flow
   - Startup: load config, ontology index, external Vietnamese lexicon, regex patterns.
   - Input: read one `.txt` file or folder of `.txt` files.
   - Extraction: detect symptoms, diagnoses, drugs, lab names, lab results.
   - Context: infer assertion tags from nearby text windows.
   - Candidate retrieval: map diagnoses/drugs to ICD-10/RxNorm when local evidence exists.
   - Validation: check type labels, offsets, assertion fields, candidate fields.
   - Output: write one `.json` per input and optionally package `submission/output.zip`.

5. Installation
   - Use:
     ```bash
     pip install -r requirements.txt
     ```
   - Mention Python version if known or assumed.

6. Running the project
   - API server:
     ```bash
     python main.py
     ```
   - API smoke call:
     ```bash
     python test.py --path input/1.txt
     ```
   - Direct local inference:
     ```bash
     python test.py --path input/1.txt --direct
     ```
   - Folder inference:
     ```bash
     python test.py --path input --output-dir outputs
     ```
   - Build ontology index:
     ```bash
     python scripts/build_ontology_index.py
     ```
   - Package submission:
     ```bash
     python scripts/package_submission.py --output-dir outputs --submission-dir submission
     ```

7. Data preparation
   - Explain that final inference uses only local files.
   - Downloader scripts are preparation-only:
     ```bash
     python scripts/download_icd10.py
     python scripts/download_rxnorm.py --limit 25
     python scripts/build_ontology_index.py
     ```

8. Output format
   - Explain each JSON item includes:
     - `text`
     - `type`
     - `position`
     - `assertions`
     - `candidates` for `CHẨN_ĐOÁN` and `THUỐC`.
   - State that offsets are zero-based and end-exclusive.

9. Known limitations and next improvements
   - Mention lexicon/rule baseline limitations.
   - Recommend expanding Vietnamese clinical lexicons from official input.
   - Recommend adding larger ICD-10/RxNorm/UMLS/HPO local resources.
   - Recommend improving lab extraction and disease/drug normalization.
