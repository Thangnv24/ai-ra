# Initial Build Report

## Commands Run

```bash
python scripts/download_all_data.py
python scripts/build_knowledge_base.py
python scripts/build_indexes.py
python scripts/inspect_data_status.py
python test.py --direct --file problem/sample_input_5.txt --out output --mode hybrid
python test.py --direct --file problem/sample_input_8.txt --out output --mode hybrid
python test.py --file problem/sample_input_5.txt --out output --mode hybrid
python test.py --direct --input-dir input --out output --mode hybrid
python scripts/validate_outputs.py --output-dir output --input-dir input
python scripts/package_submission.py --output-dir output --submission-dir submission
```

No pytest and no benchmark were run.

## Data Results

- ICD-10: public CDC ZIP download timed out; fallback seed concepts were written to `data/processed/icd10/icd10_concepts.jsonl`.
- RxNorm: RxNav API partially succeeded and wrote 1020 processed rows to `data/processed/rxnorm/rxnorm_concepts.jsonl`; a few terms timed out and were covered by fallback seeds.
- Public corpora: NCBI Disease Corpus, MedMentions, and BC5CDR are recorded as optional development resources and were not downloaded by default.
- Vietnamese aliases: 124 local aliases were written under `data/processed/vi_aliases/`.

## Built Artifacts

- KB concepts: 1039 rows.
- KB aliases: 4275 rows.
- Runtime concept index: `data/indexes/ontology_index.json`.
- Exact alias index: `data/indexes/alias_exact.json`.
- Normalized alias index: `data/indexes/alias_norm.json`.
- Manifest: `data/indexes/kb_manifest.json`.

## Inference Results

Direct hybrid mode over `input/` generated 100 JSON files in `output/`. Because LLM usage was not enabled for that run, all files used deterministic fallback mode.

`python scripts/validate_outputs.py --output-dir output --input-dir input` reported 100 files and 0 errors.

`submission/output.zip` was rebuilt from `output/` with 100 JSON files.

## Remaining Limitations

- ICD-10 public download should be retried on a better network or with a locally downloaded CDC ZIP.
- UMLS and restricted clinical resources require manual credentials/licenses.
- Hybrid LLM mode is wired for a GPT/OpenAI-compatible API and needs `.env` configuration.
- Highest-impact scoring work remains expanding Vietnamese clinical aliases and improving high-recall span proposal.
