# Requirements Digest

This project targets AI Race 2026, “Ontological Reasoning in Medical Knowledge Retrieval”.

## Required Output

For every input `.txt` file, generate one JSON file containing a list of concept dictionaries. Allowed fields are:

- `text`
- `position`
- `type`
- `assertions`
- `candidates`

Allowed types are `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`, `CHẨN_ĐOÁN`, and `THUỐC`.

`assertions` may contain only `isNegated`, `isFamily`, and `isHistorical`, and only for symptom, diagnosis, and drug concepts.

`candidates` is emitted only for `CHẨN_ĐOÁN` and `THUỐC`: ICD-10 for diagnoses and RxNorm/RxCUI for drugs.

Offsets are zero-based Python string character offsets over the original input text. `input_text[start:end]` must equal `text`.

## Required Flow

```text
download data -> build KB/index -> start FastAPI -> run test.py -> generate JSON -> package output.zip
```

Primary commands:

```bash
pip install -e .
python scripts/download_all_data.py
python scripts/build_knowledge_base.py
python scripts/build_indexes.py
python scripts/inspect_data_status.py
python scripts/run_server.py
python test.py --file problem/sample_input_5.txt --out output
python test.py --input-dir input --out output
python scripts/package_submission.py --output-dir output --submission-dir submission
```

## LLM Rules

The local LLM is optional and fail-open. It must be OpenAI-compatible and self-hosted. It may rerank candidates and disambiguate type/assertion decisions, but it must not invent ICD-10/RxNorm codes or emit final competition JSON without deterministic validation.

