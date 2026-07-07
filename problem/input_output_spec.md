# Input and Output Spec

## Input

Phase 1 inputs are plain UTF-8 `.txt` files. The official archive is expected to contain 100 files named `1.txt` through `100.txt` under an input folder. This repository uses `test_inputs/` as the default input directory and also supports nested `.txt` files.

## Output

For each input file, write one JSON file with the same stem under `outputs/`.

Each output file must contain a JSON list. Each item is a dictionary with:

- `text`: exact concept text found in the input.
- `type`: one of `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`, `CHẨN_ĐOÁN`, `THUỐC`.
- `position`: `[start, end]` character offsets, zero-based, end-exclusive.
- `assertions`: list containing zero or more of `isNegated`, `isFamily`, `isHistorical`. Assertions are only used for diagnoses, drugs, and symptoms.
- `candidates`: list of ICD-10 codes for `CHẨN_ĐOÁN` or RxNorm RxCUIs for `THUỐC`. The field is omitted for other concept types.

No extra output fields are emitted by the baseline because the public scoring spec only shows the fields above.
