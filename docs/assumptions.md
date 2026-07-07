# Assumptions

- Official test inputs are not present in this checkout. The included `test_inputs/1.txt` is a smoke input, not a real competition test file.
- Final inference must be offline; scripts under `scripts/download_*.py` are preparation utilities only.
- The public schema lists no top-level metadata and no relation field, so output files are JSON lists only.
- Candidate IDs are emitted only when the local ontology index has a matching code.
- Vietnamese labels are written exactly as Unicode strings in JSON output.

