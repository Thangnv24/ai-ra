# Data provenance

Runtime inference uses compact candidate artifacts in `data/candidates/`.

Do not commit full raw RxNorm RRF dumps to git. The local full RxNorm source used to rebuild the current slim artifacts is:

```text
D:\downloads\data_ai\RxNorm_full_07062026\rrf
```

Current restored RxNorm runtime artifacts:

- `data/candidates/rxnorm_candidates.jsonl`
- `data/candidates/candidate_aliases.jsonl`
- `data/candidates/candidate_manifest.json`

Rebuild command for the full slim KB, when local raw sources are available:

```bash
python scripts/build_slim_candidate_kb.py \
  --icd10 data/icd10/icd10.jsonl \
  --existing-icd-candidates data/candidates/icd10_candidates.jsonl \
  --rxnconso D:\downloads\data_ai\RxNorm_full_07062026\rrf\RXNCONSO.RRF \
  --rxnarchive D:\downloads\data_ai\RxNorm_full_07062026\rrf\RXNATOMARCHIVE.RRF \
  --out-dir data/candidates
```

The current GitHub-friendly restored artifact is archive-inclusive. It uses both `RXNCONSO.RRF` and
`RXNATOMARCHIVE.RRF`, then stores compact records in `rxnorm_candidates.jsonl` and lookup aliases in
`candidate_aliases.jsonl`. This keeps each file below GitHub's 100 MB single-file limit while retaining
historical RxNorm candidates needed by the organizer examples, such as `360047`.
