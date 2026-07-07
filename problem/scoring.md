# Scoring

The public statement describes three scoring components:

- Concept text detection uses Word Error Rate over predicted `text`.
- `assertions` use Jaccard similarity against the gold assertion set.
- `candidates` use Jaccard similarity against gold ICD-10 or RxNorm candidate IDs, weighted by candidate-list size.

The final score is:

```text
0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score
```

If the predicted text is correct but the concept `type` is wrong, the concept is treated as a mismatch.

Implications for this baseline:

- Prefer high-precision spans over overly broad spans.
- Keep type labels exact.
- Emit candidates only for diagnoses and drugs.
- Do not invent candidate codes when no local ontology evidence is available.
