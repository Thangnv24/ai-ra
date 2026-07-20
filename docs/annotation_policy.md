# Annotation Policy

This document turns the competition schema and recurring `input_part2` gold
patterns into an annotation contract. It describes proposal and boundary
policy, not clinical truth. When the official examples and a document template
use different boundary conventions, inference must generate both plausible
variants and let the boundary verifier rank them.

## Global contract

- Emit only a verbatim, explicitly stated medical mention.
- Do not infer a diagnosis from symptoms, tests, drugs, or clinical knowledge.
- Keep zero-based, end-exclusive offsets and require `source[start:end] == text`.
- Emit each occurrence separately and remove exact duplicates before decoding.
- Treat section headings, demographics, administrative metadata, and dates as
  context rather than concepts.
- Use context to classify a mention, but never copy text from neighboring
  context into the target span.
- Prefer one complete annotation unit over nested aliases or arbitrary generic
  fragments. Generate alternatives when the observed template is ambiguous.
- Assertions are assigned only after final span selection. Candidates are
  assigned only after final diagnosis/drug boundaries are selected.

## Type policy

### `TRIỆU_CHỨNG`

- Include an explicitly described complaint, sign, examination finding, vital
  sign, mental status, or functional status.
- Retain clinically attached location, laterality, severity, temporal, and
  exertional modifiers when they form one finding.
- Do not shorten a specific phrase such as `đau ngực trái` to a generic head.
- A negation cue may be inside or outside the gold boundary. Generate both
  variants when the template does not settle the convention; assertion scope
  must still see the cue.
- Reject generic observation words such as `bất thường`, `tăng`, or `giảm`
  unless their local construction is annotated as a complete finding.

### `CHẨN_ĐOÁN`

- Include only an explicitly stated clinician diagnosis or problem-list item.
- Prefer the named disease core. Do not infer a disease from evidence.
- Generate a core and an extended variant when stage, severity, cause, or
  anatomy is attached. Rank them using the matching document template.
- A finding becomes a diagnosis only in explicit diagnosis/problem-list
  context; the same surface may be a symptom elsewhere.

### `TÊN_XÉT_NGHIỆM`

- Include the complete visible procedure, assay, imaging, pathology, or
  measurement label.
- Keep aliases and method text when they are part of one structured test label.
- Do not emit nested abbreviations such as `PT` or `TQ` when the complete label
  is the organizer annotation unit.
- A generic word such as `xét nghiệm` is not sufficient without a concrete
  procedure role or a template that explicitly annotates it.

### `KẾT_QUẢ_XÉT_NGHIỆM`

- Include a numeric value with its visible unit when the unit belongs to it.
- Include a complete qualitative result or report finding, not an arbitrary
  adjective detached from its test.
- Generate both value-only and label-plus-value vital-sign variants when the
  template convention is uncertain.
- Do not combine independent test names and results into one entity unless the
  template consistently treats the whole report block as one result.

### `THUỐC`

- Include a medicine used, prescribed, administered, or recorded in medication
  history. A chemical in poisoning or exposure context is not automatically a
  medication.
- Keep brand/ingredient and attached strength or dose form when visible.
- Drug boundary policy is template-dependent. Official examples may include
  route/frequency/PRN, while part2 prescription templates often stop before
  count, schedule, or administration instructions.
- Generate core, core-plus-strength/form, and full-SIG variants when supported;
  never apply one unconditional SIG rule to every template.

## Assertion policy

- `isNegated`: a valid negation cue scopes over the final symptom, diagnosis,
  or drug mention without being cancelled by a contrast boundary.
- `isHistorical`: the mention belongs to prior disease, prior event, or prior
  medication context, not merely because it occurs before the current plan.
- `isFamily`: the grammatical subject is a family member rather than the
  patient.
- Multiple assertions may coexist. Do not emit assertions for tests/results.

## Candidate policy

- Emit ICD-10 candidates only for diagnoses and RxNorm candidates only for
  drugs.
- Normalize the selected entity core for retrieval, but preserve the original
  span in output.
- Treat an empty candidate set as a first-class decision.
- Never invent a code outside the validated local candidate index.

## Error taxonomy

- `hallucinated_or_inferred`: clinically plausible but not explicitly stated.
- `generic_fragment`: medically flavored token without a complete annotation
  unit.
- `boundary_too_short`: missing a required modifier, unit, alias, or method.
- `boundary_too_long`: includes explanation, cause, count, schedule, or adjacent
  finding not belonging to the gold unit.
- `nested_alias`: abbreviation emitted inside a full test label.
- `wrong_type`: exact boundary with an incorrect role-based type.
- `exposure_as_drug`: poison, substance use, or exposure chemical emitted as a
  medication.
- `assertion_scope`: correct span but wrong negation/history/family scope.
- `candidate_null`: a code emitted when gold expects empty, or vice versa.
- `candidate_identity`: nonempty candidate set contains the wrong code.

## Precedence

1. Exact source text and schema validity.
2. Organizer annotation patterns for the matching template.
3. Explicit local role and section context.
4. Cross-template statistical evidence.
5. General clinical plausibility, used only as weak proposal evidence.
