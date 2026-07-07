"""Output schema objects and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medkg.config import (
    ALLOWED_ASSERTIONS,
    ALLOWED_TYPES,
    ASSERTION_TYPES,
    CODED_TYPES,
)


@dataclass(frozen=True)
class Concept:
    text: str
    type: str
    position: tuple[int, int]
    assertions: tuple[str, ...] = field(default_factory=tuple)
    candidates: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "text": self.text,
            "type": self.type,
            "assertions": list(self.assertions),
            "position": [self.position[0], self.position[1]],
        }
        if self.type in CODED_TYPES:
            item["candidates"] = list(self.candidates)
        return item


def validate_concept(item: Any, source_text: str | None = None, index: int | None = None) -> list[str]:
    prefix = f"item[{index}]" if index is not None else "item"
    errors: list[str] = []
    if not isinstance(item, dict):
        return [f"{prefix}: expected object"]

    for key in ("text", "type", "assertions", "position"):
        if key not in item:
            errors.append(f"{prefix}: missing required field {key!r}")

    text = item.get("text")
    if not isinstance(text, str) or not text:
        errors.append(f"{prefix}.text: expected non-empty string")

    concept_type = item.get("type")
    if concept_type not in ALLOWED_TYPES:
        errors.append(f"{prefix}.type: invalid type {concept_type!r}")

    position = item.get("position")
    if (
        not isinstance(position, list)
        or len(position) != 2
        or not all(isinstance(x, int) for x in position)
    ):
        errors.append(f"{prefix}.position: expected [start, end] integers")
    else:
        start, end = position
        if start < 0 or end <= start:
            errors.append(f"{prefix}.position: invalid span [{start}, {end}]")
        if source_text is not None:
            if end > len(source_text):
                errors.append(f"{prefix}.position: end {end} exceeds input length {len(source_text)}")
            elif isinstance(text, str) and source_text[start:end] != text:
                errors.append(
                    f"{prefix}.position: span text {source_text[start:end]!r} does not match {text!r}"
                )

    assertions = item.get("assertions")
    if not isinstance(assertions, list) or not all(isinstance(x, str) for x in assertions):
        errors.append(f"{prefix}.assertions: expected list of strings")
    else:
        if len(assertions) > 3:
            errors.append(f"{prefix}.assertions: expected at most 3 assertions")
        invalid = sorted(set(assertions) - set(ALLOWED_ASSERTIONS))
        if invalid:
            errors.append(f"{prefix}.assertions: invalid assertions {invalid!r}")
        if concept_type not in ASSERTION_TYPES and assertions:
            errors.append(f"{prefix}.assertions: assertions are only valid for symptom, diagnosis, or drug")

    candidates = item.get("candidates")
    if concept_type in CODED_TYPES:
        if candidates is None:
            errors.append(f"{prefix}.candidates: required for diagnosis and drug concepts")
        elif not isinstance(candidates, list) or not all(isinstance(x, str) for x in candidates):
            errors.append(f"{prefix}.candidates: expected list of strings")
    elif candidates is not None:
        errors.append(f"{prefix}.candidates: only diagnosis and drug concepts may include candidates")

    return errors


def validate_output(payload: Any, source_text: str | None = None) -> list[str]:
    if not isinstance(payload, list):
        return ["output: expected a JSON list"]

    errors: list[str] = []
    previous_start = -1
    for i, item in enumerate(payload):
        errors.extend(validate_concept(item, source_text=source_text, index=i))
        if isinstance(item, dict):
            pos = item.get("position")
            if isinstance(pos, list) and len(pos) == 2 and all(isinstance(x, int) for x in pos):
                if pos[0] < previous_start:
                    errors.append(f"item[{i}].position: output is not sorted by start offset")
                previous_start = pos[0]
    return errors

