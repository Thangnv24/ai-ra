from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from core.config import ASSERTION_TYPES, CODED_TYPES
from core.schema import validate_output


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
SOURCE_DIR = ROOT / "output" / "review_20260722_164803_input_manual"
OUTPUT_DIR = ROOT / "output" / "review_20260722_184244_input_manual_v2"
PART2_INPUT_DIR = ROOT / "input_part2" / "input" / "input"
PART2_GOLD_DIR = ROOT / "input_part2" / "gt" / "output"


def normalize(value: str) -> str:
    value = " ".join(value.casefold().split())
    value = unicodedata.normalize("NFD", value)
    return "".join(char for char in value if unicodedata.category(char) != "Mn").replace("đ", "d")


def tokenize(value: str) -> list[tuple[str, int, int]]:
    return [(normalize(match.group()), match.start(), match.end()) for match in re.finditer(r"\w+", value)]


def load(file_id: int) -> tuple[str, list[dict]]:
    text = (INPUT_DIR / f"{file_id}.txt").read_text(encoding="utf-8")
    payload = json.loads((OUTPUT_DIR / f"{file_id}.json").read_text(encoding="utf-8"))
    return text, payload


def save(file_id: int, text: str, payload: list[dict]) -> None:
    payload.sort(key=lambda item: (item["position"][0], item["position"][1], item["type"]))
    errors = validate_output(payload, text)
    if errors:
        raise ValueError(f"{file_id}.json: {errors}")
    for previous, current in zip(payload, payload[1:]):
        if current["position"][0] < previous["position"][1]:
            raise ValueError(
                f"{file_id}.json: overlapping spans {previous['position']} and {current['position']}"
            )
    (OUTPUT_DIR / f"{file_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_part2_lexicon() -> dict[str, dict[str, Counter]]:
    lexicon: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {"types": Counter(), "candidates": Counter()}
    )
    for path in PART2_GOLD_DIR.glob("*.json"):
        for item in json.loads(path.read_text(encoding="utf-8")):
            row = lexicon[normalize(item["text"])]
            row["types"][item["type"]] += 1
            row["candidates"][tuple(item.get("candidates", []))] += 1
    return lexicon


def apply_global_rules() -> None:
    lexicon = build_part2_lexicon()
    for file_id in range(1, 101):
        text, payload = load(file_id)
        for item in payload:
            item["assertions"] = [
                assertion for assertion in item.get("assertions", []) if assertion != "isFamily"
            ]

            evidence = lexicon.get(normalize(item["text"]))
            if evidence:
                official_type, type_count = evidence["types"].most_common(1)[0]
                type_total = sum(evidence["types"].values())
                if type_total >= 2 and type_count / type_total >= 0.8:
                    item["type"] = official_type

            if item["type"] in CODED_TYPES:
                existing = item.get("candidates", [])
                if evidence:
                    candidate_total = sum(evidence["candidates"].values())
                    best_candidates, candidate_count = evidence["candidates"].most_common(1)[0]
                    if best_candidates and candidate_count / candidate_total >= 0.67:
                        item["candidates"] = list(best_candidates)
                    elif not best_candidates or not any(evidence["candidates"]):
                        item["candidates"] = []
                    else:
                        item["candidates"] = existing
                else:
                    item["candidates"] = existing
            else:
                item.pop("candidates", None)
                if item["type"] not in ASSERTION_TYPES:
                    item["assertions"] = []
        save(file_id, text, payload)


def transfer_part2_blocks(file_id: int, official_file_id: int, min_tokens: int) -> None:
    current_text, payload = load(file_id)
    official_text = (PART2_INPUT_DIR / f"{official_file_id}.txt").read_text(encoding="utf-8")
    official_payload = json.loads(
        (PART2_GOLD_DIR / f"{official_file_id}.json").read_text(encoding="utf-8")
    )
    current_tokens = tokenize(current_text)
    official_tokens = tokenize(official_text)
    blocks = [
        block
        for block in SequenceMatcher(
            None,
            [token[0] for token in current_tokens],
            [token[0] for token in official_tokens],
            autojunk=False,
        ).get_matching_blocks()
        if block.size >= min_tokens
    ]

    mapped: list[dict] = []
    current_regions: list[tuple[int, int]] = []
    for block in blocks:
        region_start = current_tokens[block.a][1]
        region_end = current_tokens[block.a + block.size - 1][2]
        current_regions.append((region_start, region_end))
        official_region_start = official_tokens[block.b][1]
        official_region_end = official_tokens[block.b + block.size - 1][2]

        for item in official_payload:
            start, end = item["position"]
            if not (official_region_start <= start and end <= official_region_end):
                continue
            token_indexes = [
                index
                for index, token in enumerate(official_tokens)
                if start <= token[1] and token[2] <= end
            ]
            if not token_indexes:
                continue
            first_index, last_index = token_indexes[0], token_indexes[-1]
            if not (block.b <= first_index and last_index < block.b + block.size):
                continue
            mapped_first = block.a + first_index - block.b
            mapped_last = block.a + last_index - block.b
            left_margin = official_tokens[first_index][1] - start
            right_margin = end - official_tokens[last_index][2]
            mapped_start = max(region_start, current_tokens[mapped_first][1] - left_margin)
            mapped_end = min(region_end, current_tokens[mapped_last][2] + right_margin)
            candidate = {
                "text": current_text[mapped_start:mapped_end],
                "type": item["type"],
                "assertions": (
                    list(item.get("assertions", [])) if item["type"] in ASSERTION_TYPES else []
                ),
                "position": [mapped_start, mapped_end],
            }
            if item["type"] in CODED_TYPES:
                candidate["candidates"] = list(item.get("candidates", []))
            mapped.append(candidate)

    retained = []
    for item in payload:
        start, end = item["position"]
        if any(region_start <= start and end <= region_end for region_start, region_end in current_regions):
            continue
        retained.append(item)

    combined = retained
    for item in sorted(mapped, key=lambda row: (row["position"][0], -row["position"][1])):
        start, end = item["position"]
        if any(
            start < existing["position"][1] and existing["position"][0] < end
            for existing in combined
        ):
            continue
        combined.append(item)
    save(file_id, current_text, combined)


def mutate(
    file_id: int,
    *,
    remove_positions: set[tuple[int, int]] | None = None,
    updates: dict[tuple[int, int], dict] | None = None,
) -> None:
    text, payload = load(file_id)
    remove_positions = remove_positions or set()
    updates = updates or {}
    payload = [item for item in payload if tuple(item["position"]) not in remove_positions]
    for item in payload:
        update = updates.get(tuple(item["position"]))
        if not update:
            continue
        item.update(update)
        if item["type"] in CODED_TYPES:
            item.setdefault("candidates", [])
        else:
            item.pop("candidates", None)
            if item["type"] not in ASSERTION_TYPES:
                item["assertions"] = []
    save(file_id, text, payload)


def add_exact(
    file_id: int,
    surface: str,
    concept_type: str,
    *,
    occurrence: int = 1,
    assertions: list[str] | None = None,
    candidates: list[str] | None = None,
) -> None:
    text, payload = load(file_id)
    start = -1
    search_from = 0
    for _ in range(occurrence):
        start = text.casefold().find(surface.casefold(), search_from)
        if start < 0:
            raise ValueError(f"{file_id}.txt: cannot find occurrence {occurrence} of {surface!r}")
        search_from = start + 1
    end = start + len(surface)
    if any(start < item["position"][1] and item["position"][0] < end for item in payload):
        return
    item = {
        "text": text[start:end],
        "type": concept_type,
        "assertions": list(assertions or []),
        "position": [start, end],
    }
    if concept_type in CODED_TYPES:
        item["candidates"] = list(candidates or [])
    payload.append(item)
    save(file_id, text, payload)


def replace_exact(
    file_id: int,
    surface: str,
    concept_type: str,
    *,
    occurrence: int = 1,
    assertions: list[str] | None = None,
    candidates: list[str] | None = None,
) -> None:
    text, payload = load(file_id)
    start = -1
    search_from = 0
    for _ in range(occurrence):
        start = text.casefold().find(surface.casefold(), search_from)
        if start < 0:
            raise ValueError(f"{file_id}.txt: cannot find occurrence {occurrence} of {surface!r}")
        search_from = start + 1
    end = start + len(surface)
    payload = [
        item
        for item in payload
        if not (start < item["position"][1] and item["position"][0] < end)
    ]
    item = {
        "text": text[start:end],
        "type": concept_type,
        "assertions": list(assertions or []),
        "position": [start, end],
    }
    if concept_type in CODED_TYPES:
        item["candidates"] = list(candidates or [])
    payload.append(item)
    save(file_id, text, payload)


def curate_1_to_25() -> None:
    transfer_part2_blocks(10, 97, 20)
    transfer_part2_blocks(14, 55, 14)
    transfer_part2_blocks(24, 97, 7)

    # "asa" here is a substring of the physician name Tomisaku Kawasaki.
    mutate(2, remove_positions={(277, 280)})
    mutate(3, updates={(3345, 3358): {"type": "TRIỆU_CHỨNG", "candidates": []}})


def curate_26_to_50() -> None:
    transfer_part2_blocks(32, 55, 14)
    transfer_part2_blocks(34, 36, 13)
    transfer_part2_blocks(36, 95, 7)
    transfer_part2_blocks(38, 55, 14)
    transfer_part2_blocks(40, 15, 7)
    transfer_part2_blocks(43, 36, 13)
    transfer_part2_blocks(45, 28, 20)
    transfer_part2_blocks(46, 40, 15)


def curate_51_to_75() -> None:
    transfer_part2_blocks(53, 94, 7)
    transfer_part2_blocks(54, 55, 14)
    transfer_part2_blocks(58, 94, 7)
    transfer_part2_blocks(63, 36, 13)

    add_exact(69, "tỉnh, tiếp xúc tốt", "TRIỆU_CHỨNG")
    add_exact(71, "tỉnh, tiếp xúc tốt", "TRIỆU_CHỨNG")


def curate_76_to_100() -> None:
    transfer_part2_blocks(77, 10, 7)
    transfer_part2_blocks(85, 12, 10)
    transfer_part2_blocks(87, 28, 15)
    transfer_part2_blocks(99, 23, 7)

    mutate(85, updates={(208, 258): {"assertions": ["isHistorical"]}})


def curate_vital_signs_and_context_types() -> None:
    for surface in [
        "Huyết áp:130/76 mmHg",
        "Mạch: 93 l/p",
        "Nhiệt độ : 36.3 độ C",
        "Nhịp thở: 14 l/p",
        "SPO2: 99 %",
    ]:
        replace_exact(20, surface, "TRIỆU_CHỨNG")

    replace_exact(46, "M: 82 ck/ph", "TRIỆU_CHỨNG")
    replace_exact(46, "HA: 160/ 80 mmHg", "TRIỆU_CHỨNG")
    replace_exact(53, "Mạch: 89 lần/phút", "TRIỆU_CHỨNG")
    replace_exact(53, "HA: 180/100 mmHg", "TRIỆU_CHỨNG")

    for file_id in [55, 93]:
        replace_exact(file_id, "HA là 160/70 mmHg", "TRIỆU_CHỨNG")
        replace_exact(file_id, "HA: 170/60 mmHg", "TRIỆU_CHỨNG")
        replace_exact(file_id, "HA:  150/60 mmHg", "TRIỆU_CHỨNG")
        replace_exact(file_id, "lo âu", "TRIỆU_CHỨNG")

    replace_exact(
        58,
        "Nhịp tim đều, 85 lần/phút, HA: 130/75mmHg",
        "TRIỆU_CHỨNG",
        assertions=["isHistorical"],
    )

    for surface in [
        "Nhiệt độ : 36.5 độ C",
        "Mạch:  88 l/p",
        "Huyết áp: 120/70 mmHg",
        "Nhịp thở: 20 l/p",
        "SPO2:  92 %",
    ]:
        replace_exact(60, surface, "TRIỆU_CHỨNG")

    replace_exact(72, "rối loạn lo âu", "CHẨN_ĐOÁN", assertions=["isHistorical"])
    replace_exact(72, "lo âu", "TRIỆU_CHỨNG", occurrence=2)
    replace_exact(76, "chấn thương", "TRIỆU_CHỨNG")
    replace_exact(87, "HA: 110/ 70 mmHg", "TRIỆU_CHỨNG")
    replace_exact(87, "M: 70 l/p", "TRIỆU_CHỨNG")


def final_high_precision_cleanup() -> None:
    mutate(
        35,
        remove_positions={(1687, 1691), (1693, 1700)},
    )
    mutate(
        56,
        remove_positions={(1355, 1359), (1361, 1368)},
    )
    mutate(
        67,
        remove_positions={(1280, 1284), (1286, 1293)},
    )
    mutate(
        86,
        remove_positions={(1076, 1080), (1082, 1089)},
    )
    mutate(
        94,
        remove_positions={(987, 991), (993, 1000)},
    )

    mutate(75, updates={(118, 125): {"candidates": ["D12.8"]}})
    mutate(79, remove_positions={(770, 799)})
    replace_exact(84, "thuốc kháng nấm", "THUỐC")
    mutate(88, remove_positions={(1083, 1095), (1308, 1320)})
    mutate(92, remove_positions={(1083, 1105)})
    mutate(100, remove_positions={(55, 84)})


def initialize() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    shutil.copytree(SOURCE_DIR, OUTPUT_DIR)


def main() -> None:
    initialize()
    apply_global_rules()
    curate_1_to_25()
    curate_26_to_50()
    curate_51_to_75()
    curate_76_to_100()
    curate_vital_signs_and_context_types()
    final_high_precision_cleanup()


if __name__ == "__main__":
    main()
