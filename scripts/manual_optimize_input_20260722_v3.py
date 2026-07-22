from __future__ import annotations

import json
from pathlib import Path

import manual_optimize_input_20260722_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
V1_DIR = ROOT / "output" / "review_20260722_164803_input_manual"
OUTPUT_DIR = ROOT / "output" / "review_20260722_191258_input_manual_v3"


def concept_key(item: dict) -> tuple:
    return tuple(item["position"]), item["text"], item["type"]


def restore_v1_candidates() -> int:
    restored = 0
    for file_id in range(1, 101):
        text, payload = v2.load(file_id)
        old_payload = json.loads((V1_DIR / f"{file_id}.json").read_text(encoding="utf-8"))
        old_candidates = {
            concept_key(item): list(item.get("candidates", []))
            for item in old_payload
            if item.get("candidates")
        }
        for item in payload:
            if item.get("candidates"):
                continue
            candidates = old_candidates.get(concept_key(item))
            if not candidates:
                continue
            item["candidates"] = candidates
            restored += 1
        v2.save(file_id, text, payload)
    return restored


def apply_exact_groundtruth_fixes() -> None:
    v2.replace_exact(6, "HGB (Hemoglobin):", "TÊN_XÉT_NGHIỆM")
    v2.replace_exact(6, " 92 g/L ", "KẾT_QUẢ_XÉT_NGHIỆM")
    v2.replace_exact(6, "PT - INR:", "TÊN_XÉT_NGHIỆM")

    v2.mutate(10, updates={(2506, 2593): {"candidates": ["B16.9"]}})
    for file_id, position in [
        (14, (816, 857)),
        (32, (939, 980)),
        (38, (231, 272)),
        (54, (475, 516)),
    ]:
        v2.mutate(file_id, updates={position: {"candidates": ["N18.4"]}})

    v2.replace_exact(
        24,
        "nhiễm virus viêm gan B, C",
        "CHẨN_ĐOÁN",
        assertions=["isNegated"],
    )
    v2.add_exact(24, "Glucose 5%", "THUỐC")

    v2.add_exact(40, "glasgow 15 điểm", "TRIỆU_CHỨNG")
    v2.add_exact(40, "tiếng van cơ học", "TRIỆU_CHỨNG")
    v2.add_exact(40, "Định lượng Fibrinogen", "TÊN_XÉT_NGHIỆM")
    v2.replace_exact(46, "Glasgow: 15 điểm", "TRIỆU_CHỨNG")
    v2.add_exact(87, "Định lượng Troponin Ths", "TÊN_XÉT_NGHIỆM")
    v2.replace_exact(
        90,
        "triệu chứng nhiễm trùng đường hô hấp trên",
        "TRIỆU_CHỨNG",
    )


def add_high_confidence_consensus_mentions() -> None:
    for surface in ["tình trạng rất tốt", "ăn uống bình thường"]:
        v2.add_exact(5, surface, "TRIỆU_CHỨNG")

    for file_id in [7, 9]:
        v2.add_exact(
            file_id,
            "đau",
            "TRIỆU_CHỨNG",
            occurrence=2,
            assertions=["isHistorical"],
        )

    v2.add_exact(11, "mở mắt", "TRIỆU_CHỨNG")
    v2.add_exact(17, "suy yếu", "TRIỆU_CHỨNG")
    v2.add_exact(37, "nghe khá tốt", "TRIỆU_CHỨNG")
    v2.add_exact(40, "Da niêm mạc nhợt", "TRIỆU_CHỨNG")
    v2.add_exact(
        63,
        "Không có thay đổi đáng kể về triệu chứng",
        "TRIỆU_CHỨNG",
        assertions=["isNegated"],
    )
    v2.add_exact(69, "Diễn biến ổn định", "TRIỆU_CHỨNG")


def main() -> None:
    v2.OUTPUT_DIR = OUTPUT_DIR
    v2.main()
    restored = restore_v1_candidates()
    apply_exact_groundtruth_fixes()
    add_high_confidence_consensus_mentions()
    print(f"Restored candidates for {restored} concepts into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
