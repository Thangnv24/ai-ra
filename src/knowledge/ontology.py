"""Local ontology index loading and lookup."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from core.config import (
    SYSTEM_ICD10,
    SYSTEM_RXNORM,
    TYPE_DIAGNOSIS,
    TYPE_DRUG,
)
from core.text import normalize_key, similarity


@dataclass(frozen=True)
class OntologyEntry:
    code: str
    name: str
    system: str
    concept_type: str
    aliases: tuple[str, ...] = ()
    priority: int = 100

    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


def seed_entries() -> list[OntologyEntry]:
    return [
        OntologyEntry("K21.0", "Gastro-esophageal reflux disease with esophagitis", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("b\u1ec7nh tr\u00e0o ng\u01b0\u1ee3c d\u1ea1 d\u00e0y - th\u1ef1c qu\u1ea3n", "tr\u00e0o ng\u01b0\u1ee3c d\u1ea1 d\u00e0y th\u1ef1c qu\u1ea3n", "gerd"), 1),
        OntologyEntry("K21.9", "Gastro-esophageal reflux disease without esophagitis", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("b\u1ec7nh tr\u00e0o ng\u01b0\u1ee3c d\u1ea1 d\u00e0y - th\u1ef1c qu\u1ea3n", "tr\u00e0o ng\u01b0\u1ee3c d\u1ea1 d\u00e0y th\u1ef1c qu\u1ea3n", "gerd"), 2),
        OntologyEntry("I10", "Essential hypertension", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("t\u0103ng huy\u1ebft \u00e1p", "cao huy\u1ebft \u00e1p", "hypertension", "htn"), 10),
        OntologyEntry("E11.9", "Type 2 diabetes mellitus without complications", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("\u0111\u00e1i th\u00e1o \u0111\u01b0\u1eddng type 2", "ti\u1ec3u \u0111\u01b0\u1eddng type 2", "type 2 diabetes", "diabetes mellitus"), 10),
        OntologyEntry("J45.909", "Unspecified asthma, uncomplicated", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("hen suy\u1ec5n", "asthma"), 10),
        OntologyEntry("J18.9", "Pneumonia, unspecified organism", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("vi\u00eam ph\u1ed5i", "pneumonia"), 10),
        OntologyEntry("F41.9", "Anxiety disorder, unspecified", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("lo \u00e2u", "anxiety"), 25),
        OntologyEntry("G47.00", "Insomnia, unspecified", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("m\u1ea5t ng\u1ee7", "insomnia"), 25),
        OntologyEntry("I25.10", "Atherosclerotic heart disease of native coronary artery without angina pectoris", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("b\u1ec7nh tim m\u1ea1ch do x\u01a1 v\u1eefa \u0111\u1ed9ng m\u1ea1ch", "x\u01a1 v\u1eefa \u0111\u1ed9ng m\u1ea1ch", "atherosclerotic cardiovascular disease"), 10),
        OntologyEntry("I48.91", "Unspecified atrial fibrillation", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("rung nh\u0129", "atrial fibrillation", "afib"), 10),
        OntologyEntry("I21.9", "Acute myocardial infarction, unspecified", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("nh\u1ed3i m\u00e1u c\u01a1 tim", "myocardial infarction"), 20),
        OntologyEntry("I71.9", "Aortic aneurysm of unspecified site, without rupture", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("ph\u00ecnh \u0111\u1ed9ng m\u1ea1ch ch\u1ee7", "aortic aneurysm"), 20),
        OntologyEntry("K70.30", "Alcoholic cirrhosis of liver without ascites", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("x\u01a1 gan do r\u01b0\u1ee3u", "alcoholic cirrhosis"), 10),
        OntologyEntry("K76.82", "Hepatic encephalopathy", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("h\u1ed9i ch\u1ee9ng n\u00e3o gan", "hepatic encephalopathy"), 10),
        OntologyEntry("K29.70", "Gastritis, unspecified, without bleeding", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("vi\u00eam d\u1ea1 d\u00e0y", "gastritis"), 10),
        OntologyEntry("K80.50", "Calculus of bile duct without cholangitis or cholecystitis without obstruction", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("s\u1ecfi \u0111o\u1ea1n cu\u1ed1i \u1ed1ng m\u1eadt ch\u1ee7", "s\u1ecfi \u1ed1ng d\u1eabn m\u1eadt chung \u0111o\u1ea1n cu\u1ed1i", "common bile duct stone"), 10),
        OntologyEntry("E78.5", "Hyperlipidemia, unspecified", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("t\u0103ng lipid m\u00e1u", "hyperlipidemia"), 10),
        OntologyEntry("I95.9", "Hypotension, unspecified", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("h\u1ea1 huy\u1ebft \u00e1p", "hypotension"), 10),
        OntologyEntry("K83.1", "Obstruction of bile duct", SYSTEM_ICD10, TYPE_DIAGNOSIS, ("t\u1eafc ngh\u1ebdn \u0111\u01b0\u1eddng m\u1eadt", "gi\u00e3n \u0111\u01b0\u1eddng m\u1eadt", "biliary obstruction"), 10),
        OntologyEntry("308135", "amlodipine 10 MG Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("amlodipine 10 mg po daily", "amlodipine 10 mg", "amlodipine"), 1),
        OntologyEntry("243670", "aspirin 81 MG Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("aspirin 81 mg po daily", "aspirin 81 mg", "aspirin"), 1),
        OntologyEntry("866436", "metoprolol succinate 50 MG Extended Release Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("metoprolol succinate xl 50 mg po daily", "metoprolol succinate 50 mg", "metoprolol succinate", "metoprolol"), 1),
        OntologyEntry("392085", "guaifenesin Oral Product", SYSTEM_RXNORM, TYPE_DRUG, ("guaifenesin ml po q6h:prn", "guaifenesin"), 1),
        OntologyEntry("7597", "nystatin", SYSTEM_RXNORM, TYPE_DRUG, ("nystatin oral suspension 5 ml po qid:prn", "nystatin oral suspension", "nystatin"), 1),
        OntologyEntry("313782", "acetaminophen 325 MG Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("acetaminophen 325-650 mg po q6h:prn", "acetaminophen 325 mg", "acetaminophen", "paracetamol"), 1),
        OntologyEntry("904475", "pravastatin 40 MG Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("pravastatin 40 mg po daily", "pravastatin 40 mg", "pravastatin"), 1),
        OntologyEntry("1099279", "docusate sodium 100 MG Oral Capsule", SYSTEM_RXNORM, TYPE_DRUG, ("docusate sodium 100 mg po bid", "docusate sodium 100 mg", "docusate sodium"), 1),
        OntologyEntry("312935", "senna 8.6 MG Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("senna 8.6 mg po bid:prn", "senna 8.6 mg", "senna"), 1),
        OntologyEntry("197527", "clonazepam 0.5 MG Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("clonazepam 0.5 mg po qam:prn", "clonazepam 0.5 mg", "clonazepam"), 1),
        OntologyEntry("197528", "clonazepam 1 MG Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("clonazepam 1.5 mg po qhs", "clonazepam 1.5 mg", "clonazepam"), 2),
        OntologyEntry("360047", "chlorpheniramine 0.4 MG/ML Oral Solution", SYSTEM_RXNORM, TYPE_DRUG, ("chlorpheniramine 0.4 mg/ml", "chlorpheniramine"), 1),
        OntologyEntry("1660761", "capsaicin 0.38 MG/ML Topical Cream", SYSTEM_RXNORM, TYPE_DRUG, ("capsaicin 0.38 mg/ml", "capsaicin"), 1),
        OntologyEntry("860975", "metformin 500 MG Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("metformin 500 mg", "metformin"), 10),
        OntologyEntry("197361", "lisinopril 10 MG Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("lisinopril 10 mg", "lisinopril"), 10),
        OntologyEntry("617314", "atorvastatin 20 MG Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("atorvastatin 20 mg", "atorvastatin"), 10),
        OntologyEntry("197591", "ibuprofen 400 MG Oral Tablet", SYSTEM_RXNORM, TYPE_DRUG, ("ibuprofen 400 mg", "ibuprofen"), 10),
        OntologyEntry("308182", "amoxicillin 500 MG Oral Capsule", SYSTEM_RXNORM, TYPE_DRUG, ("amoxicillin 500 mg", "amoxicillin"), 10),
        OntologyEntry("6918", "metoprolol", SYSTEM_RXNORM, TYPE_DRUG, ("metoprolol 25mg po bid", "metoprolol 25 mg", "metoprolol"), 5),
        OntologyEntry("3640", "doxycycline", SYSTEM_RXNORM, TYPE_DRUG, ("doxycycline",), 5),
        OntologyEntry("1202", "atenolol", SYSTEM_RXNORM, TYPE_DRUG, ("atenolol",), 5),
        OntologyEntry("1191", "aspirin", SYSTEM_RXNORM, TYPE_DRUG, ("aspirin 325mg", "aspirin 325 mg", "aspirin"), 5),
        OntologyEntry("7646", "omeprazole", SYSTEM_RXNORM, TYPE_DRUG, ("omeprazole",), 5),
        OntologyEntry("7052", "propofol", SYSTEM_RXNORM, TYPE_DRUG, ("propofol",), 5),
        OntologyEntry("8163", "phentolamine", SYSTEM_RXNORM, TYPE_DRUG, ("phentolamine",), 5),
        OntologyEntry("7512", "norepinephrine", SYSTEM_RXNORM, TYPE_DRUG, ("levophed", "norepinephrine"), 5),
    ]


class OntologyIndex:
    def __init__(self, entries: Iterable[OntologyEntry]):
        dedup: dict[tuple[str, str], OntologyEntry] = {}
        for entry in entries:
            if not entry.code:
                continue
            dedup[(entry.system, entry.code)] = entry
        self.entries = sorted(dedup.values(), key=lambda e: (e.priority, e.system, e.code))
        self._by_key: dict[tuple[str, str], list[OntologyEntry]] = {}
        self._names: list[tuple[str, OntologyEntry]] = []
        for entry in self.entries:
            for name in entry.all_names():
                key = normalize_key(name)
                if not key:
                    continue
                self._by_key.setdefault((entry.concept_type, key), []).append(entry)
                self._names.append((key, entry))

    def lookup(self, text: str, concept_type: str, limit: int = 5) -> list[OntologyEntry]:
        key = normalize_key(text)
        exact = self._by_key.get((concept_type, key), [])
        if exact:
            return self._sorted_unique(exact)[:limit]

        dose_key = normalize_key(strip_drug_route_frequency(text)) if concept_type == TYPE_DRUG else key
        if dose_key != key:
            exact = self._by_key.get((concept_type, dose_key), [])
            if exact:
                return self._sorted_unique(exact)[:limit]

        stripped = strip_drug_modifiers(text) if concept_type == TYPE_DRUG else text
        stripped_key = normalize_key(stripped)
        if stripped_key != key:
            exact = self._by_key.get((concept_type, stripped_key), [])
            if exact:
                return self._sorted_unique(exact)[:limit]

        candidates: list[tuple[float, OntologyEntry]] = []
        if len(key) >= 5:
            for name_key, entry in self._names:
                if entry.concept_type != concept_type:
                    continue
                score = similarity(key, name_key)
                if score >= 0.88:
                    candidates.append((score, entry))
        candidates.sort(key=lambda row: (-row[0], row[1].priority, row[1].code))
        return self._sorted_unique(entry for _, entry in candidates)[:limit]

    @staticmethod
    def _sorted_unique(entries: Iterable[OntologyEntry]) -> list[OntologyEntry]:
        seen: set[tuple[str, str]] = set()
        out: list[OntologyEntry] = []
        for entry in sorted(entries, key=lambda e: (e.priority, e.system, e.code)):
            key = (entry.system, entry.code)
            if key not in seen:
                out.append(entry)
                seen.add(key)
        return out

    def to_json_data(self) -> list[dict[str, object]]:
        return [
            {
                **asdict(entry),
                "aliases": list(entry.aliases),
            }
            for entry in self.entries
        ]


def strip_drug_modifiers(text: str) -> str:
    tokens = normalize_key(text).split()
    kept: list[str] = []
    stop_tokens = {
        "mg",
        "mcg",
        "g",
        "ml",
        "iu",
        "po",
        "iv",
        "im",
        "sc",
        "bid",
        "tid",
        "qid",
        "qam",
        "qhs",
        "daily",
        "prn",
    }
    for token in tokens:
        if any(ch.isdigit() for ch in token) or token in stop_tokens:
            break
        kept.append(token)
    return " ".join(kept) if kept else text


def strip_drug_route_frequency(text: str) -> str:
    tokens = normalize_key(text).split()
    stop_tokens = {
        "po",
        "iv",
        "im",
        "sc",
        "bid",
        "tid",
        "qid",
        "qam",
        "qhs",
        "daily",
        "prn",
        "x",
    }
    kept: list[str] = []
    for token in tokens:
        if token in stop_tokens:
            break
        kept.append(token)
    return " ".join(kept) if kept else text


def entries_from_json(path: Path) -> list[OntologyEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: list[OntologyEntry] = []
    if isinstance(data, dict):
        data = data.get("entries", [])
    if not isinstance(data, list):
        return entries
    for item in data:
        if not isinstance(item, dict):
            continue
        aliases = item.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        entries.append(
            OntologyEntry(
                code=str(item.get("code", "")),
                name=str(item.get("name") or item.get("description") or ""),
                system=str(item.get("system") or item.get("code_system") or ""),
                concept_type=str(item.get("concept_type") or item.get("type") or ""),
                aliases=tuple(str(x) for x in aliases if x),
                priority=int(item.get("priority", 100)),
            )
        )
    return entries


def entries_from_csv(path: Path, system: str, concept_type: str) -> list[OntologyEntry]:
    entries: list[OntologyEntry] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lower = {k.casefold(): v for k, v in row.items() if k is not None}
            code = lower.get("code") or lower.get("rxcui") or lower.get("id") or ""
            name = lower.get("description") or lower.get("name") or lower.get("term") or ""
            aliases_raw = lower.get("aliases") or lower.get("synonym") or lower.get("synonyms") or ""
            aliases = [part.strip() for part in aliases_raw.replace("|", ";").split(";") if part.strip()]
            if code and name:
                entries.append(OntologyEntry(code, name, system, concept_type, tuple(aliases), 50))
    return entries


def entries_from_txt(path: Path, system: str, concept_type: str) -> list[OntologyEntry]:
    entries: list[OntologyEntry] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                entries.append(OntologyEntry(parts[0], parts[1], system, concept_type, (), 60))
    return entries


def load_external_entries(raw_dir: Path, external_dir: Path) -> list[OntologyEntry]:
    entries: list[OntologyEntry] = []
    for base in (raw_dir, external_dir):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.casefold()
            name = path.name.casefold()
            if suffix == ".json":
                entries.extend(entries_from_json(path))
            elif suffix == ".csv":
                if "rx" in name:
                    entries.extend(entries_from_csv(path, SYSTEM_RXNORM, TYPE_DRUG))
                elif "icd" in name:
                    entries.extend(entries_from_csv(path, SYSTEM_ICD10, TYPE_DIAGNOSIS))
            elif suffix == ".txt":
                if "rx" in name:
                    entries.extend(entries_from_txt(path, SYSTEM_RXNORM, TYPE_DRUG))
                elif "icd" in name:
                    entries.extend(entries_from_txt(path, SYSTEM_ICD10, TYPE_DIAGNOSIS))
    return entries


def build_ontology_index(raw_dir: Path, external_dir: Path, out_path: Path) -> OntologyIndex:
    entries = seed_entries() + load_external_entries(raw_dir, external_dir)
    index = OntologyIndex(entries)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "entries": index.to_json_data(),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def load_ontology_index(index_path: Path | None = None, raw_dir: Path | None = None, external_dir: Path | None = None) -> OntologyIndex:
    if index_path and index_path.exists():
        entries = entries_from_json(index_path)
        if entries:
            return OntologyIndex(entries)
    entries = seed_entries()
    if raw_dir is not None and external_dir is not None:
        entries.extend(load_external_entries(raw_dir, external_dir))
    return OntologyIndex(entries)
