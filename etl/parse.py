"""Turn raw openFDA records into clean rows.

Raw records are messy: numbers come as strings, some fields are sometimes a
string and sometimes a list, units vary and ingredient names carry strengths
and dosage forms. Everything is normalised here so the database and the model
see one consistent shape.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

_AGE_TO_YEARS = {
    "year": 1.0,
    "month": 1 / 12,
    "week": 7 / 365.25,
    "day": 1 / 365.25,
    "hour": 1 / 8766,
    "minute": 1 / 525960,
}
_WEIGHT_TO_KG = {"kilogram": 1.0, "gram": 0.001, "pound": 0.45359237}
MAX_AGE_YEARS = 60.0
MAX_WEIGHT_KG = 3000.0

# Words that describe the product form or salt, not the active substance.
_FORM_WORDS = {
    "as", "salt", "base", "chew", "chews", "chewable", "tablet", "tablets", "tab",
    "solution", "spot-on", "spot", "on", "injection", "injectable", "vial", "suspension",
    "oral", "topical", "for", "dogs", "cats", "dog", "cat", "week", "month",
    "pamoate", "maleate", "citrate", "hydrochloride", "hcl", "sodium", "potassium",
    "acetate", "sulfate", "sulphate", "phosphate", "oxime", "mesylate", "tartrate",
}


@dataclass
class Parsed:
    report: Dict
    drugs: List[Dict] = field(default_factory=list)
    ingredients: List[Dict] = field(default_factory=list)
    reactions: List[Dict] = field(default_factory=list)
    outcomes: List[Dict] = field(default_factory=list)


class Rejected(ValueError):
    """Raised when a record is unusable; the message is the reason."""


def iter_raw(zip_path: Path) -> Iterator[Dict]:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                with zf.open(name) as fh:
                    payload = json.load(fh)
                yield from payload.get("results", [])


def as_bool(value) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "true":
            return 1
        if text == "false":
            return 0
    return None


def as_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0:  # NaN or negative
        return None
    return number


def as_int(value) -> Optional[int]:
    number = as_float(value)
    return int(number) if number is not None else None


def as_text(value) -> Optional[str]:
    """Scalar text. A few merged records hold lists (often the same value
    twice); those are de-duplicated and joined with '|'."""
    if value is None:
        return None
    if isinstance(value, list):
        return "|".join(as_list(value)) or None
    text = str(value).strip()
    return text or None


def as_list(value) -> List[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out = []
    for item in items:
        text = str(item).strip() if item is not None else ""
        if text and text not in out:
            out.append(text)
    return out


def age_in_years(age: Optional[Dict]) -> Optional[float]:
    if not isinstance(age, dict):
        return None
    value = as_float(age.get("min"))
    unit = (age.get("unit") or "").strip().lower()
    if value is None or unit not in _AGE_TO_YEARS:
        return None
    years = value * _AGE_TO_YEARS[unit]
    return round(years, 4) if years <= MAX_AGE_YEARS else None


def weight_in_kg(weight: Optional[Dict]) -> Optional[float]:
    if not isinstance(weight, dict):
        return None
    value = as_float(weight.get("min"))
    unit = (weight.get("unit") or "").strip().lower()
    if value is None or unit not in _WEIGHT_TO_KG:
        return None
    kg = value * _WEIGHT_TO_KG[unit]
    return round(kg, 3) if 0 < kg <= MAX_WEIGHT_KG else None


def normalize_ingredient(name: Optional[str]) -> List[str]:
    """'Pyrantel Pamoate;Sarolaner' -> ['pyrantel', 'sarolaner'].

    Splits combinations, drops strengths ('13.64%'), dosage forms and salts.
    Idempotent: running it on its own output returns the same names.
    """
    names: List[str] = []
    for part in re.split(r"[;/,+]", name or ""):
        text = re.split(r"\d", part.lower(), maxsplit=1)[0]
        text = re.sub(r"[^a-z\s-]", " ", text)
        words = [w for w in text.split() if w not in _FORM_WORDS]
        clean = " ".join(words).strip(" -")
        if len(clean) >= 3 and clean not in names:
            names.append(clean)
    return names


def receive_date(raw, today: Optional[date] = None) -> date:
    if isinstance(raw, list):  # merged records: keep the earliest date
        values = sorted(str(v).strip() for v in raw if v)
        raw = values[0] if values else None
    text = (raw or "").strip()
    if not re.fullmatch(r"\d{8}", text):
        raise Rejected("bad_receive_date")
    try:
        value = datetime.strptime(text, "%Y%m%d").date()
    except ValueError as exc:
        raise Rejected("bad_receive_date") from exc
    if value > (today or date.today()) or value.year < 1980:
        raise Rejected("receive_date_out_of_range")
    return value


def parse_record(raw: Dict, partition_key: str = "", today: Optional[date] = None) -> Parsed:
    aer_id = as_text(raw.get("unique_aer_id_number"))
    if not aer_id:
        raise Rejected("missing_id")
    received = receive_date(raw.get("original_receive_date"), today)

    animal = raw.get("animal") or {}
    health = raw.get("health_assessment_prior_to_exposure")
    breed = animal.get("breed") or {}
    report = {
        "aer_id": aer_id,
        "report_id": as_text(raw.get("report_id")),
        "partition_key": partition_key,
        "receive_date": received.isoformat(),
        "receive_year": received.year,
        "species": as_text(animal.get("species")),
        "gender": as_text(animal.get("gender")),
        "repro_status": as_text(animal.get("reproductive_status")),
        "crossbred": as_bool(breed.get("is_crossbred")) if isinstance(breed, dict) else None,
        "age_years": age_in_years(animal.get("age")),
        "weight_kg": weight_in_kg(animal.get("weight")),
        "health_prior": as_text(health.get("condition")) if isinstance(health, dict) else as_text(health),
        "reporter": as_text(raw.get("primary_reporter")),
        "type_info": as_text(raw.get("type_of_information")),
        "origin": as_text(raw.get("foreign_or_domestic")),
        "serious": as_bool(raw.get("serious_ae")),
        "treated": as_bool(raw.get("treated_for_ae")),
        "n_affected": as_int(raw.get("number_of_animals_affected")),
        "n_treated": as_int(raw.get("number_of_animals_treated")),
    }
    parsed = Parsed(report=report)

    for seq, drug in enumerate(raw.get("drug") or []):
        if not isinstance(drug, dict):
            continue
        parsed.drugs.append({
            "aer_id": aer_id,
            "seq": seq,
            "route": "|".join(as_list(drug.get("route"))) or None,
            "administered_by": "|".join(as_list(drug.get("administered_by"))) or None,
            "off_label": "|".join(as_list(drug.get("off_label_use"))) or None,
            "used_per_label": as_bool(drug.get("used_according_to_label")),
            "previous_exposure": as_bool(drug.get("previous_exposure_to_drug")),
            "dosage_form": as_text(drug.get("dosage_form")),
            "atc_code": as_text(drug.get("atc_vet_code")),
        })
        for ingredient in drug.get("active_ingredients") or []:
            raw_name = as_text((ingredient or {}).get("name"))
            for name in normalize_ingredient(raw_name):
                parsed.ingredients.append(
                    {"aer_id": aer_id, "drug_seq": seq, "name_raw": raw_name, "name": name})

    seen_terms = set()
    for reaction in raw.get("reaction") or []:
        term = as_text((reaction or {}).get("veddra_term_name"))
        if term and term not in seen_terms:
            seen_terms.add(term)
            parsed.reactions.append({
                "aer_id": aer_id,
                "term_code": as_text(reaction.get("veddra_term_code")),
                "term": term,
                "veddra_version": as_text(reaction.get("veddra_version")),
            })

    for outcome in raw.get("outcome") or []:
        status = as_text((outcome or {}).get("medical_status"))
        if status:
            parsed.outcomes.append({
                "aer_id": aer_id,
                "status": status,
                "n_animals": as_int(outcome.get("number_of_animals_affected")),
            })
    return parsed


def parse_many(records, partition_key: str = "",
               today: Optional[date] = None) -> Tuple[List[Parsed], Dict[str, int]]:
    """Parse a batch; returns (good records, rejection counts by reason)."""
    good: List[Parsed] = []
    rejected: Dict[str, int] = {}
    for raw in records:
        try:
            good.append(parse_record(raw, partition_key, today))
        except Rejected as exc:
            rejected[str(exc)] = rejected.get(str(exc), 0) + 1
        except (TypeError, AttributeError, KeyError):  # unexpected shape: count it, keep going
            rejected["malformed"] = rejected.get("malformed", 0) + 1
    return good, rejected
