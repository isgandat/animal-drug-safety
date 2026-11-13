"""Feature building shared by training and the API.

A "case" is one adverse-event report flattened to plain values and lists.
`load_cases` builds cases from SQLite; the API builds them from JSON. Both go
through `report_features`, so training and serving cannot drift apart.
"""
from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

from etl.parse import normalize_ingredient

CATEGORICAL = ("species", "gender", "repro_status", "health_prior", "reporter", "type_info")

# Reaction terms that state the animal died. Seriousness is defined by such
# outcomes, so they would leak the label. Reports containing them are routed by
# a simple rule instead of the model (see `rule_serious`).
LEAKAGE = re.compile(r"death|dead|died|euthan|fatal|mortality", re.IGNORECASE)
FATAL_OUTCOMES = {"died", "euthanized"}

_SEP = "\x1f"


@dataclass
class Example:
    aer_id: str
    year: int
    label: Optional[int]
    case: Dict


def rule_serious(reactions: Iterable[str], outcomes: Iterable[str] = ()) -> bool:
    return (any(LEAKAGE.search(t or "") for t in reactions)
            or any((o or "").strip().lower() in FATAL_OUTCOMES for o in outcomes))


def report_features(case: Mapping) -> Dict[str, float]:
    f: Dict[str, float] = {}
    for key in CATEGORICAL:
        f[f"{key}={case.get(key) or 'missing'}"] = 1.0

    for key, cap in (("age_years", 60.0), ("weight_kg", 3000.0)):
        value = case.get(key)
        if value is None:
            f[f"{key}_missing"] = 1.0
        else:
            f[f"{key}_log"] = math.log1p(min(max(float(value), 0.0), cap))

    treated = case.get("treated")
    f[f"treated={'missing' if treated is None else bool(treated)}"] = 1.0
    used = case.get("used_per_label")
    f[f"used_per_label={'missing' if used is None else bool(used)}"] = 1.0

    affected = case.get("n_affected")
    f["n_affected_log"] = math.log1p(min(affected, 100000)) if affected else 0.0
    f["n_drugs_log"] = math.log1p(case.get("n_drugs") or 0)

    for prefix, key in (("route", "routes"), ("admin", "administered_by"), ("off_label", "off_label")):
        for value in case.get(key) or []:
            f[f"{prefix}={value}"] = 1.0
    for code in case.get("atc_codes") or []:
        if code and code.upper().startswith("Q") and len(code) >= 5:
            f[f"atc={code[:5].upper()}"] = 1.0  # ATCvet level 3, e.g. QP53B
    for raw in case.get("ingredients") or []:
        for name in normalize_ingredient(raw):
            f[f"ingr={name}"] = 1.0
    kept = [t for t in case.get("reactions") or [] if t and not LEAKAGE.search(t)]
    for term in kept:
        f[f"rx={term}"] = 1.0
    f["n_reactions_log"] = math.log1p(len(kept))
    return f


def _split(value: Optional[str], sep: str = "|") -> List[str]:
    if not value:
        return []
    out: List[str] = []
    for item in value.split(sep):
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


_CASES_SQL = f"""
SELECT r.aer_id, r.receive_year, r.serious,
       r.species, r.gender, r.repro_status, r.health_prior, r.reporter, r.type_info,
       r.age_years, r.weight_kg, r.treated, r.n_affected,
       (SELECT COUNT(*) FROM drugs d WHERE d.aer_id = r.aer_id) AS n_drugs,
       (SELECT group_concat(route, '|') FROM drugs d WHERE d.aer_id = r.aer_id) AS routes,
       (SELECT group_concat(administered_by, '|') FROM drugs d WHERE d.aer_id = r.aer_id) AS admin,
       (SELECT group_concat(off_label, '|') FROM drugs d WHERE d.aer_id = r.aer_id) AS off_label,
       (SELECT group_concat(atc_code, '|') FROM drugs d WHERE d.aer_id = r.aer_id) AS atc,
       (SELECT MIN(used_per_label) FROM drugs d WHERE d.aer_id = r.aer_id) AS used_per_label,
       (SELECT group_concat(name, '|') FROM ingredients i WHERE i.aer_id = r.aer_id) AS ingredients,
       (SELECT group_concat(term, '{_SEP}') FROM reactions x WHERE x.aer_id = r.aer_id) AS reactions,
       (SELECT group_concat(status, '|') FROM outcomes o WHERE o.aer_id = r.aer_id) AS outcomes
FROM reports r
WHERE r.serious IS NOT NULL {{where}}
ORDER BY r.receive_date, r.aer_id
"""


def load_cases(conn: sqlite3.Connection, where: str = "", params: tuple = ()) -> List[Example]:
    """All labelled reports as Examples (optionally filtered with extra SQL)."""
    out: List[Example] = []
    for row in conn.execute(_CASES_SQL.format(where=where), params):
        used = row["used_per_label"]
        case = {
            "species": row["species"], "gender": row["gender"], "repro_status": row["repro_status"],
            "health_prior": row["health_prior"], "reporter": row["reporter"],
            "type_info": row["type_info"], "age_years": row["age_years"],
            "weight_kg": row["weight_kg"], "treated": row["treated"],
            "n_affected": row["n_affected"], "n_drugs": row["n_drugs"],
            "used_per_label": None if used is None else bool(used),
            "routes": _split(row["routes"]), "administered_by": _split(row["admin"]),
            "off_label": _split(row["off_label"]), "atc_codes": _split(row["atc"]),
            "ingredients": _split(row["ingredients"]),
            "reactions": _split(row["reactions"], _SEP), "outcomes": _split(row["outcomes"]),
        }
        out.append(Example(row["aer_id"], row["receive_year"], row["serious"], case))
    return out
