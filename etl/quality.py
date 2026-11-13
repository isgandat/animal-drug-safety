"""Data-quality checks that run after every load.

Each check returns pass / warn / fail. A 'fail' makes the pipeline exit
non-zero so CI (or a scheduler) notices.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import date
from typing import List, Optional


@dataclass
class Check:
    name: str
    value: float
    rule: str
    status: str  # pass | warn | fail


def _scalar(conn: sqlite3.Connection, sql: str):
    return conn.execute(sql).fetchone()[0]


def _grade(value: float, ok, warn=None) -> str:
    if ok(value):
        return "pass"
    if warn is not None and warn(value):
        return "warn"
    return "fail"


def run_checks(conn: sqlite3.Connection, today: Optional[date] = None) -> List[Check]:
    today = today or date.today()
    n = _scalar(conn, "SELECT COUNT(*) FROM reports")
    checks = [Check("reports_loaded", n, "> 0", "pass" if n > 0 else "fail")]
    if n == 0:
        return checks

    orphans = sum(
        _scalar(conn, f"SELECT COUNT(*) FROM {t} c WHERE NOT EXISTS "
                      f"(SELECT 1 FROM reports r WHERE r.aer_id = c.aer_id)")
        for t in ("drugs", "ingredients", "reactions", "outcomes"))
    checks.append(Check("orphan_child_rows", orphans, "== 0", "pass" if orphans == 0 else "fail"))

    future = _scalar(conn, f"SELECT COUNT(*) FROM reports WHERE receive_date > '{today.isoformat()}'")
    checks.append(Check("future_receive_dates", future, "== 0", "pass" if future == 0 else "fail"))

    wrong_q = _scalar(conn, "SELECT AVG(receive_year != CAST(substr(partition_key, 1, 4) AS INTEGER)) "
                            "FROM reports")
    checks.append(Check("year_outside_partition_share", round(wrong_q, 4), "<= 0.01",
                        _grade(wrong_q, lambda v: v <= 0.01, lambda v: v <= 0.05)))

    def share(sql_condition: str) -> float:
        return round(_scalar(conn, f"SELECT AVG({sql_condition}) FROM reports"), 4)

    labelled = share("serious IS NOT NULL")
    checks.append(Check("serious_label_coverage", labelled, ">= 0.80",
                        _grade(labelled, lambda v: v >= 0.80, lambda v: v >= 0.60)))

    with_reaction = share("EXISTS (SELECT 1 FROM reactions x WHERE x.aer_id = reports.aer_id)")
    checks.append(Check("reports_with_reaction", with_reaction, ">= 0.95",
                        _grade(with_reaction, lambda v: v >= 0.95, lambda v: v >= 0.80)))

    with_drug = share("EXISTS (SELECT 1 FROM drugs d WHERE d.aer_id = reports.aer_id)")
    checks.append(Check("reports_with_drug", with_drug, ">= 0.95",
                        _grade(with_drug, lambda v: v >= 0.95, lambda v: v >= 0.80)))

    for column in ("species", "age_years", "weight_kg"):
        missing = share(f"{column} IS NULL")
        checks.append(Check(f"missing_{column}", missing, "<= 0.30 (warn above)",
                            _grade(missing, lambda v: v <= 0.30, lambda v: v <= 0.60)))

    bad_numbers = _scalar(conn, "SELECT COUNT(*) FROM reports WHERE age_years < 0 OR age_years > 60 "
                                "OR weight_kg <= 0 OR weight_kg > 3000")
    checks.append(Check("implausible_age_or_weight", bad_numbers, "== 0",
                        "pass" if bad_numbers == 0 else "fail"))

    rejected = _scalar(conn, "SELECT CAST(SUM(records_rejected) AS REAL) / MAX(SUM(records_in), 1) "
                             "FROM partitions") or 0.0
    checks.append(Check("rejected_record_share", round(rejected, 4), "<= 0.01",
                        _grade(rejected, lambda v: v <= 0.01, lambda v: v <= 0.05)))
    return checks


def summary(checks: List[Check]) -> dict:
    return {
        "status": "fail" if any(c.status == "fail" for c in checks) else
                  "warn" if any(c.status == "warn" for c in checks) else "pass",
        "checks": [asdict(c) for c in checks],
    }
