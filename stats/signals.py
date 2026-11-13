"""Disproportionality analysis (PRR / ROR) for ingredient x reaction pairs.

For each pair the 2x2 table counts reports:
                      reaction    other reactions
    ingredient            a             b
    other ingredients     c             d

A pair is flagged with the Evans et al. (2001) screening rule:
a >= 3, PRR >= 2 and chi-squared (Yates) >= 4.

A flag is a hypothesis for review. Spontaneous reports cannot show that a
drug caused an event: exposure counts are unknown and reporting is biased.

    python -m stats.signals --db data/ads.db [--species Dog] [--from-year 2020]
"""
from __future__ import annotations

import argparse
import logging
import math
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

from etl import db

log = logging.getLogger("signals")


def disproportionality(a: int, b: int, c: int, d: int) -> Dict[str, Optional[float]]:
    n = a + b + c + d
    prr = (a / (a + b)) / (c / (c + d)) if (a + b) and c and (c + d) else None
    ror = (a * d) / (b * c) if b and c else None
    ror_lo = ror_hi = None
    if ror and a and d:
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        ror_lo = math.exp(math.log(ror) - 1.96 * se)
        ror_hi = math.exp(math.log(ror) + 1.96 * se)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    chi2 = n * max(abs(a * d - b * c) - n / 2, 0) ** 2 / denom if denom else None
    return {"prr": prr, "ror": ror, "ror_lo": ror_lo, "ror_hi": ror_hi, "chi2": chi2}


def is_signal(a: int, prr: Optional[float], chi2: Optional[float]) -> bool:
    return a >= 3 and prr is not None and prr >= 2 and chi2 is not None and chi2 >= 4


_PAIRS_SQL = """
WITH universe AS (
    SELECT r.aer_id FROM reports r
    WHERE 1 = 1 {filters}
      AND EXISTS (SELECT 1 FROM ingredients i WHERE i.aer_id = r.aer_id)
      AND EXISTS (SELECT 1 FROM reactions x WHERE x.aer_id = r.aer_id)
),
drug_reports AS (SELECT DISTINCT i.name, i.aer_id FROM ingredients i JOIN universe u USING (aer_id)),
event_reports AS (SELECT DISTINCT x.term, x.aer_id FROM reactions x JOIN universe u USING (aer_id)),
pairs AS (
    SELECT d.name, e.term, COUNT(*) AS a
    FROM drug_reports d JOIN event_reports e USING (aer_id)
    GROUP BY d.name, e.term HAVING COUNT(*) >= :min_a
),
drug_n AS (SELECT name, COUNT(*) AS n_drug FROM drug_reports GROUP BY name),
event_n AS (SELECT term, COUNT(*) AS n_event FROM event_reports GROUP BY term)
SELECT p.name, p.term, p.a, dn.n_drug, en.n_event, (SELECT COUNT(*) FROM universe) AS n_total
FROM pairs p JOIN drug_n dn USING (name) JOIN event_n en USING (term)
"""


def compute(conn: sqlite3.Connection, min_a: int = 3, species: Optional[str] = None,
            from_year: Optional[int] = None) -> List[Dict]:
    filters, params = "", {"min_a": min_a}
    if species:
        filters += " AND r.species = :species"
        params["species"] = species
    if from_year:
        filters += " AND r.receive_year >= :from_year"
        params["from_year"] = from_year
    rows = []
    for name, term, a, n_drug, n_event, n_total in conn.execute(_PAIRS_SQL.format(filters=filters), params):
        b, c = n_drug - a, n_event - a
        d = n_total - a - b - c
        stats = disproportionality(a, b, c, d)
        rows.append({"ingredient": name, "term": term, "a": a, "b": b, "c": c, "d": d, **stats,
                     "signal": is_signal(a, stats["prr"], stats["chi2"])})
    return rows


def save(conn: sqlite3.Connection, rows: List[Dict], scope: str) -> None:
    with conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS signals (
            scope TEXT NOT NULL, ingredient TEXT NOT NULL, term TEXT NOT NULL,
            a INTEGER, b INTEGER, c INTEGER, d INTEGER,
            prr REAL, ror REAL, ror_lo REAL, ror_hi REAL, chi2 REAL, signal INTEGER,
            PRIMARY KEY (scope, ingredient, term))""")
        conn.execute("DELETE FROM signals WHERE scope = ?", (scope,))
        conn.executemany(
            "INSERT INTO signals VALUES (:scope, :ingredient, :term, :a, :b, :c, :d, "
            ":prr, :ror, :ror_lo, :ror_hi, :chi2, :signal)",
            [{**r, "scope": scope, "signal": int(r["signal"])} for r in rows])


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("data/ads.db"))
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--species")
    ap.add_argument("--from-year", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    conn = db.connect(args.db)
    scope = f"species={args.species or 'all'};from={args.from_year or 'all'}"
    rows = compute(conn, args.min_count, args.species, args.from_year)
    save(conn, rows, scope)
    log.info("%s: %d pairs with a >= %d, %d flagged", scope, len(rows), args.min_count,
             sum(r["signal"] for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
