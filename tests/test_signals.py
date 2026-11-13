import math

import pytest

from etl import db
from etl.run import load_file
from stats.signals import compute, disproportionality, is_signal, save
from tests.conftest import raw_record, write_zip


def test_hand_computed_two_by_two():
    # a=10 b=90 c=20 d=880, N=1000
    s = disproportionality(10, 90, 20, 880)
    assert s["prr"] == pytest.approx((10 / 100) / (20 / 900))           # 4.5
    assert s["ror"] == pytest.approx((10 * 880) / (90 * 20))            # 4.889
    se = math.sqrt(1 / 10 + 1 / 90 + 1 / 20 + 1 / 880)
    assert s["ror_lo"] == pytest.approx(math.exp(math.log(4.8889) - 1.96 * se), rel=1e-3)  # 2.22
    assert s["ror_hi"] == pytest.approx(math.exp(math.log(4.8889) + 1.96 * se), rel=1e-3)  # 10.77
    # Yates: 1000 * (|8800 - 1800| - 500)^2 / (100 * 900 * 30 * 970) = 16.13
    assert s["chi2"] == pytest.approx(1000 * 6500 ** 2 / (100 * 900 * 30 * 970))


def test_zero_cells_do_not_crash():
    s = disproportionality(3, 0, 0, 10)
    assert s["prr"] is None and s["ror"] is None


@pytest.mark.parametrize("a, prr, chi2, expected", [
    (3, 2.0, 4.0, True), (2, 5.0, 10.0, False), (5, 1.9, 10.0, False), (5, 3.0, 3.9, False),
    (5, None, 10.0, False),
])
def test_evans_criteria(a, prr, chi2, expected):
    assert is_signal(a, prr, chi2) is expected


def _report(aer_id, ingredient, terms):
    rec = raw_record(aer_id, reaction=[{"veddra_term_name": t} for t in terms])
    rec["drug"][0]["active_ingredients"] = [{"name": ingredient}]
    return rec


def test_sql_counts_match_a_hand_count(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    records = [
        _report("R1", "Meloxicam", ["Vomiting"]),
        _report("R2", "Meloxicam", ["Vomiting", "Diarrhoea"]),
        _report("R3", "Ivermectin", ["Vomiting"]),
        _report("R4", "Ivermectin", ["Ataxia"]),
    ]
    load_file(conn, "2023q1-0001", write_zip(tmp_path / "a.zip", records))
    rows = {(r["ingredient"], r["term"]): r for r in compute(conn, min_a=1)}
    assert {k: rows[("meloxicam", "Vomiting")][k] for k in "abcd"} == {"a": 2, "b": 0, "c": 1, "d": 1}
    assert {k: rows[("ivermectin", "Ataxia")][k] for k in "abcd"} == {"a": 1, "b": 1, "c": 0, "d": 2}
    assert ("meloxicam", "Ataxia") not in rows

    save(conn, list(rows.values()), "species=all;from=all")
    save(conn, list(rows.values()), "species=all;from=all")  # re-run replaces, no duplicates
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == len(rows)


def test_species_filter(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cat = _report("R2", "Meloxicam", ["Vomiting"])
    cat["animal"] = {"species": "Cat"}
    load_file(conn, "2023q1-0001", write_zip(tmp_path / "a.zip", [_report("R1", "Meloxicam", ["Vomiting"]), cat]))
    rows = compute(conn, min_a=1, species="Cat")
    assert [(r["ingredient"], r["a"], r["d"]) for r in rows] == [("meloxicam", 1, 0)]
