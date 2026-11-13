from datetime import date

from etl import db, quality
from etl.run import load_file
from tests.conftest import raw_record, write_zip


def _status(checks, name):
    return next(c.status for c in checks if c.name == name)


def _db(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    load_file(conn, "2023q1-0001", write_zip(tmp_path / "a.zip", [raw_record(f"A-{i}") for i in range(5)]))
    return conn


def test_clean_load_passes(tmp_path):
    report = quality.summary(quality.run_checks(_db(tmp_path), today=date(2026, 1, 1)))
    assert report["status"] == "pass", report


def test_empty_database_fails(tmp_path):
    checks = quality.run_checks(db.connect(tmp_path / "empty.db"))
    assert quality.summary(checks)["status"] == "fail"


def test_orphans_future_dates_and_bad_numbers_fail(tmp_path):
    conn = _db(tmp_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("INSERT INTO reactions (aer_id, term) VALUES ('ghost', 'Vomiting')")
    conn.execute("UPDATE reports SET receive_date = '2030-01-01' WHERE aer_id = 'A-1'")
    conn.execute("UPDATE reports SET weight_kg = -3 WHERE aer_id = 'A-2'")
    conn.commit()
    checks = quality.run_checks(conn, today=date(2026, 1, 1))
    assert _status(checks, "orphan_child_rows") == "fail"
    assert _status(checks, "future_receive_dates") == "fail"
    assert _status(checks, "implausible_age_or_weight") == "fail"


def test_low_label_coverage_warns_then_fails(tmp_path):
    conn = _db(tmp_path)
    conn.execute("UPDATE reports SET serious = NULL WHERE aer_id IN ('A-0', 'A-1')")  # 60% labelled
    conn.commit()
    assert _status(quality.run_checks(conn), "serious_label_coverage") == "warn"
    conn.execute("UPDATE reports SET serious = NULL WHERE aer_id = 'A-2'")  # 40% labelled
    conn.commit()
    assert _status(quality.run_checks(conn), "serious_label_coverage") == "fail"
