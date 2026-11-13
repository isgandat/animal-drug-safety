from etl import db
from etl.run import load_file
from tests.conftest import raw_record, write_zip

TABLES = ("reports", "drugs", "ingredients", "reactions", "outcomes")


def counts(conn):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


def test_reloading_the_same_file_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    path = write_zip(tmp_path / "a.zip", [raw_record("A-1"), raw_record("A-2")])
    first = load_file(conn, "2023q1-0001", path)
    before = counts(conn)
    assert first["loaded"] == 2 and not first["skipped"]
    assert load_file(conn, "2023q1-0001", path)["skipped"]
    forced = load_file(conn, "2023q1-0001", path, force=True)
    assert forced["replaced"] == 2
    assert counts(conn) == before == {"reports": 2, "drugs": 2, "ingredients": 2,
                                      "reactions": 2, "outcomes": 2}


def test_updated_report_replaces_its_old_child_rows(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    old = write_zip(tmp_path / "old.zip", [raw_record("A-1")])
    new = write_zip(tmp_path / "new.zip", [raw_record(
        "A-1", reaction=[{"veddra_term_name": "Seizure NOS"}, {"veddra_term_name": "Ataxia"}])])
    load_file(conn, "2023q1-0001", old)
    result = load_file(conn, "2023q2-0001", new)
    terms = [r[0] for r in conn.execute("SELECT term FROM reactions ORDER BY term")]
    assert result["replaced"] == 1
    assert terms == ["Ataxia", "Seizure NOS"]
    assert counts(conn)["reports"] == 1


def test_duplicate_id_inside_one_file_keeps_the_last_version(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    path = write_zip(tmp_path / "a.zip", [raw_record("A-1", serious_ae="false"),
                                          raw_record("A-1", serious_ae="true")])
    load_file(conn, "2023q1-0001", path)
    assert [tuple(r) for r in conn.execute("SELECT serious FROM reports")] == [(1,)]


def test_partition_bookkeeping(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    path = write_zip(tmp_path / "a.zip", [raw_record("A-1"), raw_record(None)])
    load_file(conn, "2023q1-0001", path)
    row = conn.execute("SELECT records_in, records_loaded, records_rejected FROM partitions").fetchone()
    assert tuple(row) == (2, 1, 1)
    assert db.loaded_sha(conn, "2023q1-0001") is not None
