"""Pipeline entry point: download -> parse -> load -> quality checks.

    python -m etl.run --from-year 2016 --to-year 2026
    python -m etl.run --quarters 2024q1            # one quarter (used in CI)
    python -m etl.run --local tests/data/x.zip     # offline file, no download
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

from etl import db, download, parse, quality

log = logging.getLogger("etl")


def load_file(conn, key: str, path: Path, file_url: str = "", force: bool = False) -> Dict:
    sha = download.sha256(path)
    if not force and db.loaded_sha(conn, key) == sha:
        return {"key": key, "skipped": True}
    raw = list(parse.iter_raw(path))
    good, rejected = parse.parse_many(raw, partition_key=key)
    stats = db.load_partition(conn, key, good, sha, len(raw), sum(rejected.values()), file_url)
    return {"key": key, "skipped": False, "in": len(raw), **stats, "reject_reasons": rejected}


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("data/ads.db"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--from-year", type=int, default=2016)
    ap.add_argument("--to-year", type=int, default=2100)
    ap.add_argument("--quarters", default="", help="comma list, e.g. 2024q1,2024q2")
    ap.add_argument("--local", type=Path, nargs="*", default=[], help="load local zip files instead")
    ap.add_argument("--force", action="store_true", help="reload even if the file is unchanged")
    ap.add_argument("--refresh", action="store_true", help="re-download files already on disk")
    ap.add_argument("--quality-report", type=Path, default=Path("reports/quality_report.json"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    conn = db.connect(args.db)
    started = time.time()
    if args.local:
        jobs = [(p.name.split(".")[0], p, "") for p in args.local]
    else:
        index = download.fetch_index()
        quarters = [q.strip().lower() for q in args.quarters.split(",") if q.strip()]
        parts = download.select(index["partitions"], args.from_year, args.to_year, quarters)
        log.info("openFDA export %s: %d partitions selected", index.get("export_date"), len(parts))
        jobs = [(p["key"], download.download(p, args.raw_dir, args.refresh), p["file"]) for p in parts]

    for key, path, url in jobs:
        result = load_file(conn, key, path, url, args.force)
        if result["skipped"]:
            log.info("%s unchanged, skipped", key)
        else:
            log.info("%s in=%d loaded=%d replaced=%d rejected=%d %s", key, result["in"],
                     result["loaded"], result["replaced"], result["rejected"],
                     result["reject_reasons"] or "")

    report = quality.summary(quality.run_checks(conn))
    report["reports_total"] = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    report["partitions_total"] = conn.execute("SELECT COUNT(*) FROM partitions").fetchone()[0]
    report["seconds"] = round(time.time() - started, 1)
    args.quality_report.parent.mkdir(parents=True, exist_ok=True)
    args.quality_report.write_text(json.dumps(report, indent=2) + "\n")
    for c in report["checks"]:
        log.info("quality %-30s %-6s value=%s rule=%s", c["name"], c["status"], c["value"], c["rule"])
    log.info("quality overall: %s (%d reports)", report["status"], report["reports_total"])
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
