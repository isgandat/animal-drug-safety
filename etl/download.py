"""Find and download openFDA animal & veterinary adverse-event partitions.

openFDA publishes the dataset as one zipped JSON file per calendar quarter.
The list of files lives in https://api.fda.gov/download.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional

INDEX_URL = "https://api.fda.gov/download.json"
USER_AGENT = "animal-drug-safety/0.1 (personal research project)"

_QUARTER = re.compile(r"/event/(\d{4})q([1-4])/")
_PART = re.compile(r"-(\d{4})-of-\d{4}")


def _get(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except OSError as exc:  # URLError, timeouts and connection resets are all OSError
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed after {retries} tries: {url}") from last


def fetch_index(index_url: str = INDEX_URL) -> Dict:
    data = json.loads(_get(index_url))
    return data["results"]["animalandveterinary"]["event"]


def partition_key(file_url: str) -> Optional[str]:
    """'.../event/2023q2/animalandveterinary-event-0001-of-0001.json.zip' -> '2023q2-0001'."""
    quarter = _QUARTER.search(file_url)
    if quarter is None:
        return None
    part = _PART.search(file_url)
    return f"{quarter.group(1)}q{quarter.group(2)}-{part.group(1) if part else '0001'}"


def select(partitions: Iterable[Dict], from_year: int, to_year: int,
           quarters: Optional[Iterable[str]] = None) -> List[Dict]:
    """Keep partitions inside [from_year, to_year]; `quarters` (e.g. {'2024q1'}) narrows further."""
    wanted = set(quarters or [])
    chosen = []
    for p in partitions:
        key = partition_key(p["file"])
        if key is None:
            continue
        if wanted and key.split("-")[0] not in wanted:
            continue
        if from_year <= int(key[:4]) <= to_year:
            chosen.append({**p, "key": key})
    return sorted(chosen, key=lambda p: p["key"])


def download(partition: Dict, raw_dir: Path, refresh: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{partition['key']}.json.zip"
    if path.exists() and path.stat().st_size > 0 and not refresh:
        return path
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(_get(partition["file"]))
    tmp.replace(path)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
