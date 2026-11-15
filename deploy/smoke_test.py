"""Smoke test of a running API (local or the AWS Lambda function URL).

Usage: python deploy/smoke_test.py BASE_URL [EXPECTED_MODEL_VERSION]
Checks that /health reports the expected model and that /triage scores a report with it.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

REPORT = {"species": "Dog", "age_years": 11, "reporter": "Veterinarian", "type_info": "Safety Issue",
          "routes": ["Oral"], "ingredients": ["Meloxicam"], "reactions": ["Vomiting", "Melaena"]}


def call(url: str, body: Optional[Dict] = None) -> Dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main(argv: List[str], tries: int = 6, wait: float = 5.0) -> int:
    base = argv[1].rstrip("/")
    expected = argv[2] if len(argv) > 2 else None
    for attempt in range(1, tries + 1):  # cold starts and code updates take a few seconds
        try:
            health = call(base + "/health")
            if health.get("model_loaded") and (expected is None or health.get("model_version") == expected):
                break
            problem = f"unexpected /health: {health}"
        except (urllib.error.URLError, OSError) as e:
            problem = f"/health failed: {e}"
        print(f"attempt {attempt}/{tries}: {problem}")
        if attempt == tries:
            return 1
        time.sleep(wait)

    result = call(base + "/triage", REPORT)
    p = result.get("serious_probability")
    if result.get("route") != "model" or not isinstance(p, float) or not 0 <= p <= 1:
        print(f"unexpected /triage: {result}")
        return 1
    if expected and result.get("model_version") != expected:
        print(f"/triage answered with model {result.get('model_version')}, expected {expected}")
        return 1
    print(f"ok: model {health['model_version']}, example report scored {p} (flagged={result['flagged']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
