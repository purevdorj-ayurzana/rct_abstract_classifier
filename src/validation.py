"""Read and check the combined validation table."""
from __future__ import annotations

import csv
from pathlib import Path

RCT_LABELS = {"PRIMARY_RCT_REPORT", "NOT_PRIMARY_RCT_REPORT", "UNCLEAR", ""}
ENDPOINT_LABELS = {"POSITIVE", "NEGATIVE", "UNCLEAR", "NOT_APPLICABLE", ""}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_annotations(annotations: list[dict], publications: list[dict]) -> None:
    titles = {r["pmid"]: r["title"] for r in publications}
    seen = set()
    for row in annotations:
        pmid = row["pmid"]
        if pmid in seen:
            raise ValueError(f"Duplicate validation PMID: {pmid}")
        if pmid not in titles or row["title"] != titles[pmid]:
            raise ValueError(f"Validation publication does not match the corpus: {pmid}")
        seen.add(pmid)
        for annotator in ["pw", "ap", "llm"]:
            if row[f"{annotator}_rct"] not in RCT_LABELS or row[f"{annotator}_primary_endpoint"] not in ENDPOINT_LABELS:
                raise ValueError(f"Invalid {annotator} validation label for PMID {pmid}")
