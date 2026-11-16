"""Combine the recorded validation labels without changing the annotations."""
from __future__ import annotations

import csv
from pathlib import Path

RCT_LABELS = {
    "PRIMARY_RCT_REPORT": "PRIMARY_RCT_REPORT",
    "NOT_PRIMARY_RCT_REPORT": "NOT_PRIMARY_RCT_REPORT",
    "TRUE": "PRIMARY_RCT_REPORT", "FALSE": "NOT_PRIMARY_RCT_REPORT",
    "UNCLEAR": "UNCLEAR", "": "",
}
ENDPOINT_LABELS = {
    "TRUE": "POSITIVE", "FALSE": "NEGATIVE", "POSITIVE": "POSITIVE",
    "NEGATIVE": "NEGATIVE", "UNCLEAR": "UNCLEAR", "NA": "NOT_APPLICABLE",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def harmonize(rct_label: str, endpoint_label: str) -> tuple[str, str]:
    rct = RCT_LABELS[rct_label]
    if endpoint_label == "":
        endpoint = "NOT_APPLICABLE" if rct == "NOT_PRIMARY_RCT_REPORT" else ""
    else:
        endpoint = ENDPOINT_LABELS[endpoint_label]
    return rct, endpoint


def combine_annotations(source: list[dict], publications: list[dict]) -> list[dict]:
    titles = {r["pmid"]: r["title"] for r in publications}
    index = {}
    for row in source:
        key = row["pmid"], row["annotator"]
        if key in index:
            raise ValueError(f"Duplicate annotation: {key}")
        if row["pmid"] not in titles or row["annotator"] not in {"PW", "AP", "LLM"}:
            raise ValueError(f"Unknown publication or annotator: {key}")
        index[key] = row
    pmids = sorted({r["pmid"] for r in source}, key=int)
    combined = []
    for pmid in pmids:
        record = {"pmid": pmid, "title": titles[pmid]}
        for annotator in ["PW", "AP", "LLM"]:
            if (pmid, annotator) not in index:
                raise ValueError(f"Missing {annotator} annotation for {pmid}")
            row = index[pmid, annotator]
            rct, endpoint = harmonize(row["rct_label"], row["primary_endpoint_success"])
            record[f"{annotator.lower()}_rct"] = rct
            record[f"{annotator.lower()}_primary_endpoint"] = endpoint
        combined.append(record)
    return combined
