"""Check released data, recorded requests, and validation annotations offline."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from common import (
    build_responses_request_body, build_user_input, canonical_json_hash,
    extract_nct_ids, json_schema_for_batch, load_json, load_prompt,
    make_input_hash, sha256_file,
)
from export_history_cohort import verify_history_export
from prepare_inputs import rebuild
from primary_endpoint_common import (
    build_endpoint_user_input, endpoint_json_schema, load_endpoint_config,
    make_endpoint_input_hash,
)
from validation import read_csv, validate_annotations

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_requests(publications: dict, task: str) -> int:
    config = load_json(ROOT / "config.json") if task == "rct" else load_endpoint_config(ROOT / "primary_endpoint_config.json")
    provenance = read_csv(ROOT / f"results/provenance/{task}.csv")
    require(len(provenance) == len({r["pmid"] for r in provenance}), f"Duplicate {task} provenance PMIDs")
    settings = config["model_arms"]["sol"]
    prompt, prompt_hash = load_prompt(ROOT / config["classification_prompt"])
    schema = json_schema_for_batch() if task == "rct" else endpoint_json_schema()
    user_input = build_user_input if task == "rct" else build_endpoint_user_input
    input_hash = make_input_hash if task == "rct" else make_endpoint_input_hash
    for row in provenance:
        publication = publications[row["pmid"]]
        body = build_responses_request_body(
            config, model=settings["model"], reasoning_effort=settings["reasoning_effort"],
            input_items=[{"role": "system", "content": prompt}, {"role": "user", "content": user_input(publication)}],
            text_format=schema,
        )
        for field, value in [
            ("prompt_hash", prompt_hash), ("schema_hash", canonical_json_hash(schema)),
            ("input_hash", input_hash(publication)), ("request_hash", canonical_json_hash(body)),
            ("requested_model", settings["model"]), ("reasoning_effort", settings["reasoning_effort"]),
        ]:
            require(row[field] == value, f"{task} {field} mismatch for PMID {row['pmid']}")
        require(row["error"] == "", f"Unresolved study technical failure: {row['pmid']}")
    return len(provenance)


def verify() -> dict:
    manifest = load_json(ROOT / "results/manifest.json")
    for relative, checksum in manifest["files"].items():
        path = ROOT / relative
        require(path.is_file() and sha256_file(path) == checksum, f"Release checksum mismatch: {relative}")
    ready = read_csv(ROOT / "data/processed/publications_ready.csv")
    publications = {r["pmid"]: r for r in ready}
    require(len(ready) == len(publications) == 7552, "Expected 7,552 unique input PMIDs")
    rebuilt, missing = rebuild(ROOT / "data/raw/pubmed_publications.csv", ROOT / "data/raw/pubmed_abstracts.txt")
    require(not missing and rebuilt == ready, "The original downloads do not reproduce the study input")
    rct = read_csv(ROOT / "results/rct_classifications.csv")
    endpoint = read_csv(ROOT / "results/primary_endpoint_assessments.csv")
    require(len(rct) == 7552 and {r["pmid"] for r in rct} == set(publications), "RCT PMID reconciliation failed")
    require(Counter(r["final_label"] for r in rct) == {"PRIMARY_RCT_REPORT": 2178, "NOT_PRIMARY_RCT_REPORT": 5352, "UNCLEAR": 22}, "RCT label counts changed")
    eligible = {r["pmid"] for r in rct if r["final_label"] == "PRIMARY_RCT_REPORT"}
    require(len(endpoint) == 2178 and {r["pmid"] for r in endpoint} == eligible, "Endpoint PMID reconciliation failed")
    require(Counter(r["primary_endpoint_result"] for r in endpoint) == {"POSITIVE": 1239, "NEGATIVE": 657, "UNCLEAR": 282}, "Endpoint label counts changed")
    for row in rct:
        nct = extract_nct_ids(publications[row["pmid"]]["abstract"]) if row["pmid"] in eligible else []
        expected_flag = ("TRUE" if nct else "FALSE") if row["pmid"] in eligible else ""
        require(row["nct_abstract_regex_found"] == expected_flag and row["nct_abstract_registry_numbers"] == "; ".join(nct), f"NCT regex mismatch: {row['pmid']}")
    requests = {task: check_requests(publications, task) for task in ["rct", "primary_endpoint"]}
    require(requests == {"rct": 7552, "primary_endpoint": 2178}, "Request count mismatch")
    validation = read_csv(ROOT / "data/validation/annotations.csv")
    require(len(validation) == 50, "Expected 50 validation publications")
    validate_annotations(validation, ready)
    handoff = verify_history_export(ROOT / "results/registry_handoff")
    require(handoff["selected_publications"] == 2178 and handoff["pmid_nct_associations"] == 1890 and handoff["unique_candidate_nct_ids"] == 1759, "Registry handoff counts changed")
    return {"input_publications": len(ready), "recorded_requests_verified": sum(requests.values()), "validation_publications": len(validation), "registry_associations": handoff["pmid_nct_associations"]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
