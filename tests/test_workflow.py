from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import parse_batch_results
from batch_workflow import prepare_run, collect_retry_queue, build_stage_chunks
from common import load_json, extract_nct_ids
from export_history_cohort import export_history_cohort, verify_history_export
from finalize_rct import finalize_run
from finalize_endpoints import finalize_endpoint_run
from parse_primary_endpoint_results import parse_endpoint_results
from primary_endpoint_workflow import prepare_endpoint_run, verify_endpoint_run_integrity
from validation import harmonize, combine_annotations


def response(custom_id, payload, model="gpt-5.6-sol"):
    return {
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": {
            "id": "resp_test", "model": model, "status": "completed",
            "output_text": json.dumps(payload),
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                      "output_tokens_details": {"reasoning_tokens": 5}},
        }},
    }


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.config = load_json(ROOT / "config.json")
        self.config["classification_prompt"] = str(ROOT / self.config["classification_prompt"])
        self.config_path = self.directory / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        rows = [
            {"pmid": "101", "title": "A randomized trial", "abstract": "Randomized groups were compared. Primary endpoint P=.01. NCT01234567.", "journal": "JAMA", "doi": "", "publication_year": "2025", "publication_date": "2025/01/01"},
            {"pmid": "102", "title": "Secondary randomized trial results", "abstract": "A secondary outcome was compared between randomized groups.", "journal": "BMJ", "doi": "", "publication_year": "2025", "publication_date": "2025/02/01"},
            {"pmid": "103", "title": "An observational study", "abstract": "Observational associations, mentioning NCT76543210.", "journal": "Lancet", "doi": "", "publication_year": "2025", "publication_date": "2025/03/01"},
        ]
        self.source = self.directory / "input.csv"
        pd.DataFrame(rows).to_csv(self.source, index=False)
        self.run = prepare_run(self.config_path, self.source, self.directory / "runs", "test_rct")
        self.chunk = self.run / "sol/chunk_0001/attempt_01"

    def tearDown(self):
        self.temp.cleanup()

    def payload(self, pmid):
        positive = pmid != "103"
        return {
            "final_label": "PRIMARY_RCT_REPORT" if positive else "NOT_PRIMARY_RCT_REPORT",
            "underlying_trial_randomized": "YES" if positive else "NO",
            "publication_role": "SECONDARY_OR_FOLLOWUP" if pmid == "102" else ("PRIMARY_RESULTS" if positive else "NONRANDOMIZED_OR_OTHER"),
            "trial_name_or_acronym": "", "registration_ids": [],
            "brief_rationale": "Randomized groups are compared." if positive else "An observational study.",
        }

    def write_responses(self, chunk=None, missing=(), wrong_model=None):
        chunk = chunk or self.chunk
        manifest = pd.read_csv(chunk / "manifest.csv", dtype=str).fillna("")
        records = [response(r["custom_id"], self.payload(r["pmid"]), wrong_model or "gpt-5.6-sol") for r in manifest.to_dict(orient="records") if r["pmid"] not in missing]
        (chunk / "output.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        (chunk / "errors.jsonl").write_text("")

    def parse(self, chunk=None):
        chunk = chunk or self.chunk
        argv = ["parse_batch_results.py", "--stage", "sol", "--batch-output", str(chunk / "output.jsonl"), "--batch-errors", str(chunk / "errors.jsonl"), "--manifest", str(chunk / "manifest.csv"), "--source", str(chunk / "source.csv"), "--output", str(chunk / "results.csv"), "--technical-retry-queue", str(chunk / "technical_retry_queue.csv"), "--config", str(self.run / "config_snapshot.json")]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            parse_batch_results.main()
        return pd.read_csv(chunk / "results.csv", dtype=str).fillna("")

    def finalized(self):
        self.write_responses()
        self.parse()
        return finalize_run(self.run)["sol"]

    def endpoint_run(self):
        config = load_json(ROOT / "primary_endpoint_config.json")
        config["classification_prompt"] = str(ROOT / config["classification_prompt"])
        path = self.directory / "endpoint_config.json"
        path.write_text(json.dumps(config))
        return prepare_endpoint_run(config_path=path, rct_run_dir=self.run, output_root=self.directory / "endpoint", run_id="test_endpoint")

    def test_complete_rct_endpoint_and_registry_workflow(self):
        final = self.finalized().set_index("pmid")
        self.assertEqual(final.loc["101", "nct_abstract_regex_found"], "TRUE")
        self.assertEqual(final.loc["102", "nct_abstract_regex_found"], "FALSE")
        self.assertEqual(final.loc["103", "nct_abstract_regex_found"], "")
        endpoint = self.endpoint_run()
        chunk = endpoint / "sol/chunk_0001/attempt_01"
        manifest = pd.read_csv(chunk / "manifest.csv", dtype=str).fillna("")
        self.assertEqual(set(manifest.pmid), {"101", "102"})
        requests = [json.loads(line) for line in (chunk / "input.jsonl").read_text().splitlines()]
        for request in requests:
            user_input = request["body"]["input"][1]["content"]
            self.assertNotIn("brief_rationale", user_input)
            self.assertNotIn("PRIMARY_RCT_REPORT", user_input)
        records = []
        for row in manifest.to_dict(orient="records"):
            explicit = row["pmid"] == "101"
            payload = {"primary_endpoint_result": "POSITIVE" if explicit else "UNCLEAR", "primary_endpoint_description": "The stated primary endpoint" if explicit else "", "trial_framework": "SUPERIORITY" if explicit else "UNCLEAR", "unclear_reason": "NOT_APPLICABLE" if explicit else "PRIMARY_ENDPOINT_NOT_EXPLICIT", "brief_rationale": "Primary endpoint criterion met." if explicit else "Primary endpoint is not identified."}
            records.append(response(row["custom_id"], payload))
        (chunk / "output.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        (chunk / "errors.jsonl").write_text("")
        parse_endpoint_results(arm="sol", batch_output_path=chunk / "output.jsonl", batch_errors_path=chunk / "errors.jsonl", manifest_path=chunk / "manifest.csv", source_path=chunk / "source.csv", output_path=chunk / "results.csv", technical_retry_queue_path=chunk / "technical_retry_queue.csv", config_path=endpoint / "config_snapshot.json", run_dir=endpoint)
        endpoints = finalize_endpoint_run(endpoint)["sol"]
        self.assertEqual(set(endpoints.pmid), {"101", "102"})
        self.assertEqual(set(endpoints.primary_endpoint_result), {"POSITIVE", "UNCLEAR"})
        export = export_history_cohort(run_dir=self.run, output_root=self.directory / "exports")
        checks = verify_history_export(export)
        self.assertEqual(checks["selected_publications"], 2)
        self.assertEqual(checks["selected_publications_without_nct"], 1)

    def test_missing_response_remains_unclear_and_retries_on_same_model(self):
        self.write_responses(missing={"102"})
        parsed = self.parse().set_index("pmid")
        self.assertEqual(parsed.loc["102", "final_label"], "UNCLEAR")
        self.assertEqual(parsed.loc["102", "technical_retry_pending"], "True")
        with self.assertRaises(ValueError):
            finalize_run(self.run)
        queue, _ = collect_retry_queue(run_dir=self.run, stage="sol", attempt_number=1)
        chunks = build_stage_chunks(run_dir=self.run, run_id="test_rct", stage="sol", attempt_number=2, requested=queue)
        request = json.loads((chunks[0] / "input.jsonl").read_text().splitlines()[0])
        self.assertEqual(request["body"]["model"], "gpt-5.6-sol")
        self.write_responses(chunks[0])
        self.parse(chunks[0])
        self.assertEqual(len(finalize_run(self.run)["sol"]), 3)

    def test_failed_response_cannot_become_an_exclusion(self):
        self.write_responses()
        records = [json.loads(line) for line in (self.chunk / "output.jsonl").read_text().splitlines()]
        for row in records:
            row["response"] = {"status_code": 500, "body": {"error": {"code": "server_error", "message": "Temporary failure"}}}
        (self.chunk / "output.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
        parsed = self.parse()
        self.assertEqual(set(parsed.final_label), {"UNCLEAR"})
        self.assertTrue(parsed.error.ne("").all())

    def test_unknown_response_id_is_rejected(self):
        self.write_responses()
        with (self.chunk / "output.jsonl").open("a") as handle:
            handle.write(json.dumps(response("unknown", self.payload("101"))) + "\n")
        with self.assertRaises(ValueError):
            self.parse()

    def test_duplicate_input_and_overwritten_run_are_rejected(self):
        with self.assertRaises(FileExistsError):
            prepare_run(self.config_path, self.source, self.directory / "runs", "test_rct")
        rows = pd.read_csv(self.source, dtype=str)
        pd.concat([rows, rows.iloc[:1]]).to_csv(self.source, index=False)
        with self.assertRaises(ValueError):
            prepare_run(self.config_path, self.source, self.directory / "runs", "duplicates")

    def test_changed_parent_is_detected(self):
        self.finalized()
        endpoint = self.endpoint_run()
        with (self.run / "sol/final/all_classifications.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaises(ValueError):
            verify_endpoint_run_integrity(endpoint)

    def test_chunks_cover_each_source_pmid_once(self):
        self.config["model_arms"]["sol"]["chunk_size"] = 2
        self.config_path.write_text(json.dumps(self.config))
        run = prepare_run(self.config_path, self.source, self.directory / "runs", "two_chunks")
        manifests = sorted((run / "sol").glob("chunk_*/attempt_01/manifest.csv"))
        self.assertEqual(len(manifests), 2)
        tables = [pd.read_csv(path, dtype=str) for path in manifests]
        pmids = [pmid for table in tables for pmid in table.pmid]
        self.assertEqual(pmids, ["101", "102", "103"])
        self.assertEqual(len(pmids), len(set(pmids)))


class AnnotationTests(unittest.TestCase):
    def test_uncertainty_and_applicability_remain_distinct(self):
        self.assertEqual(harmonize("TRUE", "UNCLEAR"), ("PRIMARY_RCT_REPORT", "UNCLEAR"))
        self.assertEqual(harmonize("FALSE", ""), ("NOT_PRIMARY_RCT_REPORT", "NOT_APPLICABLE"))
        self.assertEqual(harmonize("TRUE", ""), ("PRIMARY_RCT_REPORT", ""))
        self.assertEqual(harmonize("TRUE", "FALSE"), ("PRIMARY_RCT_REPORT", "NEGATIVE"))

    def test_duplicate_annotations_are_not_silently_combined(self):
        row = {"pmid": "1", "annotator": "PW", "rct_label": "TRUE", "primary_endpoint_success": "UNCLEAR"}
        with self.assertRaises(ValueError):
            combine_annotations([row, row], [{"pmid": "1", "title": "Title"}])

    def test_nct_regex_does_not_guess_malformed_identifiers(self):
        self.assertEqual(extract_nct_ids("nct01234567; NCT-76543210; NCT1234567; NCT123456789"), ["NCT01234567", "NCT76543210"])


if __name__ == "__main__":
    unittest.main()
