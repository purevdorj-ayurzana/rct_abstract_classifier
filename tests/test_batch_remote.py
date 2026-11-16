from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import download_batch_results  # noqa: E402


class FakeBatch:
    def __init__(self, status: str, output_file_id: str | None, error_file_id: str | None) -> None:
        self.id = "batch_test"
        self.status = status
        self.output_file_id = output_file_id
        self.error_file_id = error_file_id

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "output_file_id": self.output_file_id,
            "error_file_id": self.error_file_id,
        }


class FakeClient:
    def __init__(self, batch: FakeBatch, files: dict[str, str]) -> None:
        self.batches = SimpleNamespace(retrieve=lambda _batch_id: batch)
        self.files = SimpleNamespace(
            content=lambda file_id: SimpleNamespace(text=files[file_id])
        )


class BatchRemoteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.job = self.directory / "submission.json"
        self.job.write_text(json.dumps({"id": "batch_test"}), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def call_download(self, client: FakeClient) -> int:
        argv = [
            "download_batch_results.py",
            "--job-file", str(self.job),
            "--output", str(self.directory / "output.jsonl"),
            "--error-output", str(self.directory / "errors.jsonl"),
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(download_batch_results, "OpenAI", return_value=client),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test"}),
        ):
            return download_batch_results.main()

    def test_expired_batch_downloads_partial_output_and_errors(self) -> None:
        client = FakeClient(
            FakeBatch("expired", "file-output", "file-errors"),
            {"file-output": "success\n", "file-errors": "expired\n"},
        )
        self.assertEqual(self.call_download(client), 0)
        self.assertEqual(
            (self.directory / "output.jsonl").read_text(encoding="utf-8"),
            "success\n",
        )
        self.assertEqual(
            (self.directory / "errors.jsonl").read_text(encoding="utf-8"),
            "expired\n",
        )
        history = (self.directory / "status_history.jsonl").read_text(encoding="utf-8")
        self.assertIn('"status": "expired"', history)

    def test_terminal_batch_without_files_gets_empty_placeholders(self) -> None:
        client = FakeClient(FakeBatch("failed", None, None), {})
        self.assertEqual(self.call_download(client), 0)
        self.assertEqual((self.directory / "output.jsonl").stat().st_size, 0)
        self.assertEqual((self.directory / "errors.jsonl").stat().st_size, 0)

    def test_nonterminal_batch_is_not_downloaded(self) -> None:
        client = FakeClient(FakeBatch("in_progress", None, None), {})
        self.assertEqual(self.call_download(client), 1)
        self.assertFalse((self.directory / "output.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
