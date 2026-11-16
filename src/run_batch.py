"""Submit, collect, or prepare retries for one saved Batch run."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from batch_workflow import build_stage_chunks, collect_retry_queue
from common import load_json
from primary_endpoint_workflow import build_endpoint_chunks, collect_endpoint_retries


def command(script, *arguments):
    subprocess.run([sys.executable, str(Path(__file__).with_name(script)), *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["submit", "collect", "retry"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.chunk < 1 or args.attempt < 1:
        parser.error("chunk and attempt must be positive")
    run = Path(args.run_dir)
    manifest = load_json(run / "run_manifest.json")
    endpoint = manifest["workflow_mode"] == "primary_endpoint_single_model_arm"
    if manifest["workflow_mode"] not in {"single_model_arm", "primary_endpoint_single_model_arm"} or manifest["active_arms"] != ["sol"]:
        raise ValueError("Expected a single-model sol run.")
    if args.action == "retry":
        attempt_dirs = sorted((run / "sol").glob(f"chunk_*/attempt_{args.attempt:02d}"))
        if not attempt_dirs or any(not (p / "results.csv").is_file() for p in attempt_dirs):
            raise ValueError("Collect and parse every chunk before preparing retries.")
        if endpoint:
            queue, _ = collect_endpoint_retries(run_dir=run, arm="sol", attempt_number=args.attempt)
        else:
            queue, _ = collect_retry_queue(run_dir=run, stage="sol", attempt_number=args.attempt)
        if queue.empty:
            print("No technical retries remain. Finalize the run.")
            return 0
        if endpoint:
            built = build_endpoint_chunks(run_dir=run, arm="sol", attempt_number=args.attempt + 1, requested=queue)
        else:
            built = build_stage_chunks(run_dir=run, run_id=manifest["run_id"], stage="sol", attempt_number=args.attempt + 1, requested=queue)
        print(f"Prepared {len(built)} retry chunks for attempt {args.attempt + 1}.")
        return 0
    chunk = run / "sol" / f"chunk_{args.chunk:04d}" / f"attempt_{args.attempt:02d}"
    if not (chunk / "manifest.csv").is_file():
        raise FileNotFoundError(f"No prepared chunk: {chunk}")
    stage = "endpoint_sol" if endpoint else "sol"
    if args.action == "submit":
        command("submit_batch.py", "--input", chunk / "input.jsonl", "--job-file", chunk / "submission.json", "--run-id", manifest["run_id"], "--stage", stage, "--chunk", f"{args.chunk:04d}", "--attempt", f"{args.attempt:02d}")
        return 0
    command("check_batch.py", "--job-file", chunk / "submission.json")
    if not (chunk / "output.jsonl").exists() or not (chunk / "errors.jsonl").exists():
        command("download_batch_results.py", "--job-file", chunk / "submission.json", "--output", chunk / "output.jsonl", "--error-output", chunk / "errors.jsonl")
    if (chunk / "results.csv").exists():
        print("This chunk has already been collected and parsed.")
        return 0
    arguments = ["--batch-output", chunk / "output.jsonl", "--batch-errors", chunk / "errors.jsonl", "--manifest", chunk / "manifest.csv", "--source", chunk / "source.csv", "--output", chunk / "results.csv", "--technical-retry-queue", chunk / "technical_retry_queue.csv", "--config", run / "config_snapshot.json"]
    if endpoint:
        command("parse_primary_endpoint_results.py", "--arm", "sol", "--run-dir", run, *arguments)
    else:
        command("parse_batch_results.py", "--stage", "sol", *arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
