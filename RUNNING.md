# Running the pipeline

Run commands from the repository root. Install `requirements.txt` and make
`OPENAI_API_KEY` available in your environment before submitting or collecting
API jobs. The saved study data require no API calls.

## Prepare input

The study input is already included. To rebuild it from the original downloads:

```text
python src/prepare_inputs.py
python src/validate_inputs.py
```

Rebuilt files go to `outputs/prepared_input`; the included input is preserved.

## Classify publications

```text
python src/batch_workflow.py prepare
```

Use the printed run directory in place of `RUN` below. Preparation is offline.
Each chunk contains up to 500 publications. Process chunks sequentially,
changing `--chunk` to the next number after each has completed.

```text
python src/run_batch.py submit --run-dir RUN --chunk 1
python src/run_batch.py collect --run-dir RUN --chunk 1
```

Submission incurs API charges. Collection succeeds once the Batch job is
terminal; otherwise, try collection again later. Available partial responses
are retained, including from expired jobs.

After collecting every chunk:

```text
python src/run_batch.py retry --run-dir RUN --attempt 1
```

If retries were prepared, submit and collect those chunks with `--attempt 2`.
Repeat the retry command with `--attempt 2` if needed; at most two technical
retries are used. Scientific uncertainty is not retried. Then finalize:

```text
python src/finalize_rct.py --run-dir RUN
```

## Assess primary-endpoint success

The parent RCT run must pass its checks with no unresolved technical failures.
All eligible publications proceed, regardless of publication role.

```text
python src/primary_endpoint_workflow.py prepare --rct-run-dir RUN
```

Use the newly printed endpoint run directory with the same `run_batch.py`
submit, collect, and retry commands. Finalize it with:

```text
python src/finalize_endpoints.py --run-dir ENDPOINT_RUN
```

## Export registry candidates

```text
python src/export_history_cohort.py --run-dir RUN
```

The export includes every eligible publication and a separate PMID–NCT table.
It makes no registry requests. The study's final handoff, including its two
documented corrections, is already included in `results/registry_handoff`.

Runs retain requests, responses, retries, usage, and checks in separate output
directories. Existing request and submission evidence is not overwritten.
