from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from openai import OpenAI
from common import append_jsonl, utc_now_iso

def main() -> int:
    parser = argparse.ArgumentParser(description='Check an OpenAI batch job.')
    parser.add_argument('--job-file', required=True)
    parser.add_argument('--status-history', default=None)
    args = parser.parse_args()
    if not os.getenv('OPENAI_API_KEY'):
        print('OPENAI_API_KEY is not set.', file=sys.stderr)
        return 2
    job_path = Path(args.job_file)
    job = json.loads(job_path.read_text(encoding='utf-8'))
    client = OpenAI()
    batch = client.batches.retrieve(job['id'])
    history_path = Path(args.status_history) if args.status_history else job_path.with_name('status_history.jsonl')
    append_jsonl(history_path, {'checked_at_utc': utc_now_iso(), 'action': 'status_check', **batch.model_dump()})
    print(f'Batch: {batch.id}')
    print(f'Status: {batch.status}')
    if batch.request_counts:
        c = batch.request_counts
        print(f'Requests: total={c.total}, completed={c.completed}, failed={c.failed}')
    print(f'Output file ID: {batch.output_file_id or '[not ready]'}')
    print(f'Error file ID: {batch.error_file_id or '[none]'}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
