from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from openai import OpenAI
from common import append_jsonl, utc_now_iso
TERMINAL_STATUSES = {'completed', 'failed', 'expired', 'cancelled'}

def save_remote(client: OpenAI, file_id: str, output: Path) -> None:
    response = client.files.content(file_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = response.text
    if output.exists():
        if output.read_text(encoding='utf-8') != content:
            raise FileExistsError(f'Refusing to overwrite different raw Batch data: {output}')
        return
    output.write_text(content, encoding='utf-8')

def ensure_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf-8'):
            raise FileExistsError(f'Expected an empty placeholder but found data: {path}')
        return
    path.write_text('', encoding='utf-8')

def main() -> int:
    parser = argparse.ArgumentParser(description='Download available files from a terminal Batch job.')
    parser.add_argument('--job-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--error-output', required=True)
    parser.add_argument('--status-history', default=None)
    args = parser.parse_args()
    if not os.getenv('OPENAI_API_KEY'):
        print('OPENAI_API_KEY is not set.', file=sys.stderr)
        return 2
    job = json.loads(Path(args.job_file).read_text(encoding='utf-8'))
    client = OpenAI()
    batch = client.batches.retrieve(job['id'])
    history_path = Path(args.status_history) if args.status_history else Path(args.job_file).with_name('status_history.jsonl')
    append_jsonl(history_path, {'checked_at_utc': utc_now_iso(), 'action': 'download_check', **batch.model_dump()})
    if batch.status not in TERMINAL_STATUSES:
        print(f'Batch status is {batch.status}; results are not terminal.', file=sys.stderr)
        return 1
    output_path = Path(args.output)
    if batch.output_file_id:
        save_remote(client, batch.output_file_id, output_path)
        print(f'Downloaded successful responses to {args.output}')
    else:
        ensure_empty(output_path)
        print('No successful-response file was present; created an empty placeholder.')
    error_path = Path(args.error_output)
    if batch.error_file_id:
        save_remote(client, batch.error_file_id, error_path)
        print(f'Downloaded failed responses to {args.error_output}')
    else:
        ensure_empty(error_path)
        print('No batch error file was present; created an empty placeholder.')
    print(f'Terminal batch status: {batch.status}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
