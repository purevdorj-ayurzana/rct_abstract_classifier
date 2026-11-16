from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from openai import OpenAI
from common import atomic_write_json, sha256_file, utc_now_iso

def main() -> int:
    parser = argparse.ArgumentParser(description='Submit an OpenAI Batch API job.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--job-file', required=True)
    parser.add_argument('--description', default='RCT journal classifier')
    parser.add_argument('--run-id', default='')
    parser.add_argument('--stage', default='')
    parser.add_argument('--chunk', default='')
    parser.add_argument('--attempt', default='')
    args = parser.parse_args()
    if not os.getenv('OPENAI_API_KEY'):
        print('OPENAI_API_KEY is not set.', file=sys.stderr)
        return 2
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Batch input not found: {input_path}')
    job_path = Path(args.job_file)
    if job_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing submission record: {job_path}')
    client = OpenAI()
    with input_path.open('rb') as handle:
        uploaded = client.files.create(file=handle, purpose='batch')
    batch = client.batches.create(input_file_id=uploaded.id, endpoint='/v1/responses', completion_window='24h', metadata={'description': args.description, 'run_id': args.run_id, 'stage': args.stage, 'chunk': args.chunk, 'attempt': args.attempt})
    payload = batch.model_dump()
    payload.update({'submitted_at_utc_local_record': utc_now_iso(), 'local_input_path': str(input_path), 'local_input_sha256': sha256_file(input_path), 'uploaded_file_id': uploaded.id, 'run_id': args.run_id, 'stage': args.stage, 'chunk': args.chunk, 'attempt': args.attempt})
    atomic_write_json(job_path, payload)
    print(f'Batch submitted: {batch.id}')
    print(f'Initial status: {batch.status}')
    print(f'Job metadata saved to {job_path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
