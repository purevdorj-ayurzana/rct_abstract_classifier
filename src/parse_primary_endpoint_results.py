from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import pandas as pd
from pydantic import ValidationError
from common import response_incomplete_reason, response_output_text, utc_now_iso
from parse_batch_results import error_json, index_batch_records, is_retryable_error, load_jsonl, validate_table_identity
from primary_endpoint_common import configured_endpoint_arms, endpoint_response_usage, load_endpoint_config
from primary_endpoint_schemas import PrimaryEndpointAssessment
from primary_endpoint_workflow import validate_endpoint_source, verify_endpoint_run_integrity

def endpoint_unclear_row(base: dict[str, Any], *, error: str, error_code: str, retryable: bool, usage: dict[str, int] | None=None) -> dict[str, Any]:
    return {**base, 'primary_endpoint_result': 'UNCLEAR', 'primary_endpoint_description': '', 'trial_framework': 'UNCLEAR', 'unclear_reason': 'OTHER', 'brief_rationale': 'Primary-endpoint assessment unavailable because the API response failed integrity or parsing checks.', 'error_code': error_code, 'retryable': str(retryable), **(usage or {'input_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'total_tokens': 0}), 'error': error}

def parse_endpoint_results(*, arm: str, batch_output_path: Path, batch_errors_path: Path, manifest_path: Path, source_path: Path, output_path: Path, technical_retry_queue_path: Path | None, config_path: Path, run_dir: Path | None=None) -> pd.DataFrame:
    if run_dir is not None:
        verify_endpoint_run_integrity(run_dir)
    config = load_endpoint_config(config_path)
    if arm not in configured_endpoint_arms(config):
        raise ValueError(f'Unknown endpoint arm {arm!r}; expected {configured_endpoint_arms(config)}.')
    max_technical_retries = int(config['max_technical_retries'])
    outputs = load_jsonl(batch_output_path)
    batch_errors = load_jsonl(batch_errors_path)
    manifest = pd.read_csv(manifest_path, dtype=str).fillna('')
    source = validate_endpoint_source(pd.read_csv(source_path, dtype=str).fillna(''))
    validate_table_identity(manifest, name='Endpoint manifest', required={'custom_id', 'pmid'})
    manifest['custom_id'] = manifest['custom_id'].astype(str).str.strip()
    manifest['pmid'] = manifest['pmid'].astype(str).str.strip()
    if manifest['custom_id'].duplicated().any():
        raise ValueError('Endpoint manifest contains duplicate custom_id values.')
    if 'arm' in manifest.columns:
        manifest_arms = set(manifest['arm'].astype(str).str.strip())
        if manifest_arms != {arm}:
            raise ValueError(f'Endpoint manifest arm values {sorted(manifest_arms)} do not match --arm {arm!r}.')
    if 'task' in manifest.columns:
        tasks = set(manifest['task'].astype(str).str.strip())
        if tasks != {'prespecified_primary_endpoint'}:
            raise ValueError(f'Unexpected endpoint manifest task values: {tasks}')
    manifest_ids = set(manifest['custom_id'])
    output_map = index_batch_records(outputs, source_name=str(batch_output_path), manifest_ids=manifest_ids)
    error_map = index_batch_records(batch_errors, source_name=str(batch_errors_path), manifest_ids=manifest_ids)
    overlap = set(output_map) & set(error_map)
    if overlap:
        raise ValueError(f'The same endpoint custom_id appears in output and error files: {sorted(overlap)[:5]}')
    source_map = {row['pmid']: row for row in source.to_dict(orient='records')}
    missing_source = set(manifest['pmid']) - set(source_map)
    if missing_source:
        raise ValueError(f'Endpoint manifest PMIDs are missing from source: {sorted(missing_source)[:5]}')
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest.to_dict(orient='records'):
        custom_id = manifest_row['custom_id']
        pmid = manifest_row['pmid']
        source_row = source_map[pmid]
        record = output_map.get(custom_id) or error_map.get(custom_id)
        response = record.get('response') if record else None
        batch_error = record.get('error') if record else None
        status = response.get('status_code') if response else None
        body = response.get('body', {}) if response else {}
        if not isinstance(body, dict):
            body = {}
        base = {**source_row, 'run_id': manifest_row.get('run_id', ''), 'endpoint_arm': arm, 'stage': arm, 'chunk_number': manifest_row.get('chunk_number', ''), 'attempt_number': manifest_row.get('attempt_number', '1') or '1', 'custom_id': custom_id, 'model': body.get('model', manifest_row.get('requested_model', '')), 'requested_model': manifest_row.get('requested_model', ''), 'actual_model': body.get('model', ''), 'reasoning_effort': manifest_row.get('reasoning_effort', ''), 'prompt_hash': manifest_row.get('prompt_hash', ''), 'schema_hash': manifest_row.get('schema_hash', ''), 'input_hash': manifest_row.get('input_hash', ''), 'request_hash': manifest_row.get('request_hash', ''), 'response_id': body.get('id', ''), 'http_status_code': status if status is not None else '', 'parsed_at_utc': utc_now_iso()}
        if record is None:
            rows.append(endpoint_unclear_row(base, error_code='MISSING_BATCH_RESPONSE', retryable=True, error=error_json('MISSING_BATCH_RESPONSE', 'Endpoint manifest entry was found in neither Batch output nor Batch errors.')))
            continue
        if batch_error:
            request_error_code = str(batch_error.get('code', 'BATCH_REQUEST_ERROR') if isinstance(batch_error, dict) else 'BATCH_REQUEST_ERROR')
            rows.append(endpoint_unclear_row(base, error_code=request_error_code, retryable=is_retryable_error(request_error_code, status), error=error_json('BATCH_REQUEST_ERROR', 'The Batch API returned an endpoint request-level error.', batch_error), usage=endpoint_response_usage(body)))
            continue
        if status != 200:
            rows.append(endpoint_unclear_row(base, error_code='BATCH_HTTP_ERROR', retryable=is_retryable_error('BATCH_HTTP_ERROR', status), error=error_json('BATCH_HTTP_ERROR', f'The endpoint Batch response had HTTP status {status!r}.', body), usage=endpoint_response_usage(body)))
            continue
        incomplete_reason = response_incomplete_reason(body)
        if incomplete_reason:
            error_code = 'MAX_OUTPUT_TOKENS' if incomplete_reason == 'max_output_tokens' else 'RESPONSE_INCOMPLETE'
            rows.append(endpoint_unclear_row(base, error_code=error_code, retryable=incomplete_reason == 'max_output_tokens', error=error_json(error_code, 'The Responses API returned an incomplete response.', {'reason': incomplete_reason}), usage=endpoint_response_usage(body)))
            continue
        try:
            assessment = PrimaryEndpointAssessment.model_validate_json(response_output_text(body))
            rows.append({**base, 'primary_endpoint_result': assessment.primary_endpoint_result.value, 'primary_endpoint_description': assessment.primary_endpoint_description, 'trial_framework': assessment.trial_framework.value, 'unclear_reason': assessment.unclear_reason.value, 'brief_rationale': assessment.brief_rationale, 'error_code': '', 'retryable': 'False', **endpoint_response_usage(body), 'error': ''})
        except (ValueError, ValidationError) as exc:
            rows.append(endpoint_unclear_row(base, error_code='STRUCTURED_OUTPUT_PARSE_ERROR', retryable=True, error=error_json('STRUCTURED_OUTPUT_PARSE_ERROR', 'The successful API response could not be parsed into the primary-endpoint schema.', f'{type(exc).__name__}: {exc}'), usage=endpoint_response_usage(body)))
    parsed = pd.DataFrame(rows)
    if len(parsed) != len(manifest):
        raise RuntimeError(f'Endpoint integrity failure: parsed {len(parsed)} rows for {len(manifest)} manifest entries.')
    if parsed['pmid'].duplicated().any():
        raise RuntimeError('Endpoint parsed results contain duplicate PMIDs.')
    if set(parsed['pmid']) != set(manifest['pmid']):
        raise RuntimeError('Endpoint parsed PMID set differs from its manifest.')
    attempts = pd.to_numeric(parsed['attempt_number'], errors='coerce').fillna(1).astype(int)
    retries_used = attempts - 1
    technical_failure = parsed['error'].ne('')
    retryable = parsed['retryable'].astype(str).str.casefold().eq('true')
    retry_pending = technical_failure & retryable & (retries_used < max_technical_retries)
    if technical_retry_queue_path is None:
        retry_pending = pd.Series(False, index=parsed.index)
    parsed['technical_retry_pending'] = retry_pending.map(str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(output_path, index=False, encoding='utf-8-sig')
    if technical_retry_queue_path is not None:
        retry_metadata = parsed.loc[retry_pending, ['pmid', 'error', 'error_code', 'attempt_number', 'custom_id']].rename(columns={'error': 'retry_source_error', 'error_code': 'retry_source_error_code', 'attempt_number': 'retry_source_attempt_number', 'custom_id': 'retry_source_custom_id'})
        retry_queue = source.merge(retry_metadata, on='pmid', how='inner', validate='one_to_one')
        if len(retry_queue) != int(retry_pending.sum()):
            raise RuntimeError('Endpoint technical retry queue lost PMIDs.')
        technical_retry_queue_path.parent.mkdir(parents=True, exist_ok=True)
        retry_queue.to_csv(technical_retry_queue_path, index=False, encoding='utf-8-sig')
    return parsed

def main() -> int:
    parser = argparse.ArgumentParser(description='Reconcile and parse primary-endpoint Batch responses against their authoritative manifest.')
    parser.add_argument('--arm', required=True)
    parser.add_argument('--batch-output', required=True)
    parser.add_argument('--batch-errors', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--technical-retry-queue', default=None)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', default=None)
    args = parser.parse_args()
    parsed = parse_endpoint_results(arm=args.arm, batch_output_path=Path(args.batch_output), batch_errors_path=Path(args.batch_errors), manifest_path=Path(args.manifest), source_path=Path(args.source), output_path=Path(args.output), technical_retry_queue_path=Path(args.technical_retry_queue) if args.technical_retry_queue else None, config_path=Path(args.config), run_dir=Path(args.run_dir) if args.run_dir else None)
    output_count = len(load_jsonl(Path(args.batch_output)))
    error_count = len(load_jsonl(Path(args.batch_errors)))
    print(f'Reconciled {len(parsed)} endpoint manifest entries to {args.output} ({output_count} output, {error_count} error, {len(parsed) - output_count - error_count} missing).')
    if args.technical_retry_queue:
        pending = parsed['technical_retry_pending'].str.casefold().eq('true')
        print(f'Endpoint technical retry queue: {int(pending.sum())} records saved to {args.technical_retry_queue}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
