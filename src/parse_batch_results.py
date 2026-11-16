from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd
from pydantic import ValidationError
from common import configured_model_arms, extract_registry_ids, load_json, response_incomplete_reason, response_output_text, response_usage, utc_now_iso
from schemas import RCTClassification

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open('r', encoding='utf-8') as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON on line {number} of {path}: {exc}') from exc
            if not isinstance(record, dict):
                raise ValueError(f'JSONL line {number} of {path} is not an object.')
            records.append(record)
    return records

def index_batch_records(records: list[dict[str, Any]], *, source_name: str, manifest_ids: set[str]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        custom_id = str(record.get('custom_id') or '').strip()
        if not custom_id:
            raise ValueError(f'A record in {source_name} has no custom_id.')
        if custom_id not in manifest_ids:
            raise ValueError(f'Unknown custom_id {custom_id!r} found in {source_name}.')
        if custom_id in indexed:
            raise ValueError(f'Duplicate custom_id {custom_id!r} found in {source_name}.')
        indexed[custom_id] = record
    return indexed

def error_json(code: str, message: str, details: Any=None) -> str:
    payload: dict[str, Any] = {'code': code, 'message': message}
    if details not in (None, '', {}, []):
        payload['details'] = details
    return json.dumps(payload, ensure_ascii=False, default=str)

def is_retryable_error(code: str, status: Any=None) -> bool:
    normalized = str(code or '').casefold()
    transient_fragments = ('expired', 'rate_limit', 'timeout', 'server_error', 'internal_error', 'temporarily_unavailable', 'missing_batch_response', 'structured_output_parse_error')
    if any((fragment in normalized for fragment in transient_fragments)):
        return True
    try:
        http_status = int(status)
    except (TypeError, ValueError):
        return False
    return http_status in {408, 409, 429} or http_status >= 500

def unclear_row(base: dict[str, Any], *, error: str, error_code: str, retryable: bool, usage: dict[str, int] | None=None) -> dict[str, Any]:
    row = {**base, 'final_label': 'UNCLEAR', 'underlying_trial_randomized': 'UNCLEAR', 'publication_role': 'UNCLEAR', 'trial_name_or_acronym': '', 'registration_ids': '[]', 'brief_rationale': 'Classification unavailable because the API response failed integrity or parsing checks.', 'error_code': error_code, 'retryable': str(retryable), **(usage or {'input_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'total_tokens': 0}), 'error': error}
    return row

def validate_table_identity(table: pd.DataFrame, *, name: str, required: set[str]) -> None:
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f'{name} is missing columns: {sorted(missing)}')
    for column in required:
        if table[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'{name} contains empty {column} values.')
    if table['pmid'].duplicated().any():
        raise ValueError(f'{name} contains duplicate PMID values.')

def main() -> int:
    parser = argparse.ArgumentParser(description='Reconcile and parse Batch API output against its manifest.')
    parser.add_argument('--stage', required=True, help='Use sol.')
    parser.add_argument('--batch-output', required=True)
    parser.add_argument('--batch-errors', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--technical-retry-queue', default=None)
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()
    outputs = load_jsonl(Path(args.batch_output))
    batch_errors = load_jsonl(Path(args.batch_errors))
    manifest = pd.read_csv(args.manifest, dtype=str).fillna('')
    source = pd.read_csv(args.source, dtype=str).fillna('')
    config = load_json(args.config) if Path(args.config).is_file() else {}
    if args.stage not in configured_model_arms(config):
        raise ValueError(f'Unknown model arm {args.stage!r}; expected one of {configured_model_arms(config)}.')
    max_technical_retries = int(config.get('max_technical_retries', 0))
    validate_table_identity(manifest, name='Manifest', required={'custom_id', 'pmid'})
    validate_table_identity(source, name='Source CSV', required={'pmid'})
    manifest['custom_id'] = manifest['custom_id'].astype(str).str.strip()
    manifest['pmid'] = manifest['pmid'].astype(str).str.strip()
    source['pmid'] = source['pmid'].astype(str).str.strip()
    if manifest['custom_id'].duplicated().any():
        raise ValueError('Manifest contains duplicate custom_id values.')
    if 'stage' in manifest.columns:
        stages = set(manifest['stage'].astype(str).str.strip())
        if stages != {args.stage}:
            raise ValueError(f'Manifest stage values {sorted(stages)} do not match --stage {args.stage!r}.')
    manifest_ids = set(manifest['custom_id'])
    output_map = index_batch_records(outputs, source_name=str(args.batch_output), manifest_ids=manifest_ids)
    error_map = index_batch_records(batch_errors, source_name=str(args.batch_errors), manifest_ids=manifest_ids)
    overlap = set(output_map) & set(error_map)
    if overlap:
        raise ValueError(f'The same custom_id appears in both batch output and error files: {sorted(overlap)[:5]}')
    source_map = {r['pmid']: r for r in source.to_dict(orient='records')}
    missing_source = set(manifest['pmid']) - set(source_map)
    if missing_source:
        raise ValueError(f'Manifest PMIDs are missing from the source CSV: {sorted(missing_source)[:5]}')
    rows = []
    for m in manifest.to_dict(orient='records'):
        custom_id = m['custom_id']
        pmid = m['pmid']
        s = source_map[pmid]
        record = output_map.get(custom_id) or error_map.get(custom_id)
        response = record.get('response') if record else None
        batch_error = record.get('error') if record else None
        status = response.get('status_code') if response else None
        body = response.get('body', {}) if response else {}
        if not isinstance(body, dict):
            body = {}
        base = {'run_id': m.get('run_id', ''), 'pmid': pmid, 'title': s.get('title', ''), 'abstract': s.get('abstract', ''), 'doi': s.get('doi', ''), 'journal': s.get('journal', ''), 'publication_year': s.get('publication_year', ''), 'publication_date': s.get('publication_date', ''), 'publication_types': s.get('publication_types', ''), 'stage': args.stage, 'chunk_number': m.get('chunk_number', ''), 'attempt_number': m.get('attempt_number', '1') or '1', 'custom_id': custom_id, 'model': body.get('model', m.get('requested_model', m.get('model', ''))), 'requested_model': m.get('requested_model', m.get('model', '')), 'actual_model': body.get('model', ''), 'reasoning_effort': m.get('reasoning_effort', ''), 'prompt_hash': m.get('prompt_hash', ''), 'schema_hash': m.get('schema_hash', ''), 'input_hash': m.get('input_hash', ''), 'request_hash': m.get('request_hash', ''), 'response_id': body.get('id', ''), 'http_status_code': status if status is not None else '', 'parsed_at_utc': utc_now_iso()}
        if record is None:
            rows.append(unclear_row(base, error_code='MISSING_BATCH_RESPONSE', retryable=True, error=error_json('MISSING_BATCH_RESPONSE', 'Manifest entry was found in neither Batch output nor Batch errors.')))
            continue
        if batch_error:
            request_error_code = str(batch_error.get('code', 'BATCH_REQUEST_ERROR') if isinstance(batch_error, dict) else 'BATCH_REQUEST_ERROR')
            rows.append(unclear_row(base, error_code=request_error_code, retryable=is_retryable_error(request_error_code, status), error=error_json('BATCH_REQUEST_ERROR', 'The Batch API returned a request-level error.', batch_error), usage=response_usage(body)))
            continue
        if status != 200:
            rows.append(unclear_row(base, error_code='BATCH_HTTP_ERROR', retryable=is_retryable_error('BATCH_HTTP_ERROR', status), error=error_json('BATCH_HTTP_ERROR', f'The Batch API response had HTTP status {status!r}.', body), usage=response_usage(body)))
            continue
        incomplete_reason = response_incomplete_reason(body)
        if incomplete_reason:
            error_code = 'MAX_OUTPUT_TOKENS' if incomplete_reason == 'max_output_tokens' else 'RESPONSE_INCOMPLETE'
            rows.append(unclear_row(base, error_code=error_code, retryable=incomplete_reason == 'max_output_tokens', error=error_json(error_code, 'The Responses API returned an incomplete response.', {'reason': incomplete_reason}), usage=response_usage(body)))
            continue
        try:
            text = response_output_text(body)
            schema_class = RCTClassification
            classification = schema_class.model_validate_json(text)
            ids = list(dict.fromkeys(extract_registry_ids(f'{s.get('title', '')} {s.get('abstract', '')}') + classification.registration_ids))
            parsed_row = {**base, 'final_label': classification.final_label.value, 'underlying_trial_randomized': classification.underlying_trial_randomized.value, 'publication_role': classification.publication_role.value, 'trial_name_or_acronym': classification.trial_name_or_acronym, 'registration_ids': json.dumps(ids, ensure_ascii=False), 'brief_rationale': classification.brief_rationale, 'error_code': '', 'retryable': 'False', **response_usage(body), 'error': ''}
            rows.append(parsed_row)
        except (ValueError, ValidationError) as exc:
            rows.append(unclear_row(base, error_code='STRUCTURED_OUTPUT_PARSE_ERROR', retryable=True, error=error_json('STRUCTURED_OUTPUT_PARSE_ERROR', 'The successful API response could not be parsed into the classification schema.', f'{type(exc).__name__}: {exc}'), usage=response_usage(body)))
    parsed = pd.DataFrame(rows)
    if len(parsed) != len(manifest):
        raise RuntimeError(f'Integrity failure: parsed {len(parsed)} rows for {len(manifest)} manifest entries.')
    if parsed['pmid'].duplicated().any():
        raise RuntimeError('Integrity failure: parsed results contain duplicate PMIDs.')
    if set(parsed['pmid']) != set(manifest['pmid']):
        raise RuntimeError('Integrity failure: parsed PMID set differs from the manifest.')
    attempts = pd.to_numeric(parsed['attempt_number'], errors='coerce').fillna(1).astype(int)
    retries_used = attempts - 1
    technical_failure = parsed['error'].ne('')
    retryable = parsed['retryable'].astype(str).str.casefold().eq('true')
    retry_pending = technical_failure & retryable & (retries_used < max_technical_retries)
    if not args.technical_retry_queue:
        retry_pending = pd.Series(False, index=parsed.index)
    parsed['technical_retry_pending'] = retry_pending.map(str)
    exhausted = technical_failure & ~retry_pending
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'Reconciled {len(parsed)} manifest entries to {args.output} ({len(output_map)} output, {len(error_map)} error, {len(manifest) - len(output_map) - len(error_map)} missing).')
    if args.technical_retry_queue:
        retry_metadata = parsed.loc[retry_pending, ['pmid', 'error', 'error_code', 'attempt_number', 'custom_id']].rename(columns={'error': 'retry_source_error', 'error_code': 'retry_source_error_code', 'attempt_number': 'retry_source_attempt_number', 'custom_id': 'retry_source_custom_id'})
        retry_queue = source.merge(retry_metadata, on='pmid', how='inner', validate='one_to_one')
        if len(retry_queue) != int(retry_pending.sum()):
            raise RuntimeError('Integrity failure while constructing technical retry queue.')
        retry_path = Path(args.technical_retry_queue)
        retry_path.parent.mkdir(parents=True, exist_ok=True)
        retry_queue.to_csv(retry_path, index=False, encoding='utf-8-sig')
        print(f'Technical retry queue: {len(retry_queue)} records saved to {retry_path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
