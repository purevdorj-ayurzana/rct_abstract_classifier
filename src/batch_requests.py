from __future__ import annotations
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
import pandas as pd
from common import build_responses_request_body, build_user_input, canonical_json_hash, json_schema_for_batch, load_prompt, make_input_hash, normalize_pmid, responses_endpoint, stage_settings, utc_now_iso

def safe_custom_id(pmid, *, stage, chunk_number, attempt_number):
    if stage != 'sol':
        raise ValueError('Use stage sol.')
    safe_pmid = re.sub('[^A-Za-z0-9_-]', '-', pmid)
    return f'sol-c{chunk_number:04d}-a{attempt_number:02d}-pmid-{safe_pmid}'

def validate_source(df: pd.DataFrame) -> pd.DataFrame:
    missing = {'pmid', 'title', 'abstract'} - set(df.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    validated = df.copy().fillna('')
    validated['pmid'] = validated['pmid'].map(normalize_pmid)
    if validated['pmid'].eq('').any():
        raise ValueError('Input contains empty PMIDs.')
    if validated['pmid'].duplicated().any():
        raise ValueError('Input contains duplicate PMIDs.')
    return validated

def _atomic_write_csv(table: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_csv(temporary, index=False, encoding='utf-8-sig')
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

def write_batch_artifacts(df: pd.DataFrame, *, config: dict[str, Any], stage: str, output_path: Path, manifest_path: Path, run_id: str='', chunk_number: int=0, attempt_number: int=1, prompt_override: str | Path | None=None, prompt_record_path: str | None=None) -> pd.DataFrame:
    """Build one write-once request JSONL and its authoritative manifest."""
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite an existing Batch input or manifest: {output_path}, {manifest_path}')
    source = validate_source(df)
    model, configured_prompt, reasoning_effort = stage_settings(config, stage)
    prompt_path = str(prompt_override or configured_prompt)
    prompt, prompt_hash = load_prompt(prompt_path)
    schema = json_schema_for_batch()
    schema_hash = canonical_json_hash(schema)
    built_at = utc_now_iso()
    request_lines: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    for row in source.to_dict(orient='records'):
        pmid = row['pmid']
        custom_id = safe_custom_id(pmid, stage=stage, chunk_number=chunk_number, attempt_number=attempt_number)
        body = build_responses_request_body(config, model=model, reasoning_effort=reasoning_effort, input_items=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': build_user_input(row)}], text_format=schema)
        request = {'custom_id': custom_id, 'method': 'POST', 'url': responses_endpoint(config), 'body': body}
        request_lines.append(json.dumps(request, ensure_ascii=False))
        manifest_rows.append({'run_id': run_id, 'stage': stage, 'chunk_number': chunk_number, 'attempt_number': attempt_number, 'custom_id': custom_id, 'pmid': pmid, 'source_row_number': row.get('source_row_number', ''), 'requested_model': model, 'reasoning_effort': reasoning_effort, 'prompt_path': prompt_record_path or prompt_path, 'prompt_hash': prompt_hash, 'schema_hash': schema_hash, 'input_hash': make_input_hash(row), 'request_hash': canonical_json_hash(body), 'built_at_utc': built_at})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(request_lines) + '\n', encoding='utf-8')
    manifest = pd.DataFrame(manifest_rows)
    _atomic_write_csv(manifest, manifest_path)
    return manifest
