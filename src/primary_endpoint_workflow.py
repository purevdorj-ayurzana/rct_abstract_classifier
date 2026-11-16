from common import relative_location
import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from common import atomic_write_json, build_responses_request_body, canonical_json_hash, load_prompt, normalize_pmid, responses_endpoint, sha256_file, utc_now_iso
from primary_endpoint_common import REQUIRED_PUBLICATION_COLUMNS, build_endpoint_user_input, configured_endpoint_arms, endpoint_arm_settings, endpoint_json_schema, endpoint_request_hash, load_endpoint_config, make_endpoint_input_hash, safe_endpoint_custom_id
PUBLICATION_COLUMNS = ('source_row_number', 'pmid', 'title', 'abstract', 'doi', 'journal', 'publication_year', 'publication_date', 'publication_types')

def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not-installed'

def write_csv_exclusive(table: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing file: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_csv(temporary, index=False, encoding='utf-8-sig')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def validate_endpoint_source(table: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PUBLICATION_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f'Endpoint source is missing columns: {sorted(missing)}')
    validated = table.copy().fillna('')
    validated['pmid'] = validated['pmid'].map(normalize_pmid)
    if validated['pmid'].eq('').any():
        raise ValueError('Endpoint source contains empty PMIDs.')
    if validated['pmid'].duplicated().any():
        raise ValueError('Endpoint source contains duplicate PMIDs.')
    return validated

def _count_dict(table: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in table[column].value_counts().sort_index().to_dict().items()}

def select_upstream_positive(rct_run_dir: Path, upstream_arm: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = rct_run_dir / 'run_manifest.json'
    source_path = rct_run_dir / 'source_snapshot.csv'
    final_path = rct_run_dir / upstream_arm / 'final' / 'all_classifications.csv'
    qa_path = rct_run_dir / upstream_arm / 'qa' / 'run_summary.json'
    required_paths = (manifest_path, source_path, final_path, qa_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'The upstream RCT run is not finalized for {upstream_arm}; missing: ' + ', '.join(missing))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if upstream_arm not in (manifest.get('active_arms') or []):
        raise ValueError(f'The upstream RCT run does not contain a {upstream_arm} arm.')
    qa = json.loads(qa_path.read_text(encoding='utf-8'))
    if qa.get('integrity_passed') is not True:
        raise ValueError(f'The upstream {upstream_arm} QA report did not pass integrity checks.')
    if int(qa.get('unresolved_technical_failures', 0)) != 0:
        raise ValueError(f'The upstream {upstream_arm} run has unresolved technical failures and cannot serve as a complete endpoint source.')
    parent_source = validate_endpoint_source(pd.read_csv(source_path, dtype=str).fillna(''))
    upstream = pd.read_csv(final_path, dtype=str).fillna('')
    required = REQUIRED_PUBLICATION_COLUMNS | {'final_label', 'publication_role', 'error'}
    missing_columns = required - set(upstream.columns)
    if missing_columns:
        raise ValueError(f'Upstream {upstream_arm} final output is missing columns: {sorted(missing_columns)}')
    upstream['pmid'] = upstream['pmid'].map(normalize_pmid)
    if upstream['pmid'].eq('').any() or upstream['pmid'].duplicated().any():
        raise ValueError(f'Upstream {upstream_arm} final output has invalid PMID identity.')
    if len(upstream) != len(parent_source) or set(upstream['pmid']) != set(parent_source['pmid']):
        raise ValueError(f'Upstream {upstream_arm} final output does not exactly cover the parent source.')
    if int(qa.get('final_rows', -1)) != len(upstream):
        raise ValueError(f'Upstream {upstream_arm} QA final_rows does not match its final CSV.')
    if int(qa.get('unique_pmids', -1)) != upstream['pmid'].nunique():
        raise ValueError(f'Upstream {upstream_arm} QA unique_pmids does not match its final CSV.')
    if qa.get('label_counts', {}) != _count_dict(upstream, 'final_label'):
        raise ValueError(f'Upstream {upstream_arm} QA label counts do not match its final CSV.')
    if upstream.loc[upstream['final_label'].eq('PRIMARY_RCT_REPORT'), 'error'].astype(str).str.strip().ne('').any():
        raise ValueError(f'A {upstream_arm}-positive upstream row contains a technical error.')
    selected = upstream[upstream['final_label'].eq('PRIMARY_RCT_REPORT')].copy()
    if selected.empty:
        raise ValueError(f'The upstream {upstream_arm} run contains no PRIMARY_RCT_REPORT rows.')
    renamed: dict[str, str] = {}
    for column in selected.columns:
        if column not in PUBLICATION_COLUMNS:
            renamed[column] = f'upstream_rct_{column}'
    selected = selected.rename(columns=renamed)
    parent_run_id = str(manifest.get('run_id', rct_run_dir.name))
    if 'upstream_rct_run_id' in selected.columns:
        recorded_run_ids = set(selected['upstream_rct_run_id'].astype(str).str.strip())
        if recorded_run_ids != {parent_run_id}:
            raise ValueError(f'Upstream {upstream_arm} rows do not consistently record the parent run ID.')
    else:
        selected.insert(min(2, len(selected.columns)), 'upstream_rct_run_id', parent_run_id)
    if 'upstream_rct_arm' in selected.columns:
        recorded_arms = set(selected['upstream_rct_arm'].astype(str).str.strip())
        if recorded_arms != {upstream_arm}:
            raise ValueError(f'Upstream {upstream_arm} rows do not consistently record the {upstream_arm} arm.')
    else:
        selected.insert(min(3, len(selected.columns)), 'upstream_rct_arm', upstream_arm)
    selected = validate_endpoint_source(selected)
    if 'source_row_number' not in selected.columns:
        selected.insert(0, 'source_row_number', range(1, len(selected) + 1))
    else:
        selected['source_row_number'] = pd.to_numeric(selected['source_row_number'], errors='raise').astype(int)
        selected = selected.sort_values('source_row_number', kind='stable').reset_index(drop=True)
    lineage = {'upstream_rct_run_id': str(manifest.get('run_id', rct_run_dir.name)), 'upstream_rct_run_dir': relative_location(rct_run_dir.resolve()), 'upstream_arm': upstream_arm, 'selection_rule': 'final_label == PRIMARY_RCT_REPORT', 'upstream_run_manifest_path': relative_location(manifest_path), 'upstream_run_manifest_sha256': sha256_file(manifest_path), 'upstream_source_path': relative_location(source_path), 'upstream_source_sha256': sha256_file(source_path), 'upstream_final_path': relative_location(final_path), 'upstream_final_sha256': sha256_file(final_path), 'upstream_qa_path': relative_location(qa_path), 'upstream_qa_sha256': sha256_file(qa_path), 'upstream_rows': len(upstream), 'selected_rows': len(selected), 'selected_pmid_hash': canonical_json_hash(sorted(selected['pmid'].tolist()))}
    return (selected, lineage)

def write_endpoint_batch_artifacts(table: pd.DataFrame, *, config: dict[str, Any], arm: str, prompt_path: Path, output_path: Path, manifest_path: Path, run_id: str, chunk_number: int, attempt_number: int) -> pd.DataFrame:
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError('Refusing to overwrite endpoint Batch input or manifest.')
    source = validate_endpoint_source(table)
    model, reasoning_effort, _ = endpoint_arm_settings(config, arm)
    prompt, prompt_hash = load_prompt(prompt_path)
    schema = endpoint_json_schema()
    schema_hash = canonical_json_hash(schema)
    built_at = utc_now_iso()
    requests: list[str] = []
    rows: list[dict[str, Any]] = []
    for source_row in source.to_dict(orient='records'):
        custom_id = safe_endpoint_custom_id(source_row['pmid'], arm=arm, chunk_number=chunk_number, attempt_number=attempt_number)
        body = build_responses_request_body(config, model=model, reasoning_effort=reasoning_effort, input_items=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': build_endpoint_user_input(source_row)}], text_format=schema)
        requests.append(json.dumps({'custom_id': custom_id, 'method': 'POST', 'url': responses_endpoint(config), 'body': body}, ensure_ascii=False))
        rows.append({'run_id': run_id, 'task': 'prespecified_primary_endpoint', 'arm': arm, 'stage': arm, 'chunk_number': chunk_number, 'attempt_number': attempt_number, 'custom_id': custom_id, 'pmid': source_row['pmid'], 'source_row_number': source_row.get('source_row_number', ''), 'requested_model': model, 'reasoning_effort': reasoning_effort, 'prompt_path': 'prompts/primary_endpoint.txt', 'prompt_hash': prompt_hash, 'schema_hash': schema_hash, 'input_hash': make_endpoint_input_hash(source_row), 'request_hash': endpoint_request_hash(body), 'built_at_utc': built_at})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(requests) + '\n', encoding='utf-8')
    manifest = pd.DataFrame(rows)
    write_csv_exclusive(manifest, manifest_path)
    return manifest

def verify_endpoint_run_integrity(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / 'run_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Endpoint run manifest is missing: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    config = load_endpoint_config(run_dir / 'config_snapshot.json')
    if manifest.get('workflow_mode') != config.get('workflow_mode'):
        raise ValueError('Endpoint run manifest and configuration disagree on workflow_mode.')
    if manifest.get('active_arms') != configured_endpoint_arms(config):
        raise ValueError('Endpoint run manifest and configuration disagree on active arms.')
    local_paths = {'config_snapshot_sha256': run_dir / 'config_snapshot.json', 'prompt_snapshot_sha256': run_dir / 'prompts' / 'primary_endpoint.txt', 'schema_snapshot_sha256': run_dir / 'schema_snapshot.json', 'source_snapshot_sha256': run_dir / 'source_snapshot.csv', 'parent_lineage_sha256': run_dir / 'parent_rct_lineage.json'}
    for field, path in local_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Endpoint run snapshot is missing: {path}')
        if sha256_file(path) != manifest.get(field):
            raise ValueError(f'Endpoint run snapshot hash mismatch: {path}')
    lineage = json.loads((run_dir / 'parent_rct_lineage.json').read_text(encoding='utf-8'))
    parent_paths: dict[str, Path] = {'upstream_run_manifest_sha256': Path(lineage['upstream_run_manifest_path']), 'upstream_source_sha256': Path(lineage['upstream_source_path'])}
    parent_paths.update({'upstream_final_sha256': Path(lineage['upstream_final_path']), 'upstream_qa_sha256': Path(lineage['upstream_qa_path'])})
    for field, path in parent_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Recorded upstream RCT artifact is missing: {path}')
        if sha256_file(path) != lineage.get(field):
            raise ValueError(f'Recorded upstream RCT artifact changed: {path}')
    source = validate_endpoint_source(pd.read_csv(run_dir / 'source_snapshot.csv', dtype=str).fillna(''))
    if len(source) != int(manifest.get('source_row_count', -1)):
        raise ValueError('Endpoint source row count differs from the run manifest.')
    if canonical_json_hash(sorted(source['pmid'].tolist())) != manifest.get('source_pmid_hash'):
        raise ValueError('Endpoint source PMID hash differs from the run manifest.')
    return manifest

def build_endpoint_chunks(*, run_dir: Path, arm: str, attempt_number: int, requested: pd.DataFrame, verify_integrity: bool=True) -> list[Path]:
    manifest = verify_endpoint_run_integrity(run_dir) if verify_integrity else json.loads((run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    config = load_endpoint_config(run_dir / 'config_snapshot.json')
    if arm not in configured_endpoint_arms(config):
        raise ValueError(f'Unknown endpoint arm: {arm}')
    if attempt_number <= 0:
        raise ValueError('Attempt number must be positive.')
    source = validate_endpoint_source(pd.read_csv(run_dir / 'source_snapshot.csv', dtype=str).fillna(''))
    requested = validate_endpoint_source(requested)
    requested_pmids = requested['pmid'].tolist()
    if len(requested_pmids) != len(set(requested_pmids)):
        raise ValueError('Requested endpoint input contains duplicate PMIDs.')
    source_index = source.set_index('pmid', drop=False)
    unknown = set(requested_pmids) - set(source_index.index)
    if unknown:
        raise ValueError(f'Requested endpoint PMIDs are outside the run source: {sorted(unknown)[:5]}')
    selected = source_index.loc[requested_pmids].reset_index(drop=True)
    selected = selected.sort_values('pmid', kind='stable').reset_index(drop=True)
    _, _, chunk_size = endpoint_arm_settings(config, arm)
    built: list[Path] = []
    for start in range(0, len(selected), chunk_size):
        chunk_number = start // chunk_size + 1
        attempt_dir = run_dir / arm / f'chunk_{chunk_number:04d}' / f'attempt_{attempt_number:02d}'
        if attempt_dir.exists():
            raise FileExistsError(f'Refusing to overwrite existing endpoint attempt: {attempt_dir}')
        attempt_dir.mkdir(parents=True)
        chunk = selected.iloc[start:start + chunk_size].copy()
        write_csv_exclusive(chunk, attempt_dir / 'source.csv')
        write_endpoint_batch_artifacts(chunk, config=config, arm=arm, prompt_path=run_dir / 'prompts' / 'primary_endpoint.txt', output_path=attempt_dir / 'input.jsonl', manifest_path=attempt_dir / 'manifest.csv', run_id=str(manifest['run_id']), chunk_number=chunk_number, attempt_number=attempt_number)
        built.append(attempt_dir)
    if len(selected) != len(requested_pmids) or set(selected['pmid']) != set(requested_pmids):
        raise RuntimeError('Endpoint chunks do not exactly cover requested PMIDs.')
    return built

def prepare_endpoint_run(*, config_path: Path, rct_run_dir: Path, output_root: Path, run_id: str='') -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f'Endpoint config not found: {config_path}')
    config = load_endpoint_config(config_path)
    upstream_arm = str(config['source_selector']['upstream_arm'])
    selected, lineage = select_upstream_positive(rct_run_dir, upstream_arm)
    prompt_path = Path(config['classification_prompt'])
    prompt, prompt_hash = load_prompt(prompt_path)
    schema = endpoint_json_schema()
    schema_hash = canonical_json_hash(schema)
    fingerprint = canonical_json_hash({'config': config, 'upstream_final_sha256': lineage['upstream_final_sha256'], 'selected_pmid_hash': lineage['selected_pmid_hash'], 'prompt_hash': prompt_hash, 'schema_hash': schema_hash})
    if not run_id:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_id = f'{timestamp}_{fingerprint[:8]}'
    if not all((character.isalnum() or character in '-_' for character in run_id)):
        raise ValueError('Endpoint run ID may contain only letters, numbers, hyphens, and underscores.')
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    temporary_dir = output_root / f'.{run_id}.preparing'
    if final_dir.exists() or temporary_dir.exists():
        raise FileExistsError(f'Endpoint run already exists or is being prepared: {run_id}')
    temporary_dir.mkdir()
    try:
        relatives = ['prompts', 'aggregate', 'qa']
        for arm in configured_endpoint_arms(config):
            relatives.extend((f'{arm}/aggregate', f'{arm}/final', f'{arm}/qa'))
        for relative in relatives:
            (temporary_dir / relative).mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, temporary_dir / 'config_snapshot.json')
        (temporary_dir / 'prompts' / 'primary_endpoint.txt').write_text(prompt + '\n', encoding='utf-8')
        atomic_write_json(temporary_dir / 'schema_snapshot.json', schema)
        atomic_write_json(temporary_dir / 'parent_rct_lineage.json', lineage)
        write_csv_exclusive(selected, temporary_dir / 'source_snapshot.csv')
        run_manifest = {'run_id': run_id, 'created_at_utc': utc_now_iso(), 'status': 'prepared', 'workflow_mode': config['workflow_mode'], 'active_arms': configured_endpoint_arms(config), 'selection_rule': lineage['selection_rule'], 'upstream_rct_run_id': lineage['upstream_rct_run_id'], 'upstream_rct_run_dir': lineage['upstream_rct_run_dir'], 'upstream_arm': upstream_arm, 'source_row_count': len(selected), 'source_pmid_hash': canonical_json_hash(sorted(selected['pmid'].tolist())), 'run_fingerprint': fingerprint, 'prompt_hash': prompt_hash, 'schema_hash': schema_hash, 'config_snapshot_sha256': sha256_file(temporary_dir / 'config_snapshot.json'), 'prompt_snapshot_sha256': sha256_file(temporary_dir / 'prompts' / 'primary_endpoint.txt'), 'schema_snapshot_sha256': sha256_file(temporary_dir / 'schema_snapshot.json'), 'source_snapshot_sha256': sha256_file(temporary_dir / 'source_snapshot.csv'), 'parent_lineage_sha256': sha256_file(temporary_dir / 'parent_rct_lineage.json'), 'parent_artifact_hashes': {key: lineage[key] for key in ('upstream_run_manifest_sha256', 'upstream_source_sha256', 'upstream_final_sha256', 'upstream_qa_sha256')}, 'arm_settings': {arm: {'model': endpoint_arm_settings(config, arm)[0], 'reasoning_effort': endpoint_arm_settings(config, arm)[1], 'chunk_size': endpoint_arm_settings(config, arm)[2]} for arm in configured_endpoint_arms(config)}, 'max_technical_retries': int(config['max_technical_retries']), 'python_version': sys.version, 'package_versions': {'openai': package_version('openai'), 'pandas': package_version('pandas'), 'pydantic': package_version('pydantic')}}
        atomic_write_json(temporary_dir / 'run_manifest.json', run_manifest)
        for arm in configured_endpoint_arms(config):
            build_endpoint_chunks(run_dir=temporary_dir, arm=arm, attempt_number=1, requested=selected, verify_integrity=False)
        os.replace(temporary_dir, final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return final_dir

def collect_endpoint_retries(*, run_dir: Path, arm: str, attempt_number: int) -> tuple[pd.DataFrame, Path]:
    verify_endpoint_run_integrity(run_dir)
    config = load_endpoint_config(run_dir / 'config_snapshot.json')
    if arm not in configured_endpoint_arms(config):
        raise ValueError(f'Unknown endpoint arm: {arm}')
    paths = sorted(run_dir.glob(f'{arm}/chunk_*/attempt_{attempt_number:02d}/results.csv'))
    if not paths:
        raise ValueError(f'No parsed endpoint results found for {arm} attempt {attempt_number}.')
    results = pd.concat([pd.read_csv(path, dtype=str).fillna('') for path in paths], ignore_index=True)
    if results['pmid'].duplicated().any():
        raise ValueError('Endpoint retry collection found duplicate PMIDs.')
    if 'technical_retry_pending' not in results.columns:
        raise ValueError('Endpoint results are missing technical_retry_pending.')
    pending = results['technical_retry_pending'].str.casefold().eq('true')
    metadata = results.loc[pending, ['pmid', 'error', 'error_code', 'custom_id', 'attempt_number']].rename(columns={'error': 'retry_source_error', 'error_code': 'retry_source_error_code', 'custom_id': 'retry_source_custom_id', 'attempt_number': 'retry_source_attempt_number'})
    source = validate_endpoint_source(pd.read_csv(run_dir / 'source_snapshot.csv', dtype=str).fillna(''))
    queue = source.merge(metadata, on='pmid', how='inner', validate='one_to_one')
    if len(queue) != int(pending.sum()):
        raise RuntimeError('Endpoint retry queue does not match pending results.')
    output = run_dir / 'aggregate' / f'{arm}_technical_retry_attempt_{attempt_number + 1:02d}.csv'
    write_csv_exclusive(queue, output)
    return (queue, output)

def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare immutable downstream primary-endpoint Batch runs.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare = subparsers.add_parser('prepare', help='Select positive RCT reports from the configured upstream arm and prepare complete endpoint chunks.')
    prepare.add_argument('--config', default='primary_endpoint_config.json')
    prepare.add_argument('--rct-run-dir', required=True)
    prepare.add_argument('--output-root', default=None)
    prepare.add_argument('--run-id', default='')
    build = subparsers.add_parser('build-stage', help='Build a write-once same-model retry attempt.')
    build.add_argument('--run-dir', required=True)
    build.add_argument('--arm', required=True)
    build.add_argument('--input', required=True)
    build.add_argument('--attempt-number', type=int, required=True)
    collect = subparsers.add_parser('collect-retries', help='Collect pending same-model endpoint retries.')
    collect.add_argument('--run-dir', required=True)
    collect.add_argument('--arm', required=True)
    collect.add_argument('--attempt-number', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        config = load_endpoint_config(args.config)
        run_dir = prepare_endpoint_run(config_path=Path(args.config), rct_run_dir=Path(args.rct_run_dir), output_root=Path(args.output_root or config['output_root']), run_id=args.run_id)
        print(f'Prepared primary-endpoint Batch run: {run_dir}')
        return 0
    run_dir = Path(args.run_dir)
    if args.command == 'collect-retries':
        queue, output = collect_endpoint_retries(run_dir=run_dir, arm=args.arm, attempt_number=args.attempt_number)
        print(f'Collected {len(queue)} endpoint retries in {output}.')
        return 0
    requested = pd.read_csv(args.input, dtype=str).fillna('')
    built = build_endpoint_chunks(run_dir=run_dir, arm=args.arm, attempt_number=args.attempt_number, requested=requested)
    print(f'Built {len(built)} endpoint {args.arm} chunk(s) for attempt {args.attempt_number}.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
