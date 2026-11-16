from common import relative_location
import argparse
import importlib.metadata
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from batch_requests import validate_source, write_batch_artifacts
from common import atomic_write_json, canonical_json_hash, configured_model_arms, json_schema_for_batch, load_json, load_prompt, sha256_file, utc_now_iso

def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not-installed'

def write_csv_exclusive(table: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing file: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding='utf-8-sig')

def deterministic_source(path: Path) -> pd.DataFrame:
    source = validate_source(pd.read_csv(path, dtype=str).fillna(''))
    source.insert(0, 'source_row_number', range(1, len(source) + 1))
    return source

def configured_stages(config):
    return configured_model_arms(config)

def validate_stage(config: dict[str, Any], stage: str) -> None:
    if stage not in configured_stages(config):
        raise ValueError(f'Unknown stage/model arm {stage!r}; expected one of {configured_stages(config)}.')

def stage_chunk_size(config, stage):
    validate_stage(config, stage)
    size = int(config['model_arms'][stage]['chunk_size'])
    if size <= 0:
        raise ValueError('chunk_size must be positive.')
    return size

def snapshot_prompt_path(run_dir, stage):
    validate_stage(load_json(run_dir / 'config_snapshot.json'), stage)
    return run_dir / 'prompts/classification.txt'

def prompt_record_path(config, stage):
    validate_stage(config, stage)
    return 'prompts/classification.txt'

def build_stage_chunks(*, run_dir: Path, run_id: str, stage: str, attempt_number: int, requested: pd.DataFrame) -> list[Path]:
    config = load_json(run_dir / 'config_snapshot.json')
    validate_stage(config, stage)
    if attempt_number <= 0:
        raise ValueError('Attempt number must be positive.')
    source = pd.read_csv(run_dir / 'source_snapshot.csv', dtype=str).fillna('')
    requested = validate_source(requested)
    request_pmids = requested['pmid'].tolist()
    if len(request_pmids) != len(set(request_pmids)):
        raise ValueError('Requested stage input contains duplicate PMIDs.')
    source_index = source.set_index('pmid', drop=False)
    unknown = set(request_pmids) - set(source_index.index)
    if unknown:
        raise ValueError(f'Stage input contains PMIDs outside the run source: {sorted(unknown)[:5]}')
    request_columns = [column for column in requested.columns if column not in source.columns and column != 'pmid']
    requested_metadata = requested[['pmid', *request_columns]].copy()
    selected = source_index.loc[request_pmids].reset_index(drop=True)
    if request_columns:
        selected = selected.merge(requested_metadata, on='pmid', how='left', validate='one_to_one')
    selected = selected.sort_values('pmid', kind='stable').reset_index(drop=True)
    chunk_size = stage_chunk_size(config, stage)
    built: list[Path] = []
    for start in range(0, len(selected), chunk_size):
        chunk_number = start // chunk_size + 1
        attempt_dir = run_dir / stage / f'chunk_{chunk_number:04d}' / f'attempt_{attempt_number:02d}'
        if attempt_dir.exists():
            raise FileExistsError(f'Refusing to overwrite existing attempt: {attempt_dir}')
        attempt_dir.mkdir(parents=True)
        chunk = selected.iloc[start:start + chunk_size].copy()
        write_csv_exclusive(chunk, attempt_dir / 'source.csv')
        write_batch_artifacts(chunk, config=config, stage=stage, output_path=attempt_dir / 'input.jsonl', manifest_path=attempt_dir / 'manifest.csv', run_id=run_id, chunk_number=chunk_number, attempt_number=attempt_number, prompt_override=snapshot_prompt_path(run_dir, stage), prompt_record_path=prompt_record_path(config, stage))
        built.append(attempt_dir)
    if set(selected['pmid']) != set(request_pmids) or len(selected) != len(request_pmids):
        raise RuntimeError('Built stage chunks do not exactly cover the requested PMIDs.')
    return built

def collect_retry_queue(*, run_dir: Path, stage: str, attempt_number: int) -> tuple[pd.DataFrame, Path]:
    config = load_json(run_dir / 'config_snapshot.json')
    validate_stage(config, stage)
    paths = sorted(run_dir.glob(f'{stage}/chunk_*/attempt_{attempt_number:02d}/results.csv'))
    if not paths:
        raise ValueError(f'No parsed {stage} results found for attempt {attempt_number}.')
    frames = [pd.read_csv(path, dtype=str).fillna('') for path in paths]
    results = pd.concat(frames, ignore_index=True)
    if results['pmid'].duplicated().any():
        raise ValueError('Retry collection found duplicate PMIDs within the same attempt.')
    if 'technical_retry_pending' not in results.columns:
        raise ValueError('Parsed results are missing technical_retry_pending.')
    pending = results['technical_retry_pending'].str.casefold().eq('true')
    metadata_columns = [column for column in ('pmid', 'error', 'error_code', 'custom_id', 'attempt_number') if column in results.columns]
    metadata = results.loc[pending, metadata_columns].copy().rename(columns={'error': 'retry_source_error', 'error_code': 'retry_source_error_code', 'custom_id': 'retry_source_custom_id', 'attempt_number': 'retry_source_attempt_number'})
    source = pd.read_csv(run_dir / 'source_snapshot.csv', dtype=str).fillna('')
    queue = source.merge(metadata, on='pmid', how='inner', validate='one_to_one')
    if len(queue) != int(pending.sum()):
        raise RuntimeError('Collected retry queue does not match pending results.')
    next_attempt = attempt_number + 1
    output = run_dir / 'aggregate' / f'{stage}_technical_retry_attempt_{next_attempt:02d}.csv'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing retry queue: {output}')
    write_csv_exclusive(queue, output)
    return (queue, output)

def _run_identity(config, input_path, arms):
    available = configured_model_arms(config)
    if arms is not None and arms != available:
        raise ValueError('Use exactly the sol model.')
    prompt, prompt_hash = load_prompt(config['classification_prompt'])
    schema = json_schema_for_batch()
    identity = {'workflow_mode': 'single_model_arm', 'active_arms': available, 'classification_prompt_hash': prompt_hash}
    identity['run_fingerprint'] = canonical_json_hash({'config': config, 'source_file_hash': sha256_file(input_path), 'identity': identity, 'schema_hash': canonical_json_hash(schema)})
    return (identity, {'classification': prompt}, schema)

def prepare_run(config_path: Path, input_path: Path, output_root: Path, run_id: str, arms: list[str] | None=None) -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    if not input_path.is_file():
        raise FileNotFoundError(f'Input not found: {input_path}')
    config = load_json(config_path)
    source = deterministic_source(input_path)
    identity, prompts, schema = _run_identity(config, input_path, arms)
    fingerprint = identity['run_fingerprint']
    if not run_id:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_id = f'{timestamp}_{fingerprint[:8]}'
    if not all((ch.isalnum() or ch in '-_' for ch in run_id)):
        raise ValueError('Run ID may contain only letters, numbers, hyphens, and underscores.')
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    temporary_dir = output_root / f'.{run_id}.preparing'
    if final_dir.exists() or temporary_dir.exists():
        raise FileExistsError(f'Run already exists or is being prepared: {run_id}')
    temporary_dir.mkdir()
    try:
        (temporary_dir / 'prompts').mkdir()
        (temporary_dir / 'aggregate').mkdir()
        (temporary_dir / 'final').mkdir()
        (temporary_dir / 'qa').mkdir()
        shutil.copy2(config_path, temporary_dir / 'config_snapshot.json')
        for name, prompt in prompts.items():
            filename = 'classification.txt' if name == 'classification' else f'{name}.txt'
            (temporary_dir / 'prompts' / filename).write_text(prompt + '\n', encoding='utf-8')
        atomic_write_json(temporary_dir / 'schema_snapshot.json', schema)
        write_csv_exclusive(source, temporary_dir / 'source_snapshot.csv')
        manifest: dict[str, Any] = {'run_id': run_id, 'created_at_utc': utc_now_iso(), 'status': 'prepared', 'workflow_mode': identity['workflow_mode'], 'active_arms': identity['active_arms'], 'source_original_path': relative_location(input_path), 'source_original_sha256': sha256_file(input_path), 'source_snapshot_sha256': sha256_file(temporary_dir / 'source_snapshot.csv'), 'source_row_count': len(source), 'source_pmid_hash': canonical_json_hash(sorted(source['pmid'].tolist())), 'config_sha256': sha256_file(temporary_dir / 'config_snapshot.json'), 'schema_hash': canonical_json_hash(schema), 'run_fingerprint': fingerprint, 'python_version': sys.version, 'package_versions': {'openai': package_version('openai'), 'pandas': package_version('pandas'), 'pydantic': package_version('pydantic')}, 'max_technical_retries': int(config['max_technical_retries'])}
        manifest['classification_prompt_hash'] = identity['classification_prompt_hash']
        manifest['arm_settings'] = {arm: {'model': config['model_arms'][arm]['model'], 'reasoning_effort': config['model_arms'][arm]['reasoning_effort'], 'chunk_size': stage_chunk_size(config, arm)} for arm in identity['active_arms']}
        atomic_write_json(temporary_dir / 'run_manifest.json', manifest)
        for stage in identity['active_arms']:
            build_stage_chunks(run_dir=temporary_dir, run_id=run_id, stage=stage, attempt_number=1, requested=source)
        os.replace(temporary_dir, final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return final_dir

def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare immutable, chunked Batch runs.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare = subparsers.add_parser('prepare', help='Prepare a complete RCT classification run.')
    prepare.add_argument('--config', default='config.json')
    prepare.add_argument('--input', default=None)
    prepare.add_argument('--output-root', default=None)
    prepare.add_argument('--run-id', default='')
    build = subparsers.add_parser('build-stage', help='Build write-once same-model retry chunks in an existing run.')
    build.add_argument('--run-dir', required=True)
    build.add_argument('--stage', required=True, help='Use sol.')
    build.add_argument('--input', required=True)
    build.add_argument('--attempt-number', type=int, required=True)
    collect = subparsers.add_parser('collect-retries', help='Collect pending per-chunk same-model technical retries.')
    collect.add_argument('--run-dir', required=True)
    collect.add_argument('--stage', required=True, help='Use sol.')
    collect.add_argument('--attempt-number', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        config = load_json(args.config)
        input_path = Path(args.input or config['input_csv'])
        default_root = config.get('output_root', config.get('comparison_output_root', 'outputs/batch/runs'))
        run_dir = prepare_run(Path(args.config), input_path, Path(args.output_root or default_root), args.run_id, None)
        print(f'Prepared Batch run: {run_dir}')
        return 0
    run_dir = Path(args.run_dir)
    run_manifest = load_json(run_dir / 'run_manifest.json')
    if args.command == 'collect-retries':
        queue, output = collect_retry_queue(run_dir=run_dir, stage=args.stage, attempt_number=args.attempt_number)
        print(f'Collected {len(queue)} technical retries in {output}.')
        return 0
    requested = pd.read_csv(args.input, dtype=str).fillna('')
    built = build_stage_chunks(run_dir=run_dir, run_id=run_manifest['run_id'], stage=args.stage, attempt_number=args.attempt_number, requested=requested)
    print(f'Built {len(built)} {args.stage} chunk(s) for attempt {args.attempt_number}.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
