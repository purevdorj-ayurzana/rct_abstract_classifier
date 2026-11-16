from __future__ import annotations
import argparse
import os
import tempfile
from pathlib import Path
from typing import Any
import pandas as pd
from common import atomic_write_json, utc_now_iso
from primary_endpoint_common import VALID_FRAMEWORKS, VALID_RESULTS, VALID_UNCLEAR_REASONS
from primary_endpoint_schemas import PrimaryEndpointAssessment
from primary_endpoint_workflow import validate_endpoint_source, verify_endpoint_run_integrity
LABEL_ORDER = ['POSITIVE', 'NEGATIVE', 'UNCLEAR']
FINAL_FILENAMES = ('all_primary_endpoint_assessments.csv', 'positive_primary_endpoint_reports.csv', 'negative_primary_endpoint_reports.csv', 'unclear_primary_endpoint_reports.csv')
TOKEN_COLUMNS = ('input_tokens', 'cache_write_tokens', 'cached_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens')

def atomic_write_csv(table: pd.DataFrame, output: Path) -> None:
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

def validate_endpoint_results(table: pd.DataFrame, *, name: str) -> None:
    required = {'pmid', 'primary_endpoint_result', 'primary_endpoint_description', 'trial_framework', 'unclear_reason', 'brief_rationale', 'error'}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f'{name} is missing columns: {sorted(missing)}')
    if table['pmid'].astype(str).str.strip().eq('').any():
        raise ValueError(f'{name} contains empty PMIDs.')
    if table['pmid'].duplicated().any():
        raise ValueError(f'{name} contains duplicate PMIDs.')
    invalid_results = set(table['primary_endpoint_result']) - VALID_RESULTS
    invalid_frameworks = set(table['trial_framework']) - VALID_FRAMEWORKS
    invalid_reasons = set(table['unclear_reason']) - VALID_UNCLEAR_REASONS
    if invalid_results:
        raise ValueError(f'{name} has invalid endpoint results: {invalid_results}')
    if invalid_frameworks:
        raise ValueError(f'{name} has invalid trial frameworks: {invalid_frameworks}')
    if invalid_reasons:
        raise ValueError(f'{name} has invalid unclear reasons: {invalid_reasons}')
    for row in table[table['error'].eq('')].to_dict(orient='records'):
        PrimaryEndpointAssessment.model_validate({'primary_endpoint_result': row['primary_endpoint_result'], 'primary_endpoint_description': row['primary_endpoint_description'], 'trial_framework': row['trial_framework'], 'unclear_reason': row['unclear_reason'], 'brief_rationale': row['brief_rationale']})

def read_endpoint_attempts(run_dir: Path, arm: str) -> pd.DataFrame:
    paths = sorted(run_dir.glob(f'{arm}/chunk_*/attempt_*/results.csv'))
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str).fillna('')
        if frame.empty:
            continue
        frame['result_file'] = str(path.relative_to(run_dir))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def resolve_endpoint_results(run_dir: Path, arm: str, expected_pmids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts = read_endpoint_attempts(run_dir, arm)
    if attempts.empty:
        raise ValueError(f'No parsed primary-endpoint results found for {arm}.')
    required = {'pmid', 'primary_endpoint_result', 'error', 'attempt_number'}
    missing = required - set(attempts.columns)
    if missing:
        raise ValueError(f'{arm} endpoint attempts are missing columns: {sorted(missing)}')
    attempts['attempt_number_numeric'] = pd.to_numeric(attempts['attempt_number'], errors='coerce').fillna(1).astype(int)
    unknown = set(attempts['pmid']) - expected_pmids
    if unknown:
        raise ValueError(f'{arm} endpoint results contain unexpected PMIDs: {sorted(unknown)[:5]}')
    selected_rows: list[dict[str, Any]] = []
    classification_columns = ['primary_endpoint_result', 'primary_endpoint_description', 'trial_framework', 'unclear_reason', 'brief_rationale']
    for pmid, group in attempts.groupby('pmid', sort=False):
        successes = group[group['error'].eq('')]
        if len(successes) > 1:
            comparable = successes[classification_columns].drop_duplicates()
            if len(comparable) > 1:
                raise ValueError(f'Conflicting successful {arm} endpoint assessments for PMID {pmid}.')
        candidates = successes if not successes.empty else group
        chosen = candidates.sort_values('attempt_number_numeric', kind='stable').iloc[-1]
        selected_rows.append(chosen.drop(labels=['attempt_number_numeric']).to_dict())
    selected = pd.DataFrame(selected_rows)
    missing_pmids = expected_pmids - set(selected['pmid'])
    if missing_pmids:
        raise ValueError(f'{arm} endpoint aggregation is incomplete; missing PMIDs: {sorted(missing_pmids)[:5]}')
    if selected['pmid'].duplicated().any() or len(selected) != len(expected_pmids):
        raise RuntimeError(f'{arm} endpoint aggregation did not produce one row per PMID.')
    if 'technical_retry_pending' in selected.columns:
        pending = selected['technical_retry_pending'].str.casefold().eq('true')
        if pending.any():
            raise ValueError(f'{arm} still has pending same-model endpoint retries: {selected.loc[pending, 'pmid'].tolist()[:5]}')
    validate_endpoint_results(selected, name=f'Aggregated {arm} endpoint results')
    return (selected, attempts)

def restore_source_order(table: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    order = {pmid: position for position, pmid in enumerate(source['pmid'].tolist(), start=1)}
    restored = table.copy()
    restored['_endpoint_source_order'] = restored['pmid'].map(order)
    if restored['_endpoint_source_order'].isna().any():
        raise RuntimeError('Endpoint source-order restoration found unknown PMIDs.')
    return restored.sort_values('_endpoint_source_order', kind='stable').drop(columns=['_endpoint_source_order']).reset_index(drop=True)

def usage_summary(attempts: pd.DataFrame) -> dict[str, int]:
    summary = {'attempt_rows': len(attempts)}
    for column in TOKEN_COLUMNS:
        summary[column] = int(pd.to_numeric(attempts.get(column, pd.Series(dtype=str)), errors='coerce').fillna(0).sum())
    return summary

def write_endpoint_exports(final: pd.DataFrame, final_dir: Path) -> None:
    groups = {'positive_primary_endpoint_reports.csv': final[final['primary_endpoint_result'].eq('POSITIVE')].copy(), 'negative_primary_endpoint_reports.csv': final[final['primary_endpoint_result'].eq('NEGATIVE')].copy(), 'unclear_primary_endpoint_reports.csv': final[final['primary_endpoint_result'].eq('UNCLEAR')].copy()}
    pmid_sets = [set(frame['pmid']) for frame in groups.values()]
    if any((pmid_sets[left] & pmid_sets[right] for left in range(3) for right in range(left + 1, 3))):
        raise RuntimeError('Primary-endpoint export groups overlap.')
    if set().union(*pmid_sets) != set(final['pmid']):
        raise RuntimeError('Primary-endpoint export groups do not cover all PMIDs.')
    atomic_write_csv(final, final_dir / 'all_primary_endpoint_assessments.csv')
    for filename, frame in groups.items():
        atomic_write_csv(frame, final_dir / filename)

def write_arm_qa(run_dir: Path, arm: str, final: pd.DataFrame, attempts: pd.DataFrame) -> dict[str, Any]:
    qa_dir = run_dir / arm / 'qa'
    label_counts = final.groupby('primary_endpoint_result', dropna=False).size().rename('count').reset_index()
    unclear_counts = final[final['primary_endpoint_result'].eq('UNCLEAR')].groupby('unclear_reason', dropna=False).size().rename('count').reset_index()
    framework_counts = final.groupby('trial_framework', dropna=False).size().rename('count').reset_index()
    technical_failures = final[final['error'].ne('')].copy()
    usage = usage_summary(attempts)
    token_table = pd.DataFrame([{'arm': arm, **usage}])
    for filename, frame in (('label_counts.csv', label_counts), ('unclear_reason_counts.csv', unclear_counts), ('trial_framework_counts.csv', framework_counts), ('technical_failures.csv', technical_failures), ('token_usage_all_attempts.csv', token_table)):
        atomic_write_csv(frame, qa_dir / filename)
    summary = {'generated_at_utc': utc_now_iso(), 'integrity_passed': True, 'arm': arm, 'final_rows': len(final), 'unique_pmids': int(final['pmid'].nunique()), 'label_counts': final['primary_endpoint_result'].value_counts().sort_index().to_dict(), 'unclear_reason_counts': final.loc[final['primary_endpoint_result'].eq('UNCLEAR'), 'unclear_reason'].value_counts().sort_index().to_dict(), 'unresolved_technical_failures': len(technical_failures), 'usage_all_attempts': usage}
    atomic_write_json(qa_dir / 'run_summary.json', summary)
    return summary

def finalize_arm(run_dir: Path, arm: str, source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected = set(source['pmid'])
    selected, attempts = resolve_endpoint_results(run_dir, arm, expected)
    selected = restore_source_order(selected, source)
    selected['endpoint_arm'] = arm
    selected['final_model'] = selected.apply(lambda row: row.get('actual_model', '') or row.get('model', ''), axis=1)
    if len(selected) != len(source) or set(selected['pmid']) != expected:
        raise RuntimeError(f'{arm} final endpoint output does not match its source.')
    validate_endpoint_results(selected, name=f'Final {arm} endpoint results')
    atomic_write_csv(selected, run_dir / arm / 'aggregate' / 'results.csv')
    write_endpoint_exports(selected, run_dir / arm / 'final')
    summary = write_arm_qa(run_dir, arm, selected, attempts)
    return (selected, summary)

def finalize_endpoint_run(run_dir: Path) -> dict[str, pd.DataFrame]:
    manifest = verify_endpoint_run_integrity(run_dir)
    arms = list(manifest.get('active_arms') or [])
    if not arms:
        raise ValueError('Endpoint finalization requires at least one active arm.')
    source = validate_endpoint_source(pd.read_csv(run_dir / 'source_snapshot.csv', dtype=str).fillna(''))
    results: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm in arms:
        results[arm], summaries[arm] = finalize_arm(run_dir, arm, source)
    atomic_write_json(run_dir / 'qa' / 'run_summary.json', {'generated_at_utc': utc_now_iso(), 'integrity_passed': True, 'source_rows': len(source), 'upstream_rct_run_id': manifest['upstream_rct_run_id'], 'selection_rule': manifest['selection_rule'], 'arms': {arm: {'final_rows': len(results[arm]), 'label_counts': results[arm]['primary_endpoint_result'].value_counts().sort_index().to_dict(), 'usage_all_attempts': summaries[arm]['usage_all_attempts']} for arm in arms}})
    return results

def main():
    parser = argparse.ArgumentParser(description='Finalize and check a primary-endpoint run.')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    results = finalize_endpoint_run(Path(args.run_dir))
    print(f'Finalized {len(results['sol'])} primary-endpoint assessments.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
