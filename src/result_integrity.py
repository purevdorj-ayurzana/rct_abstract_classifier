from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Any
import pandas as pd

FINAL_FILENAMES = ('all_classifications.csv', 'primary_rct_candidates.csv', 'unclear_publications.csv', 'excluded_publications.csv')

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

def read_result_attempts(run_dir: Path, stage: str) -> pd.DataFrame:
    paths = sorted(run_dir.glob(f'{stage}/chunk_*/attempt_*/results.csv'))
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str).fillna('')
        if frame.empty:
            continue
        frame['result_file'] = str(path.relative_to(run_dir))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def resolve_stage_results(run_dir: Path, stage: str, expected_pmids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts = read_result_attempts(run_dir, stage)
    if not expected_pmids and attempts.empty:
        return (pd.DataFrame(), attempts)
    if attempts.empty:
        raise ValueError(f'No parsed {stage} results were found.')
    required = {'pmid', 'final_label', 'error', 'attempt_number'}
    missing = required - set(attempts.columns)
    if missing:
        raise ValueError(f'{stage} result attempts are missing columns: {sorted(missing)}')
    attempts['attempt_number_numeric'] = pd.to_numeric(attempts['attempt_number'], errors='coerce').fillna(1).astype(int)
    unknown = set(attempts['pmid']) - expected_pmids
    if unknown:
        raise ValueError(f'{stage} results contain unexpected PMIDs: {sorted(unknown)[:5]}')
    selected_rows: list[dict[str, Any]] = []
    classification_columns = ['final_label', 'underlying_trial_randomized', 'publication_role', 'trial_name_or_acronym', 'registration_ids', 'brief_rationale']
    for pmid, group in attempts.groupby('pmid', sort=False):
        successes = group[group['error'].eq('')]
        if len(successes) > 1:
            comparable = successes[[column for column in classification_columns if column in successes.columns]].drop_duplicates()
            if len(comparable) > 1:
                raise ValueError(f'Conflicting successful {stage} classifications for PMID {pmid}.')
        candidates = successes if not successes.empty else group
        chosen = candidates.sort_values('attempt_number_numeric', kind='stable').iloc[-1]
        selected_rows.append(chosen.drop(labels=['attempt_number_numeric']).to_dict())
    selected = pd.DataFrame(selected_rows)
    missing_pmids = expected_pmids - set(selected.get('pmid', pd.Series(dtype=str)))
    if missing_pmids:
        raise ValueError(f'{stage} aggregation is incomplete; missing PMIDs: {sorted(missing_pmids)[:5]}')
    if selected['pmid'].duplicated().any() or len(selected) != len(expected_pmids):
        raise RuntimeError(f'{stage} aggregation did not produce exactly one row per expected PMID.')
    if 'technical_retry_pending' in selected.columns:
        pending = selected['technical_retry_pending'].str.casefold().eq('true')
        if pending.any():
            sample = selected.loc[pending, 'pmid'].tolist()[:5]
            raise ValueError(f'{stage} still has pending same-model technical retries: {sample}')
    validate_results(selected, name=f'Aggregated {stage} results')
    return (selected, attempts)

def restore_source_order(table: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    order = source[['pmid', 'source_row_number']].copy()
    ordered = order.merge(table, on='pmid', how='left', validate='one_to_one')
    if ordered.isna().all(axis=1).any():
        raise RuntimeError('Source-order restoration lost result rows.')
    ordered['source_row_number'] = pd.to_numeric(ordered['source_row_number'], errors='raise').astype(int)
    return ordered.sort_values('source_row_number', kind='stable').reset_index(drop=True)

def usage_summary(attempts: pd.DataFrame) -> dict[str, int]:
    summary: dict[str, int] = {'attempt_rows': len(attempts)}
    for column in ('input_tokens', 'cache_write_tokens', 'cached_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens'):
        summary[column] = int(pd.to_numeric(attempts.get(column, pd.Series(dtype=str)), errors='coerce').fillna(0).sum())
    return summary

def write_final_exports(final: pd.DataFrame, final_dir: Path) -> None:
    groups = {'primary_rct_candidates.csv': final[final['final_label'].eq('PRIMARY_RCT_REPORT')].copy(), 'unclear_publications.csv': final[final['final_label'].eq('UNCLEAR')].copy(), 'excluded_publications.csv': final[final['final_label'].eq('NOT_PRIMARY_RCT_REPORT')].copy()}
    pmid_sets = [set(table['pmid']) for table in groups.values()]
    if any((pmid_sets[i] & pmid_sets[j] for i in range(3) for j in range(i + 1, 3))):
        raise RuntimeError('Final export groups overlap.')
    if set().union(*pmid_sets) != set(final['pmid']):
        raise RuntimeError('Final export groups do not cover every PMID.')
    atomic_write_csv(final, final_dir / 'all_classifications.csv')
    for filename, table in groups.items():
        atomic_write_csv(table, final_dir / filename)


VALID_LABELS = {'PRIMARY_RCT_REPORT', 'NOT_PRIMARY_RCT_REPORT', 'UNCLEAR'}


def validate_results(table: pd.DataFrame, *, name: str) -> None:
    required = {'pmid', 'final_label', 'error'}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f'{name} is missing columns: {sorted(missing)}')
    if table['pmid'].astype(str).str.strip().eq('').any():
        raise ValueError(f'{name} contains empty PMIDs.')
    if table['pmid'].duplicated().any():
        raise ValueError(f'{name} contains duplicate PMIDs.')
    invalid = set(table['final_label']) - VALID_LABELS
    if invalid:
        raise ValueError(f'{name} contains invalid final labels: {sorted(invalid)}')
    error_not_unclear = table['error'].ne('') & table['final_label'].ne('UNCLEAR')
    if error_not_unclear.any():
        raise ValueError(f'{name} has error rows that are not labelled UNCLEAR.')
    if 'nct_abstract_regex_found' in table.columns:
        positive = table['final_label'].eq('PRIMARY_RCT_REPORT')
        invalid_positive = positive & ~table['nct_abstract_regex_found'].isin({'TRUE', 'FALSE'})
        invalid_nonpositive = ~positive & table['nct_abstract_regex_found'].ne('')
        if invalid_positive.any() or invalid_nonpositive.any():
            raise ValueError(f'{name} has invalid final-label-specific NCT regex flags.')
        if 'nct_abstract_registry_numbers' in table.columns:
            numbers_present = table['nct_abstract_registry_numbers'].ne('')
            inconsistent_numbers = table['nct_abstract_regex_found'].eq('TRUE') & ~numbers_present | table['nct_abstract_regex_found'].ne('TRUE') & numbers_present
            if inconsistent_numbers.any():
                raise ValueError(f'{name} has NCT numbers inconsistent with its regex flags.')
