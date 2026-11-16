from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import pandas as pd
from result_integrity import atomic_write_csv, resolve_stage_results, restore_source_order, usage_summary, write_final_exports
from result_integrity import validate_results
from common import atomic_write_json, extract_nct_ids, load_json, utc_now_iso
LABEL_ORDER = ['PRIMARY_RCT_REPORT', 'NOT_PRIMARY_RCT_REPORT', 'UNCLEAR']

def add_final_fields(results: pd.DataFrame, arm: str) -> pd.DataFrame:
    final = results.copy().fillna('')
    final['arm'] = arm
    final['final_model'] = final.apply(lambda row: row.get('actual_model', '') or row.get('model', ''), axis=1)
    nct_ids = [extract_nct_ids(row.get('abstract', '')) if row.get('final_label') == 'PRIMARY_RCT_REPORT' else [] for row in final.to_dict(orient='records')]
    final['nct_abstract_regex_found'] = [('TRUE' if identifiers else 'FALSE') if label == 'PRIMARY_RCT_REPORT' else '' for label, identifiers in zip(final['final_label'], nct_ids, strict=True)]
    final['nct_abstract_registry_numbers'] = ['; '.join(identifiers) if label == 'PRIMARY_RCT_REPORT' else '' for label, identifiers in zip(final['final_label'], nct_ids, strict=True)]
    validate_results(final, name=f'Final {arm} results')
    return final

def write_arm_qa(run_dir: Path, arm: str, final: pd.DataFrame, attempts: pd.DataFrame) -> dict[str, Any]:
    qa_dir = run_dir / arm / 'qa'
    label_counts = final.groupby('final_label', dropna=False).size().rename('count').reset_index()
    role_counts = final.groupby('publication_role', dropna=False).size().rename('count').reset_index()
    technical_failures = final[final['error'].ne('')].copy()
    primary = final[final['final_label'].eq('PRIMARY_RCT_REPORT')].copy()
    nct_found = int(primary['nct_abstract_regex_found'].eq('TRUE').sum())
    nct_total = len(primary)
    nct_summary = pd.DataFrame([{'primary_rct_reports': nct_total, 'nct_abstract_regex_found': nct_found, 'nct_abstract_regex_not_found': nct_total - nct_found, 'nct_abstract_regex_found_percentage': round(100 * nct_found / nct_total, 2) if nct_total else 0.0}])
    for filename, table in (('label_counts.csv', label_counts), ('publication_role_counts.csv', role_counts), ('technical_failures.csv', technical_failures), ('nct_registry_summary.csv', nct_summary)):
        atomic_write_csv(table, qa_dir / filename)
    summary = {'generated_at_utc': utc_now_iso(), 'integrity_passed': True, 'arm': arm, 'final_rows': len(final), 'unique_pmids': int(final['pmid'].nunique()), 'label_counts': final['final_label'].value_counts().sort_index().to_dict(), 'unresolved_technical_failures': len(technical_failures), 'primary_rct_reports': nct_total, 'nct_abstract_regex_found': nct_found, 'nct_abstract_regex_found_percentage': round(100 * nct_found / nct_total, 2) if nct_total else 0.0, 'usage_all_attempts': usage_summary(attempts)}
    atomic_write_json(qa_dir / 'run_summary.json', summary)
    return summary

def finalize_arm(run_dir: Path, arm: str, source: pd.DataFrame) -> pd.DataFrame:
    expected = set(source['pmid'])
    selected, attempts = resolve_stage_results(run_dir, arm, expected)
    selected = restore_source_order(selected, source)
    final = add_final_fields(selected, arm)
    if len(final) != len(source) or set(final['pmid']) != expected:
        raise RuntimeError(f'{arm} final output does not exactly match the run source.')
    atomic_write_csv(final, run_dir / arm / 'aggregate' / 'results.csv')
    write_final_exports(final, run_dir / arm / 'final')
    write_arm_qa(run_dir, arm, final, attempts)
    return final

def finalize_run(run_dir):
    manifest = load_json(run_dir / 'run_manifest.json')
    if manifest.get('workflow_mode') != 'single_model_arm' or manifest.get('active_arms') != ['sol']:
        raise ValueError('Expected a complete single-model sol run.')
    source = pd.read_csv(run_dir / 'source_snapshot.csv', dtype=str).fillna('')
    if source['pmid'].duplicated().any() or source['pmid'].eq('').any():
        raise ValueError('Run source has invalid PMID identity.')
    return {'sol': finalize_arm(run_dir, 'sol', source)}

def main():
    parser = argparse.ArgumentParser(description='Finalize and check an RCT run.')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    final = finalize_run(Path(args.run_dir))['sol']
    print(f'Finalized {len(final)} RCT classifications.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
