from common import relative_location
import argparse
import json
import os
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any
import pandas as pd
from common import NCT_REGISTRY_PATTERN, canonical_json_hash, load_json, sha256_file, sha256_text, utc_now_iso
from primary_endpoint_workflow import select_upstream_positive
EXPORT_FORMAT_VERSION = 2
CURATED_EXPORT_FORMAT_VERSION = 3
DEFAULT_OUTPUT_ROOT = Path('outputs/exports/clinicaltrials_history')
CLASSIFIER_ARM = 'sol'
SELECTION_RULE = 'sol.final_label == PRIMARY_RCT_REPORT'
ABSTRACT_REGEX_METHOD = 'ABSTRACT_REGEX'
MANUAL_CORRECTION_METHOD = 'MANUAL_VERIFIED_CORRECTION'
CURATED_CANDIDATE_RULE = 'abstract_regex OR manual_verified_correction'
PUBLICATION_COLUMNS = ('source_row_number', 'pmid', 'title', 'abstract', 'doi', 'journal', 'publication_year', 'publication_date', 'publication_types', 'classifier_run_id', 'classifier_arm', 'classifier_model', 'classifier_reasoning_effort', 'classifier_label', 'underlying_trial_randomized', 'publication_role', 'trial_name_or_acronym', 'registration_ids', 'brief_rationale', 'nct_abstract_regex_found', 'nct_abstract_registry_numbers', 'classifier_prompt_hash', 'classifier_schema_hash', 'classifier_input_hash', 'classifier_request_hash', 'classifier_response_id')
NCT_ASSOCIATION_COLUMNS = ('pmid', 'nct_id_raw', 'nct_id_normalized', 'classifier_run_id', 'classifier_arm', 'classifier_label', 'nct_extraction_method', 'nct_source_field')
CORRECTION_COLUMNS = ('pmid', 'nct_id_source_text', 'nct_id_normalized', 'nct_source_field', 'correction_type', 'correction_reason', 'publication_evidence_url', 'registry_evidence_url', 'verified_date')
CORRECTION_TYPES = {'MISSING_TOKEN_BOUNDARY', 'TRUNCATED_NCT_IDENTIFIER'}

def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, encoding='utf-8-sig')

def validate_curated_corrections(publications: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """Validate a manually verified correction ledger and build association rows."""
    if list(corrections.columns) != list(CORRECTION_COLUMNS):
        raise ValueError('Curated NCT correction ledger schema mismatch.')
    corrections = corrections.fillna('').copy()
    if corrections.empty:
        raise ValueError('Curated NCT correction ledger must contain at least one row.')
    if corrections.duplicated(['pmid', 'nct_id_normalized']).any():
        raise ValueError('Curated NCT correction ledger contains duplicate associations.')
    publications_by_pmid = {str(row['pmid']): row for row in publications.to_dict(orient='records')}
    association_rows: list[dict[str, str]] = []
    for correction in corrections.to_dict(orient='records'):
        pmid = str(correction['pmid']).strip()
        publication = publications_by_pmid.get(pmid)
        if publication is None:
            raise ValueError(f'Curated NCT correction references unknown PMID {pmid}.')
        normalized = str(correction['nct_id_normalized']).strip().upper()
        if not re.fullmatch('NCT\\d{8}', normalized):
            raise ValueError(f'Curated NCT correction for PMID {pmid} has a non-canonical target.')
        if correction['nct_id_normalized'] != normalized:
            raise ValueError(f'Curated NCT correction for PMID {pmid} must store an uppercase canonical target.')
        source_field = str(correction['nct_source_field']).strip()
        if source_field != 'abstract':
            raise ValueError('Curated NCT corrections currently require abstract source text.')
        source_text = str(correction['nct_id_source_text']).strip()
        if not source_text or source_text.casefold() not in str(publication[source_field]).casefold():
            raise ValueError(f'Curated NCT correction source text is absent from PMID {pmid}.')
        if str(publication['nct_abstract_regex_found']).strip().upper() != 'FALSE':
            raise ValueError(f'Curated NCT correction for PMID {pmid} duplicates a regex-positive publication.')
        correction_type = str(correction['correction_type']).strip()
        if correction_type not in CORRECTION_TYPES:
            raise ValueError(f'Curated NCT correction for PMID {pmid} has an unsupported correction type.')
        if not str(correction['correction_reason']).strip():
            raise ValueError(f'Curated NCT correction for PMID {pmid} has no correction reason.')
        expected_publication_url = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
        if correction['publication_evidence_url'] != expected_publication_url:
            raise ValueError(f'Curated NCT correction for PMID {pmid} has an unexpected publication URL.')
        expected_registry_url = f'https://clinicaltrials.gov/study/{normalized}'
        if correction['registry_evidence_url'] != expected_registry_url:
            raise ValueError(f'Curated NCT correction for PMID {pmid} has an unexpected registry URL.')
        try:
            date.fromisoformat(str(correction['verified_date']))
        except ValueError as exc:
            raise ValueError(f'Curated NCT correction for PMID {pmid} has an invalid verification date.') from exc
        association_rows.append({'pmid': pmid, 'nct_id_raw': source_text, 'nct_id_normalized': normalized, 'classifier_run_id': str(publication['classifier_run_id']), 'classifier_arm': str(publication['classifier_arm']), 'classifier_label': str(publication['classifier_label']), 'nct_extraction_method': MANUAL_CORRECTION_METHOD, 'nct_source_field': source_field})
    return pd.DataFrame(association_rows, columns=NCT_ASSOCIATION_COLUMNS)

def _validate_frozen_run_artifacts(run_dir: Path, manifest: dict[str, Any], classifier_arm: str) -> dict[str, dict[str, str]]:
    artifacts = {'run_manifest': run_dir / 'run_manifest.json', 'source_snapshot': run_dir / 'source_snapshot.csv', 'config_snapshot': run_dir / 'config_snapshot.json', 'schema_snapshot': run_dir / 'schema_snapshot.json', 'prompt_snapshot': run_dir / 'prompts' / 'classification.txt', 'classifier_final': run_dir / classifier_arm / 'final' / 'all_classifications.csv', 'classifier_qa': run_dir / classifier_arm / 'qa' / 'run_summary.json'}
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Cannot export an incomplete finalized {classifier_arm} run; missing: ' + ', '.join(missing))
    source_hash = sha256_file(artifacts['source_snapshot'])
    if source_hash != manifest.get('source_snapshot_sha256'):
        raise ValueError('The source snapshot hash does not match the run manifest.')
    config_hash = sha256_file(artifacts['config_snapshot'])
    if config_hash != manifest.get('config_sha256'):
        raise ValueError('The configuration snapshot hash does not match the run manifest.')
    schema = load_json(artifacts['schema_snapshot'])
    if canonical_json_hash(schema) != manifest.get('schema_hash'):
        raise ValueError('The schema snapshot hash does not match the run manifest.')
    prompt = artifacts['prompt_snapshot'].read_text(encoding='utf-8').strip()
    if sha256_text(prompt) != manifest.get('classification_prompt_hash'):
        raise ValueError('The prompt snapshot hash does not match the run manifest.')
    return {name: {'path': relative_location(path.resolve()), 'sha256': sha256_file(path)} for name, path in artifacts.items()}

def _selected_publications(selected: pd.DataFrame, classifier_arm: str) -> pd.DataFrame:
    source = selected.copy().fillna('')
    mapping = {'upstream_rct_run_id': 'classifier_run_id', 'upstream_rct_arm': 'classifier_arm', 'upstream_rct_final_model': 'classifier_model', 'upstream_rct_reasoning_effort': 'classifier_reasoning_effort', 'upstream_rct_final_label': 'classifier_label', 'upstream_rct_underlying_trial_randomized': 'underlying_trial_randomized', 'upstream_rct_publication_role': 'publication_role', 'upstream_rct_trial_name_or_acronym': 'trial_name_or_acronym', 'upstream_rct_registration_ids': 'registration_ids', 'upstream_rct_brief_rationale': 'brief_rationale', 'upstream_rct_nct_abstract_regex_found': 'nct_abstract_regex_found', 'upstream_rct_nct_abstract_registry_numbers': 'nct_abstract_registry_numbers', 'upstream_rct_prompt_hash': 'classifier_prompt_hash', 'upstream_rct_schema_hash': 'classifier_schema_hash', 'upstream_rct_input_hash': 'classifier_input_hash', 'upstream_rct_request_hash': 'classifier_request_hash', 'upstream_rct_response_id': 'classifier_response_id'}
    source = source.rename(columns=mapping)
    optional_publication_columns = {'doi', 'journal', 'publication_year', 'publication_date', 'publication_types'}
    for column in optional_publication_columns:
        if column not in source.columns:
            source[column] = ''
    missing = set(PUBLICATION_COLUMNS) - set(source.columns)
    if missing:
        raise ValueError(f'Finalized {classifier_arm} output cannot satisfy the history export schema; missing columns: {sorted(missing)}')
    result = source[list(PUBLICATION_COLUMNS)].copy()
    if set(result['classifier_arm']) != {classifier_arm}:
        raise ValueError(f'History exports must contain only finalized {classifier_arm} rows.')
    if set(result['classifier_label']) != {'PRIMARY_RCT_REPORT'}:
        raise ValueError(f'History exports must contain only {classifier_arm}-positive rows.')
    if result['pmid'].eq('').any() or result['pmid'].duplicated().any():
        raise ValueError('Selected history publications have invalid PMID identity.')
    return result

def _nct_associations(publications: pd.DataFrame) -> pd.DataFrame:
    associations: list[dict[str, str]] = []
    for row in publications.to_dict(orient='records'):
        matches: dict[str, str] = {}
        for match in NCT_REGISTRY_PATTERN.finditer(str(row['abstract'] or '')):
            normalized = f'NCT{match.group(1)}'
            matches.setdefault(normalized, match.group(0))
        normalized_ids = list(matches)
        expected_value = '; '.join(normalized_ids)
        expected_found = 'TRUE' if normalized_ids else 'FALSE'
        if row['nct_abstract_regex_found'].strip().upper() != expected_found:
            raise ValueError(f'PMID {row['pmid']} has an NCT-found flag inconsistent with its abstract.')
        if row['nct_abstract_registry_numbers'].strip() != expected_value:
            raise ValueError(f'PMID {row['pmid']} has stored NCT identifiers inconsistent with its abstract.')
        for normalized, raw in matches.items():
            associations.append({'pmid': row['pmid'], 'nct_id_raw': raw, 'nct_id_normalized': normalized, 'classifier_run_id': row['classifier_run_id'], 'classifier_arm': row['classifier_arm'], 'classifier_label': row['classifier_label'], 'nct_extraction_method': ABSTRACT_REGEX_METHOD, 'nct_source_field': 'abstract'})
    result = pd.DataFrame(associations, columns=NCT_ASSOCIATION_COLUMNS)
    if not result.empty and result.duplicated(['pmid', 'nct_id_normalized']).any():
        raise RuntimeError('History export produced duplicate PMID-NCT associations.')
    return result

def verify_history_export(output_dir: Path) -> dict[str, int]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'History export manifest not found: {manifest_path}')
    manifest = load_json(manifest_path)
    version = manifest.get('export_format_version')
    expected_arm = 'sol'
    expected_rule = SELECTION_RULE
    if manifest.get('selection_rule') != expected_rule:
        raise ValueError('History export selection rule is not the expected rule.')
    if manifest.get('classifier_arm') != expected_arm:
        raise ValueError('History export classifier arm does not match its version.')
    loaded: dict[str, pd.DataFrame] = {}
    expected_columns = {'selected_publications.csv': list(PUBLICATION_COLUMNS), 'nct_candidates.csv': list(NCT_ASSOCIATION_COLUMNS)}
    if version == CURATED_EXPORT_FORMAT_VERSION:
        expected_columns['nct_candidate_corrections.csv'] = list(CORRECTION_COLUMNS)
    for filename, columns in expected_columns.items():
        path = output_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f'History export file is missing: {path}')
        record = (manifest.get('files') or {}).get(filename)
        if not isinstance(record, dict):
            raise ValueError(f'Manifest has no file record for {filename}.')
        if sha256_file(path) != record.get('sha256'):
            raise ValueError(f'History export hash mismatch for {filename}.')
        table = pd.read_csv(path, dtype=str).fillna('')
        if list(table.columns) != columns or record.get('columns') != columns:
            raise ValueError(f'History export schema mismatch for {filename}.')
        if len(table) != int(record.get('rows', -1)):
            raise ValueError(f'History export row-count mismatch for {filename}.')
        loaded[filename] = table
    publications = loaded['selected_publications.csv']
    associations = loaded['nct_candidates.csv']
    if publications['pmid'].eq('').any() or publications['pmid'].duplicated().any():
        raise ValueError('History export publications have invalid PMID identity.')
    if set(publications['classifier_arm']) != {expected_arm}:
        raise ValueError(f'History export contains a non-{expected_arm} publication row.')
    if set(publications['classifier_label']) != {'PRIMARY_RCT_REPORT'}:
        raise ValueError('History export contains a non-positive publication row.')
    if not associations.empty:
        if associations.duplicated(['pmid', 'nct_id_normalized']).any():
            raise ValueError('History export has duplicate PMID-NCT associations.')
        if not set(associations['pmid']).issubset(set(publications['pmid'])):
            raise ValueError('History export has NCT associations for unknown PMIDs.')
    pmids_flagged_with_nct = set(publications.loc[publications['nct_abstract_regex_found'].eq('TRUE'), 'pmid'])
    if version == CURATED_EXPORT_FORMAT_VERSION:
        methods = set(associations['nct_extraction_method'])
        if not methods.issubset({ABSTRACT_REGEX_METHOD, MANUAL_CORRECTION_METHOD}):
            raise ValueError('Curated history export contains an unsupported NCT method.')
        regex_associations = associations.loc[associations['nct_extraction_method'].eq(ABSTRACT_REGEX_METHOD)]
        manual_associations = associations.loc[associations['nct_extraction_method'].eq(MANUAL_CORRECTION_METHOD)]
        if set(regex_associations['pmid']) != pmids_flagged_with_nct:
            raise ValueError('Curated history export regex associations do not match publication flags.')
        corrections = loaded['nct_candidate_corrections.csv']
        expected_manual = validate_curated_corrections(publications, corrections)
        manual_columns = list(NCT_ASSOCIATION_COLUMNS)
        if manual_associations[manual_columns].to_dict(orient='records') != expected_manual[manual_columns].to_dict(orient='records'):
            raise ValueError('Curated history export manual associations do not match its correction ledger.')
        if manifest.get('correction_set_hash') != canonical_json_hash(corrections.to_dict(orient='records')):
            raise ValueError('Curated history export correction-set hash mismatch.')
        if manifest.get('candidate_selection_rule') != CURATED_CANDIDATE_RULE:
            raise ValueError('Curated history export candidate-selection rule mismatch.')
        base_export = manifest.get('base_export')
        if not isinstance(base_export, dict):
            raise ValueError('Curated history export has no base-export lineage.')
        if base_export.get('export_format_version') != EXPORT_FORMAT_VERSION:
            raise ValueError('Curated history export must derive from format version 2.')
        if base_export.get('classifier_run_id') != manifest.get('classifier_run_id'):
            raise ValueError('Curated history export classifier lineage mismatch.')
        if base_export.get('selected_publications_sha256') != sha256_file(output_dir / 'selected_publications.csv'):
            raise ValueError('Curated history export changed its base publication file.')
        regex_association_hash = canonical_json_hash(sorted((f'{row.pmid}|{row.nct_id_normalized}' for row in regex_associations.itertuples(index=False))))
        if base_export.get('association_hash') != regex_association_hash:
            raise ValueError('Curated history export regex lineage hash mismatch.')
        regex_nct_ids = sorted(regex_associations['nct_id_normalized'].unique().tolist())
        if base_export.get('unique_nct_id_hash') != canonical_json_hash(regex_nct_ids):
            raise ValueError('Curated history export base NCT lineage hash mismatch.')
    else:
        if set(associations['pmid']) != pmids_flagged_with_nct:
            raise ValueError('History export NCT candidates do not match publication flags.')
        manual_associations = associations.iloc[0:0]
    pmids_with_candidates = set(associations['pmid'])
    unique_nct_ids = sorted(associations['nct_id_normalized'].unique().tolist())
    association_hash = canonical_json_hash(sorted((f'{row.pmid}|{row.nct_id_normalized}' for row in associations.itertuples(index=False))))
    checks = {'selected_publications': len(publications), 'selected_publications_with_nct': len(pmids_with_candidates), 'selected_publications_without_nct': len(publications) - len(pmids_with_candidates), 'pmid_nct_associations': len(associations), 'unique_candidate_nct_ids': len(unique_nct_ids)}
    if version == CURATED_EXPORT_FORMAT_VERSION:
        checks.update({'selected_publications_with_abstract_regex_nct': len(pmids_flagged_with_nct), 'selected_publications_with_manual_verified_correction': len(set(manual_associations['pmid'])), 'manual_verified_corrections': len(manual_associations)})
    for name, value in checks.items():
        if int(manifest.get(name, -1)) != value:
            raise ValueError(f'History export manifest count mismatch for {name}.')
    if manifest.get('selected_pmid_hash') != canonical_json_hash(sorted(publications['pmid'].tolist())):
        raise ValueError('History export selected PMID hash mismatch.')
    if manifest.get('association_hash') != association_hash:
        raise ValueError('History export association hash mismatch.')
    if manifest.get('unique_nct_id_hash') != canonical_json_hash(unique_nct_ids):
        raise ValueError('History export unique NCT ID hash mismatch.')
    return checks

def export_history_cohort(*, run_dir: Path, output_root: Path=DEFAULT_OUTPUT_ROOT) -> Path:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / 'run_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Run manifest not found: {manifest_path}')
    run_manifest = load_json(manifest_path)
    if CLASSIFIER_ARM not in (run_manifest.get('active_arms') or []):
        raise ValueError(f'History export requires a finalized {CLASSIFIER_ARM} arm.')
    run_id = str(run_manifest.get('run_id') or run_dir.name)
    if run_id != run_dir.name:
        raise ValueError('Run directory name does not match the recorded run ID.')
    artifacts = _validate_frozen_run_artifacts(run_dir, run_manifest, CLASSIFIER_ARM)
    selected, lineage = select_upstream_positive(run_dir, CLASSIFIER_ARM)
    publications = _selected_publications(selected, CLASSIFIER_ARM)
    associations = _nct_associations(publications)
    expected_source_rows = int(run_manifest.get('source_row_count', -1))
    if int(lineage['upstream_rows']) != expected_source_rows:
        raise ValueError(f'Finalized {CLASSIFIER_ARM} row count does not match the run manifest.')
    source = pd.read_csv(run_dir / 'source_snapshot.csv', dtype=str).fillna('')
    if canonical_json_hash(sorted(source['pmid'].tolist())) != run_manifest.get('source_pmid_hash'):
        raise ValueError('The source PMID hash does not match the run manifest.')
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / run_id
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing history export: {output_dir}')
    temporary = Path(tempfile.mkdtemp(prefix=f'.{run_id}.', dir=output_root))
    try:
        publication_path = temporary / 'selected_publications.csv'
        association_path = temporary / 'nct_candidates.csv'
        _write_csv(publications, publication_path)
        _write_csv(associations, association_path)
        with_nct = int(publications['nct_abstract_regex_found'].eq('TRUE').sum())
        unique_nct_ids = sorted(associations['nct_id_normalized'].unique().tolist())
        export_manifest = {'export_format_version': EXPORT_FORMAT_VERSION, 'export_type': 'clinicaltrials_gov_history_cohort', 'created_at_utc': utc_now_iso(), 'api_requests_made': False, 'source_project': 'rct_abstract_classifier', 'classifier_run_id': run_id, 'classifier_run_dir': relative_location(run_dir), 'classifier_workflow_mode': run_manifest['workflow_mode'], 'classifier_arm': CLASSIFIER_ARM, 'classifier_model': run_manifest['arm_settings'][CLASSIFIER_ARM]['model'], 'classifier_reasoning_effort': run_manifest['arm_settings'][CLASSIFIER_ARM]['reasoning_effort'], 'selection_rule': SELECTION_RULE, 'source_rows': expected_source_rows, 'selected_publications': len(publications), 'selected_publications_with_nct': with_nct, 'selected_publications_without_nct': len(publications) - with_nct, 'pmid_nct_associations': len(associations), 'unique_candidate_nct_ids': len(unique_nct_ids), 'selected_pmid_hash': canonical_json_hash(sorted(publications['pmid'].tolist())), 'association_hash': canonical_json_hash(sorted((f'{row.pmid}|{row.nct_id_normalized}' for row in associations.itertuples(index=False)))), 'unique_nct_id_hash': canonical_json_hash(unique_nct_ids), 'source_artifacts': artifacts, 'files': {'selected_publications.csv': {'sha256': sha256_file(publication_path), 'rows': len(publications), 'columns': list(publications.columns)}, 'nct_candidates.csv': {'sha256': sha256_file(association_path), 'rows': len(associations), 'columns': list(associations.columns)}}, 'limitations': ['NCT candidates are detected only by deterministic regex in the supplied abstract.', 'No ClinicalTrials.gov existence check was performed by this exporter.', 'Absence of an abstract NCT identifier does not establish that a trial was unregistered.', 'Human agreement was assessed on a separate 50-publication sample.']}
        (temporary / 'manifest.json').write_text(json.dumps(export_manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    verify_history_export(output_dir)
    return output_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create an immutable Sol-positive cohort export for a separate ClinicalTrials.gov history project.')
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    output = export_history_cohort(run_dir=args.run_dir, output_root=args.output_root)
    manifest = load_json(output / 'manifest.json')
    print(f'Created immutable ClinicalTrials.gov history export: {output}')
    print(f'Selected Sol-positive publications: {manifest['selected_publications']}')
    print(f'Publications with abstract NCT identifiers: {manifest['selected_publications_with_nct']}')
    print(f'Publications without abstract NCT identifiers: {manifest['selected_publications_without_nct']}')
    print(f'PMID-NCT associations: {manifest['pmid_nct_associations']}')
    print(f'Unique candidate NCT IDs: {manifest['unique_candidate_nct_ids']}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
