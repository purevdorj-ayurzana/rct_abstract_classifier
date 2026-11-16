from __future__ import annotations
import argparse
import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
import pandas as pd
from common import canonical_json_hash, load_json, sha256_file, utc_now_iso
from export_history_cohort import CLASSIFIER_ARM, CORRECTION_COLUMNS, CURATED_CANDIDATE_RULE, CURATED_EXPORT_FORMAT_VERSION, DEFAULT_OUTPUT_ROOT, EXPORT_FORMAT_VERSION, NCT_ASSOCIATION_COLUMNS, PUBLICATION_COLUMNS, SELECTION_RULE, _write_csv, validate_curated_corrections, verify_history_export

def _file_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {'sha256': sha256_file(path), 'rows': len(table), 'columns': list(table.columns)}

def augment_history_export(*, base_export_dir: Path, corrections_file: Path, output_root: Path=DEFAULT_OUTPUT_ROOT) -> Path:
    """Create a new immutable curated export without modifying its version-2 base."""
    base_export_dir = base_export_dir.resolve()
    corrections_file = corrections_file.resolve()
    verify_history_export(base_export_dir)
    base_manifest = load_json(base_export_dir / 'manifest.json')
    if base_manifest.get('export_format_version') != EXPORT_FORMAT_VERSION:
        raise ValueError('Curated history export requires a version-2 base export.')
    if base_manifest.get('classifier_arm') != CLASSIFIER_ARM:
        raise ValueError('Curated history export requires a Sol base export.')
    if base_manifest.get('selection_rule') != SELECTION_RULE:
        raise ValueError('Curated history export base selection rule mismatch.')
    if not corrections_file.is_file():
        raise FileNotFoundError(f'NCT correction ledger not found: {corrections_file}')
    publications = pd.read_csv(base_export_dir / 'selected_publications.csv', dtype=str).fillna('')
    associations = pd.read_csv(base_export_dir / 'nct_candidates.csv', dtype=str).fillna('')
    corrections = pd.read_csv(corrections_file, dtype=str).fillna('')
    if list(publications.columns) != list(PUBLICATION_COLUMNS):
        raise ValueError('Base history export publication schema mismatch.')
    if list(associations.columns) != list(NCT_ASSOCIATION_COLUMNS):
        raise ValueError('Base history export association schema mismatch.')
    manual_associations = validate_curated_corrections(publications, corrections)
    augmented = pd.concat([associations, manual_associations], ignore_index=True)
    if augmented.duplicated(['pmid', 'nct_id_normalized']).any():
        raise ValueError('Curated NCT corrections duplicate an existing association.')
    correction_set_hash = canonical_json_hash(corrections[list(CORRECTION_COLUMNS)].to_dict(orient='records'))
    export_id = f'{base_manifest['classifier_run_id']}_curated_{correction_set_hash[:8]}'
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / export_id
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing curated history export: {output_dir}')
    temporary = Path(tempfile.mkdtemp(prefix=f'.{export_id}.', dir=output_root))
    try:
        publication_path = temporary / 'selected_publications.csv'
        association_path = temporary / 'nct_candidates.csv'
        correction_path = temporary / 'nct_candidate_corrections.csv'
        shutil.copy2(base_export_dir / 'selected_publications.csv', publication_path)
        _write_csv(augmented, association_path)
        _write_csv(corrections[list(CORRECTION_COLUMNS)], correction_path)
        unique_nct_ids = sorted(augmented['nct_id_normalized'].unique().tolist())
        pmids_with_candidates = set(augmented['pmid'])
        regex_pmids = set(publications.loc[publications['nct_abstract_regex_found'].eq('TRUE'), 'pmid'])
        manual_pmids = set(manual_associations['pmid'])
        manifest = deepcopy(base_manifest)
        manifest.update({'export_format_version': CURATED_EXPORT_FORMAT_VERSION, 'export_id': export_id, 'created_at_utc': utc_now_iso(), 'api_requests_made': False, 'candidate_selection_rule': CURATED_CANDIDATE_RULE, 'selected_publications_with_nct': len(pmids_with_candidates), 'selected_publications_without_nct': len(publications) - len(pmids_with_candidates), 'selected_publications_with_abstract_regex_nct': len(regex_pmids), 'selected_publications_with_manual_verified_correction': len(manual_pmids), 'manual_verified_corrections': len(manual_associations), 'pmid_nct_associations': len(augmented), 'unique_candidate_nct_ids': len(unique_nct_ids), 'association_hash': canonical_json_hash(sorted((f'{row.pmid}|{row.nct_id_normalized}' for row in augmented.itertuples(index=False)))), 'unique_nct_id_hash': canonical_json_hash(unique_nct_ids), 'correction_set_hash': correction_set_hash, 'base_export': {'export_format_version': EXPORT_FORMAT_VERSION, 'classifier_run_id': base_manifest['classifier_run_id'], 'manifest_sha256': sha256_file(base_export_dir / 'manifest.json'), 'selected_publications_sha256': sha256_file(base_export_dir / 'selected_publications.csv'), 'nct_candidates_sha256': sha256_file(base_export_dir / 'nct_candidates.csv'), 'association_hash': base_manifest['association_hash'], 'unique_nct_id_hash': base_manifest['unique_nct_id_hash']}, 'files': {'selected_publications.csv': _file_record(publication_path, publications), 'nct_candidates.csv': _file_record(association_path, augmented), 'nct_candidate_corrections.csv': _file_record(correction_path, corrections)}, 'limitations': ['Regex-derived NCT associations remain exactly reproducible from the supplied abstracts.', 'Additional associations are limited to the hashed, source-cited manual correction ledger.', 'This augmentation command made no OpenAI or ClinicalTrials.gov requests.', 'Absence of an accepted NCT candidate does not establish that a trial was unregistered.', 'Human agreement was assessed on a separate 50-publication sample.']})
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    verify_history_export(output_dir)
    return output_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create an immutable version-3 ClinicalTrials.gov history export from a verified version-2 export and a curated NCT correction ledger.')
    parser.add_argument('--base-export-dir', required=True, type=Path)
    parser.add_argument('--corrections-file', required=True, type=Path)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    output = augment_history_export(base_export_dir=args.base_export_dir, corrections_file=args.corrections_file, output_root=args.output_root)
    manifest = load_json(output / 'manifest.json')
    print(f'Created immutable curated history export: {output}')
    print(f'Selected Sol-positive publications: {manifest['selected_publications']}')
    print(f'Publications with abstract-regex NCT identifiers: {manifest['selected_publications_with_abstract_regex_nct']}')
    print(f'Publications with manual verified corrections: {manifest['selected_publications_with_manual_verified_correction']}')
    print(f'PMID-NCT associations: {manifest['pmid_nct_associations']}')
    print(f'Unique candidate NCT IDs: {manifest['unique_candidate_nct_ids']}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
