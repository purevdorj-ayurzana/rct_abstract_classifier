from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from common import REQUIRED_INPUT_COLUMNS, normalize_pmid

def main() -> int:
    parser = argparse.ArgumentParser(description='Validate classification input.')
    parser.add_argument('--input', default='data/processed/publications_ready.csv')
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f'Input CSV not found: {path}')
    df = pd.read_csv(path, dtype=str).fillna('')
    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df['pmid'] = df['pmid'].map(normalize_pmid)
    errors = []
    if df['pmid'].eq('').any():
        errors.append(f'{int(df['pmid'].eq('').sum())} empty PMID values')
    if df['pmid'].duplicated().any():
        errors.append(f'{int(df['pmid'].duplicated().sum())} duplicate PMID values')
    if df['title'].str.strip().eq('').any():
        errors.append(f'{int(df['title'].str.strip().eq('').sum())} empty titles')
    if df['abstract'].str.strip().eq('').any():
        errors.append(f'{int(df['abstract'].str.strip().eq('').sum())} empty abstracts')
    print(f'Rows: {len(df)}')
    print(f'Unique PMIDs: {df['pmid'].nunique()}')
    print(f'Median abstract characters: {int(df['abstract'].str.len().median())}')
    print(f'Maximum abstract characters: {int(df['abstract'].str.len().max())}')
    if errors:
        print('\nValidation failed:')
        for error in errors:
            print(f'- {error}')
        return 1
    print('\nValidation passed.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
