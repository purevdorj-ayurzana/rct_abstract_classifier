from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
REQUIRED_INPUT_COLUMNS = {'pmid', 'title', 'abstract'}
NCT_REGISTRY_PATTERN = re.compile('(?<![A-Za-z0-9])NCT\\s*[-:]?\\s*(\\d{8})(?!\\d)', flags=re.IGNORECASE)

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def normalize_whitespace(value: Any) -> str:
    return re.sub('\\s+', ' ', str(value or '')).strip()

def normalize_pmid(value: Any) -> str:
    text = normalize_whitespace(value)
    return re.sub('^PMID\\s*:?\\s*', '', text, flags=re.IGNORECASE)

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def canonical_json_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')))

def atomic_write_text(path: str | Path, content: str, *, encoding: str='utf-8') -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding=encoding)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n')

def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def responses_endpoint(config):
    return '/v1/responses'

def build_responses_request_body(config, *, model, reasoning_effort, input_items, text_format):
    return {'model': model, 'reasoning': {'effort': reasoning_effort}, 'input': input_items, 'text': {'format': text_format}}

def load_prompt(path: str | Path) -> tuple[str, str]:
    text = Path(path).read_text(encoding='utf-8').strip()
    return (text, sha256_text(text))

def build_user_input(row: dict[str, Any]) -> str:
    fields = [('PMID', normalize_pmid(row.get('pmid', ''))), ('TITLE', normalize_whitespace(row.get('title', ''))), ('ABSTRACT', normalize_whitespace(row.get('abstract', ''))), ('JOURNAL', normalize_whitespace(row.get('journal', ''))), ('PUBLICATION DATE', normalize_whitespace(row.get('publication_date', row.get('publication_year', '')))), ('PUBMED PUBLICATION TYPES', normalize_whitespace(row.get('publication_types', ''))), ('DOI', normalize_whitespace(row.get('doi', '')))]
    return '\n\n'.join((f'{name}:\n{value or '[not supplied]'}' for name, value in fields))

def make_input_hash(row: dict[str, Any]) -> str:
    return sha256_text(build_user_input(row))

def configured_model_arms(config):
    if config.get('workflow_mode') != 'single_model_arm' or list(config.get('model_arms', {})) != ['sol']:
        raise ValueError('The RCT workflow requires exactly the sol model.')
    settings = config['model_arms']['sol']
    if settings.get('model') != 'gpt-5.6-sol' or settings.get('reasoning_effort') != 'medium':
        raise ValueError('The study uses gpt-5.6-sol with medium reasoning.')
    return ['sol']

def stage_settings(config, stage):
    if stage not in configured_model_arms(config):
        raise ValueError('Use stage sol.')
    settings = config['model_arms'][stage]
    return (str(settings['model']), str(config['classification_prompt']), str(settings['reasoning_effort']))

def extract_registry_ids(text: str) -> list[str]:
    patterns = ['\\bNCT\\d{8}\\b', '\\bISRCTN\\d{8}\\b', '\\bACTRN\\d{14}\\b', '\\bUMIN(?:-CTR)?\\s*[:#]?\\s*U?MIN\\d+\\b', '\\bChiCTR[-A-Za-z]*\\d+\\b', '\\bEudraCT\\s*(?:No\\.?|number)?\\s*[:#]?\\s*\\d{4}-\\d{6}-\\d{2}\\b']
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    unique, seen = ([], set())
    for item in found:
        normalized = normalize_whitespace(item)
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique

def extract_nct_ids(text: str) -> list[str]:
    """Return unique, canonical ClinicalTrials.gov identifiers found in text."""
    found: list[str] = []
    seen: set[str] = set()
    for digits in NCT_REGISTRY_PATTERN.findall(str(text or '')):
        identifier = f'NCT{digits}'
        if identifier not in seen:
            seen.add(identifier)
            found.append(identifier)
    return found

def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')

def json_schema_for_batch():
    return load_json(Path(__file__).resolve().parents[1] / 'schemas/rct.json')

def response_output_text(body: dict[str, Any]) -> str:
    direct = body.get('output_text')
    if isinstance(direct, str) and direct.strip():
        return direct
    pieces: list[str] = []
    for output_item in body.get('output', []) or []:
        if output_item.get('type') != 'message':
            continue
        for content_item in output_item.get('content', []) or []:
            if content_item.get('type') == 'output_text':
                text = content_item.get('text')
                if isinstance(text, str):
                    pieces.append(text)
    if not pieces:
        raise ValueError('No output_text content found in Responses API body.')
    return ''.join(pieces)

def response_incomplete_reason(body: dict[str, Any]) -> str:
    if body.get('status') != 'incomplete':
        return ''
    details = body.get('incomplete_details') or {}
    return str(details.get('reason') or 'unknown')

def response_usage(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get('usage') or {}
    input_details = usage.get('input_tokens_details') or {}
    output_details = usage.get('output_tokens_details') or {}
    return {'input_tokens': int(usage.get('input_tokens') or 0), 'cache_write_tokens': int(input_details.get('cache_write_tokens') or 0), 'cached_tokens': int(input_details.get('cached_tokens') or 0), 'output_tokens': int(usage.get('output_tokens') or 0), 'reasoning_tokens': int(output_details.get('reasoning_tokens') or 0), 'total_tokens': int(usage.get('total_tokens') or 0)}


def relative_location(path):
    return Path(os.path.relpath(Path(path), Path.cwd())).as_posix()
