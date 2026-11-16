from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
from common import canonical_json_hash, normalize_pmid, normalize_whitespace, sha256_text
WORKFLOW_MODE = 'primary_endpoint_single_model_arm'
REQUIRED_PUBLICATION_COLUMNS = {'pmid', 'title', 'abstract'}
VALID_RESULTS = {'POSITIVE', 'NEGATIVE', 'UNCLEAR'}
VALID_FRAMEWORKS = {'SUPERIORITY', 'NONINFERIORITY', 'EQUIVALENCE', 'OTHER', 'UNCLEAR'}
VALID_UNCLEAR_REASONS = {'NOT_APPLICABLE', 'PRIMARY_ENDPOINT_NOT_EXPLICIT', 'PRIMARY_ENDPOINT_RESULT_NOT_REPORTED', 'SUCCESS_RULE_NOT_DETERMINABLE', 'MULTIPLE_PRIMARY_ENDPOINTS_AMBIGUOUS', 'OTHER'}

def load_endpoint_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config.get('workflow_mode') != WORKFLOW_MODE:
        raise ValueError('Use the single-model primary-endpoint workflow.')
    if config.get('source_selector') != {'upstream_arm': 'sol', 'upstream_label': 'PRIMARY_RCT_REPORT'}:
        raise ValueError('Endpoint input must contain all sol-positive RCT reports.')
    if list(config.get('model_arms', {})) != ['sol']:
        raise ValueError('The endpoint workflow requires exactly sol.')
    settings = config['model_arms']['sol']
    if settings.get('model') != 'gpt-5.6-sol' or settings.get('reasoning_effort') != 'medium':
        raise ValueError('The study uses gpt-5.6-sol with medium reasoning.')
    if int(settings.get('chunk_size', 0)) <= 0 or int(config.get('max_technical_retries', -1)) < 0:
        raise ValueError('Invalid chunk size or retry limit.')
    return config

def configured_endpoint_arms(config: dict[str, Any]) -> list[str]:
    return [str(arm) for arm in config['model_arms']]

def endpoint_arm_settings(config: dict[str, Any], arm: str) -> tuple[str, str, int]:
    if arm not in config['model_arms']:
        raise ValueError(f'Unknown endpoint arm {arm!r}; expected {configured_endpoint_arms(config)}.')
    settings = config['model_arms'][arm]
    return (str(settings['model']), str(settings['reasoning_effort']), int(settings['chunk_size']))

def endpoint_json_schema():
    return json.loads((Path(__file__).resolve().parents[1] / 'schemas/primary_endpoint.json').read_text(encoding='utf-8'))

def build_endpoint_user_input(row: dict[str, Any]) -> str:
    fields = [('PMID', normalize_pmid(row.get('pmid', ''))), ('TITLE', normalize_whitespace(row.get('title', ''))), ('ABSTRACT', normalize_whitespace(row.get('abstract', ''))), ('JOURNAL', normalize_whitespace(row.get('journal', ''))), ('PUBLICATION DATE', normalize_whitespace(row.get('publication_date', row.get('publication_year', '')))), ('PUBMED PUBLICATION TYPES', normalize_whitespace(row.get('publication_types', ''))), ('DOI', normalize_whitespace(row.get('doi', '')))]
    return '\n\n'.join((f'{name}:\n{value or '[not supplied]'}' for name, value in fields))

def make_endpoint_input_hash(row: dict[str, Any]) -> str:
    return sha256_text(build_endpoint_user_input(row))

def safe_endpoint_custom_id(pmid: str, *, arm: str, chunk_number: int, attempt_number: int) -> str:
    safe_arm = re.sub('[^A-Za-z0-9_-]', '-', arm)
    safe_pmid = re.sub('[^A-Za-z0-9_-]', '-', normalize_pmid(pmid))
    return f'endpoint-{safe_arm}-c{chunk_number:04d}-a{attempt_number:02d}-pmid-{safe_pmid}'

def endpoint_request_hash(body: dict[str, Any]) -> str:
    return canonical_json_hash(body)

def endpoint_response_usage(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get('usage') or {}
    input_details = usage.get('input_tokens_details') or {}
    output_details = usage.get('output_tokens_details') or {}
    return {'input_tokens': int(usage.get('input_tokens') or 0), 'cache_write_tokens': int(input_details.get('cache_write_tokens') or 0), 'cached_tokens': int(input_details.get('cached_tokens') or 0), 'output_tokens': int(usage.get('output_tokens') or 0), 'reasoning_tokens': int(output_details.get('reasoning_tokens') or 0), 'total_tokens': int(usage.get('total_tokens') or 0)}
