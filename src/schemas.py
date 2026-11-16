from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field

class FinalLabel(str, Enum):
    """Legacy labels; PRIMARY_RCT_REPORT includes all eligible comparative RCT results reports."""
    PRIMARY_RCT_REPORT = 'PRIMARY_RCT_REPORT'
    NOT_PRIMARY_RCT_REPORT = 'NOT_PRIMARY_RCT_REPORT'
    UNCLEAR = 'UNCLEAR'

class RandomizedStatus(str, Enum):
    YES = 'YES'
    NO = 'NO'
    UNCLEAR = 'UNCLEAR'

class PublicationRole(str, Enum):
    PRIMARY_RESULTS = 'PRIMARY_RESULTS'
    PROTOCOL_OR_DESIGN = 'PROTOCOL_OR_DESIGN'
    SECONDARY_OR_FOLLOWUP = 'SECONDARY_OR_FOLLOWUP'
    REVIEW_OR_META_ANALYSIS = 'REVIEW_OR_META_ANALYSIS'
    NONRANDOMIZED_OR_OTHER = 'NONRANDOMIZED_OR_OTHER'
    UNCLEAR = 'UNCLEAR'

class RCTClassification(BaseModel):
    """Classification fields returned for each publication."""
    model_config = ConfigDict(extra='forbid')
    final_label: FinalLabel
    underlying_trial_randomized: RandomizedStatus
    publication_role: PublicationRole
    trial_name_or_acronym: str = Field(description='Trial name or acronym explicitly present in the input, or empty.')
    registration_ids: list[str] = Field(description='Registry identifiers explicitly present in the input, or empty.')
    brief_rationale: str = Field(description='Concise classification rationale.')
