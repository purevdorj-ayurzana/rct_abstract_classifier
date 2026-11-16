from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator

class PrimaryEndpointResult(str, Enum):
    POSITIVE = 'POSITIVE'
    NEGATIVE = 'NEGATIVE'
    UNCLEAR = 'UNCLEAR'

class TrialFramework(str, Enum):
    SUPERIORITY = 'SUPERIORITY'
    NONINFERIORITY = 'NONINFERIORITY'
    EQUIVALENCE = 'EQUIVALENCE'
    OTHER = 'OTHER'
    UNCLEAR = 'UNCLEAR'

class UnclearReason(str, Enum):
    NOT_APPLICABLE = 'NOT_APPLICABLE'
    PRIMARY_ENDPOINT_NOT_EXPLICIT = 'PRIMARY_ENDPOINT_NOT_EXPLICIT'
    PRIMARY_ENDPOINT_RESULT_NOT_REPORTED = 'PRIMARY_ENDPOINT_RESULT_NOT_REPORTED'
    SUCCESS_RULE_NOT_DETERMINABLE = 'SUCCESS_RULE_NOT_DETERMINABLE'
    MULTIPLE_PRIMARY_ENDPOINTS_AMBIGUOUS = 'MULTIPLE_PRIMARY_ENDPOINTS_AMBIGUOUS'
    OTHER = 'OTHER'

class PrimaryEndpointAssessment(BaseModel):
    """Strict model-output contract for the downstream endpoint task."""
    model_config = ConfigDict(extra='forbid')
    primary_endpoint_result: PrimaryEndpointResult
    primary_endpoint_description: str = Field(description='Brief description of the endpoint explicitly identified as primary, or an empty string when it is not identifiable.')
    trial_framework: TrialFramework
    unclear_reason: UnclearReason
    brief_rationale: str = Field(min_length=1, description='Concise rationale limited to the explicitly identified primary endpoint.')

    @model_validator(mode='after')
    def validate_internal_consistency(self) -> 'PrimaryEndpointAssessment':
        decisive = self.primary_endpoint_result in {PrimaryEndpointResult.POSITIVE, PrimaryEndpointResult.NEGATIVE}
        if decisive and self.unclear_reason != UnclearReason.NOT_APPLICABLE:
            raise ValueError('POSITIVE and NEGATIVE assessments require unclear_reason NOT_APPLICABLE.')
        if decisive and (not self.primary_endpoint_description.strip()):
            raise ValueError('POSITIVE and NEGATIVE assessments require an explicitly identified primary endpoint.')
        if self.primary_endpoint_result == PrimaryEndpointResult.UNCLEAR and self.unclear_reason == UnclearReason.NOT_APPLICABLE:
            raise ValueError('UNCLEAR assessments require a specific unclear_reason.')
        return self
