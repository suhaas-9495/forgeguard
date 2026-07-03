from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


@dataclass(slots=True)
class InferenceResult:
    """
    Standard inference response returned by ForgeGuard.

    This object is shared across the entire platform:
    - API
    - Evaluation Harness
    - Langfuse
    - MLflow
    - Dashboards
    - Risk Engine
    - Logging
    """
    # Generated Response

    response: str

    # Confidence Metrics

    sequence_confidence: float

    token_confidences: list[float]

    min_confidence: float

    max_confidence: float

    # Performance Metrics

    latency_ms: Optional[float] = None

    input_tokens: Optional[int] = None

    output_tokens: Optional[int] = None

    total_tokens: Optional[int] = None

    input_cost: Optional[float] = None

    output_cost: Optional[float] = None

    total_cost: Optional[float] = None

    # Model Information

    model_name: str = ""

    adapter_name: Optional[str] = None

    provider: Optional[str] = None

    # Generation Configuration

    temperature: Optional[float] = None

    max_new_tokens: Optional[int] = None

    do_sample: Optional[bool] = None

    # Metadata
    timestamp: str = ""

    request_id: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Convert the inference result into a dictionary.
        """
        return asdict(self)

    @classmethod
    def create_timestamp(cls) -> str:
        """
        Generate the current timestamp.
        """
        return datetime.now().astimezone().isoformat()