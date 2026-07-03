from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class ConfidenceResult:
    """
    Stores confidence statistics for one generated response.
    """

    sequence_confidence: float
    token_confidences: list[float]
    min_confidence: float
    max_confidence: float
    average_confidence: float
    confidence_variance: float


class ConfidenceEstimator:
    """
    Computes confidence statistics from generation logits.
    """

    def compute(
        self,
        scores: tuple[torch.Tensor, ...],
    ) -> ConfidenceResult:

        token_confidences: list[float] = []

        for logits in scores:

            probabilities = F.softmax(
                logits,
                dim=-1,
            )

            confidence = (
                probabilities.max(dim=-1)
                .values
                .mean()
                .item()
            )

            token_confidences.append(confidence)

        if not token_confidences:

            return ConfidenceResult(
                sequence_confidence=0.0,
                token_confidences=[],
                min_confidence=0.0,
                max_confidence=0.0,
                average_confidence=0.0,
                confidence_variance=0.0,
            )

        sequence_confidence = (
            sum(token_confidences)
            / len(token_confidences)
        )

        variance = (
            sum(
                (c - sequence_confidence) ** 2
                for c in token_confidences
            )
            / len(token_confidences)
        )

        return ConfidenceResult(
            sequence_confidence=round(sequence_confidence, 4),
            token_confidences=[
                round(c, 4)
                for c in token_confidences
            ],
            min_confidence=round(
                min(token_confidences),
                4,
            ),
            max_confidence=round(
                max(token_confidences),
                4,
            ),
            average_confidence=round(
                sequence_confidence,
                4,
            ),
            confidence_variance=round(
                variance,
                6,
            ),
        )