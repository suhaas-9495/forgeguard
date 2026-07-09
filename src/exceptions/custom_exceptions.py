"""
ForgeGuard Custom Exceptions
Production-grade exception hierarchy.
"""


class ForgeGuardError(Exception):
    """Base exception for ForgeGuard."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ModelLoadError(ForgeGuardError):
    """Raised when the base model cannot be loaded."""


class AdapterLoadError(ForgeGuardError):
    """Raised when a LoRA adapter cannot be loaded."""


class TokenizationError(ForgeGuardError):
    """Raised when tokenization fails."""


class InferenceError(ForgeGuardError):
    """Raised when text generation fails."""


class ConfidenceError(ForgeGuardError):
    """Raised when confidence estimation fails."""


class CostEstimationError(ForgeGuardError):
    """Raised when token cost estimation fails."""