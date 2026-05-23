"""Custom exceptions used by the AI service layer."""


class LLMServiceError(Exception):
    """Base class for provider-related errors."""


class LLMTimeoutError(LLMServiceError):
    """Raised when the LLM call exceeds the configured timeout."""


class LLMRateLimitError(LLMServiceError):
    """Raised when the provider rejects requests due to rate limits."""


class LLMTemporaryError(LLMServiceError):
    """Raised for temporary provider/network failures after retries."""
