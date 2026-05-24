"""
Provider-level exceptions shared across bounded contexts.

These represent failures coming from the external LLM provider
(timeouts, rate limits, transient errors, malformed output).
They are raised by infrastructure adapters and caught by the API
layer so that internal details never leak to clients.
"""


class LLMServiceError(Exception):
    """Base class for provider-related errors."""


class LLMTimeoutError(LLMServiceError):
    """Raised when the LLM call exceeds the configured timeout."""


class LLMRateLimitError(LLMServiceError):
    """Raised when the provider rejects requests due to rate limits."""


class LLMTemporaryError(LLMServiceError):
    """Raised for temporary provider/network failures after retries."""
