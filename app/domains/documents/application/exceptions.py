"""
Domain-meaningful errors raised by the documents use cases.

They are surfaced explicitly so the API layer can map them to HTTP without
leaking infrastructure details.
"""


class DocumentValidationError(Exception):
    """Raised when an uploaded document fails domain-level validation."""


class EmptyVectorStoreError(Exception):
    """Raised when a search is attempted before any document has been uploaded."""
