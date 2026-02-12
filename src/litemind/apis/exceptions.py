"""Custom exception classes for API-related errors in litemind."""


class APIError(Exception):
    """Base exception for all API-related errors in litemind.

    All API-specific exceptions should inherit from this class so that
    callers can catch a single type when handling API failures generically.
    """

    pass


class APINotAvailableError(APIError):
    """Raised when an API provider is unavailable or cannot be initialized.

    This typically occurs when the provider's service is unreachable, the
    required SDK is not installed, or the provided credentials are invalid.
    """

    pass


class FeatureNotAvailableError(APIError):
    """Raised when a requested feature is not supported by any available model.

    This is raised, for example, when the caller requests image generation
    but no configured model supports that capability.
    """

    pass
