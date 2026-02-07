class APIError(Exception):
    """Base exception for all API-related errors in litemind."""

    pass


class APINotAvailableError(APIError):
    """Raised when an API provider is unavailable or cannot be initialized."""

    pass


class FeatureNotAvailableError(APIError):
    """Raised when a requested feature is not supported by any available model."""

    pass
