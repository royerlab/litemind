class APIError(Exception):
    pass


class APINotAvailableError(APIError):
    pass


class FeatureNotAvailableError(APIError):
    pass
