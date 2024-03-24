class OLMoError(Exception):
    """
    Base exception for OLMo custom error types.
    """


class OLMoNetworkError(OLMoError):
    pass


class OLMoEnvironmentError(OLMoError):
    pass


class OLMoUserError(OLMoError):
    pass
