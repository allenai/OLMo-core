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


class OLMoCheckpointError(OLMoError):
    pass


class OLMoConfigurationError(OLMoError):
    pass


class OLMoCLIError(OLMoError):
    pass


class OLMoThreadError(OLMoError):
    pass


class BeakerExperimentFailedError(OLMoError):
    pass
