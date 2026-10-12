"""
Training-side glue between :mod:`ctc.format` and olmo-core's trainer.

Only what training needs to *record*. No model code and no launcher configuration live here --
those belong to olmo-core and to the launch layer respectively. Importing this module requires
olmo-core; the rest of ``ctc`` does not.
"""

from .fingerprint import FormatFingerprintCallback

__all__ = ["FormatFingerprintCallback"]
