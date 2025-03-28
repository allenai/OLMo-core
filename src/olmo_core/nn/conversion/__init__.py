"""
Common logic for converting :mod:`olmo_core.nn` features to/from other formats (like Hugging Face).
"""

from .state_converter import StateConverter
from .state_mapping import StateMapping, StateMappingTemplate, TemplatePlaceholder

__all__ = [
    "StateConverter",
    "StateMapping",
    "StateMappingTemplate",
    "TemplatePlaceholder",
]
