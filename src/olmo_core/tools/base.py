from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..config import Config, Registrable

__all__ = ["ToolCall", "ToolResult", "Tool", "ToolConfig"]


@dataclass
class ToolCall:
    """
    A single tool call parsed out of a model completion.
    """

    name: str
    """
    The name of the tool being called.
    """

    arguments: Dict[str, Any] = field(default_factory=dict)
    """
    The keyword arguments the model passed to the tool.
    """


@dataclass
class ToolResult:
    """
    The outcome of executing a :class:`ToolCall`.
    """

    name: str
    """
    The name of the tool that was called.
    """

    content: str
    """
    The text handed back to the model.
    """

    error: Optional[str] = None
    """
    The failure reason, or ``None`` if the call succeeded.
    """

    @property
    def ok(self) -> bool:
        """
        Whether the call succeeded.
        """
        return self.error is None


class Tool(metaclass=ABCMeta):
    """
    Base class for tools that a language model can call.

    Subclasses are built from a corresponding :class:`ToolConfig`. The :data:`name`,
    :data:`description` and :data:`parameters` of a tool are serialized into the model's
    prompt, so they are part of the model-facing interface rather than incidental metadata.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name the model uses to call this tool.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """
        A description of what the tool does, shown to the model.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        A JSON-schema object describing the arguments the tool accepts.
        """
        raise NotImplementedError

    @abstractmethod
    def call(self, **kwargs) -> str:
        """
        Run the tool.

        Exceptions raised here are caught by :class:`~olmo_core.tools.registry.ToolRegistry`
        and turned into an error result, so implementations are free to raise on bad input.

        :param kwargs: The arguments parsed from the model's tool call.

        :returns: The result to hand back to the model.
        """
        raise NotImplementedError

    def json_schema(self) -> Dict[str, Any]:
        """
        Build the OpenAI-style function specification for this tool.

        :returns: The spec to include in the model's prompt.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolConfig(Config, Registrable, metaclass=ABCMeta):
    """
    Base class for :class:`Tool` configs.
    """

    @abstractmethod
    def build(self) -> Tool:
        """
        Build the tool.

        :returns: The tool instance.
        """
        raise NotImplementedError
