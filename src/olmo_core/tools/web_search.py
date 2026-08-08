import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from ..config import Config, Registrable
from ..exceptions import OLMoConfigurationError
from .base import Tool, ToolConfig

__all__ = [
    "SearchResult",
    "WebSearchBackend",
    "WebSearchBackendConfig",
    "DdgsBackend",
    "DdgsBackendConfig",
    "TavilyBackend",
    "TavilyBackendConfig",
    "SerperBackend",
    "SerperBackendConfig",
    "StaticBackend",
    "StaticBackendConfig",
    "WebSearchTool",
    "WebSearchToolConfig",
    "has_ddgs",
]


@dataclass
class SearchResult:
    """
    A single search hit.
    """

    title: str
    url: str
    snippet: str


class WebSearchBackend(metaclass=ABCMeta):
    """
    Base class for the search provider behind :class:`WebSearchTool`.
    """

    @abstractmethod
    def search(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Run a search.

        :param query: The search query.
        :param max_results: The most results to return.

        :returns: The results, best first.
        """
        raise NotImplementedError


@dataclass
class WebSearchBackendConfig(Config, Registrable, metaclass=ABCMeta):
    """
    Base class for :class:`WebSearchBackend` configs.
    """

    @abstractmethod
    def build(self) -> WebSearchBackend:
        """
        Build the backend.

        :returns: The backend instance.
        """
        raise NotImplementedError


def has_ddgs() -> bool:
    """
    Check whether the ``ddgs`` package is installed.

    :returns: Whether the default search backend can be used.
    """
    try:
        import ddgs  # type: ignore # noqa: F401

        return True
    except ImportError:
        return False


class DdgsBackend(WebSearchBackend):
    """
    A keyless backend built on `ddgs <https://github.com/deedy5/ddgs>`_.

    :param timeout: Seconds to wait for the provider before giving up.
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'ddgs' package is required for the default web search backend. "
                "Install it with: pip install 'ai2-olmo-core[tools]'"
            ) from e

        with DDGS(timeout=self.timeout) as client:
            hits = client.text(query, max_results=max_results)

        return [
            SearchResult(
                title=hit.get("title", ""),
                url=hit.get("href", "") or hit.get("url", ""),
                snippet=hit.get("body", ""),
            )
            for hit in hits
        ]


@WebSearchBackendConfig.register("ddgs")
@dataclass
class DdgsBackendConfig(WebSearchBackendConfig):
    """
    Configuration for building a :class:`DdgsBackend`.
    """

    timeout: float = 10.0

    def build(self) -> DdgsBackend:
        return DdgsBackend(timeout=self.timeout)


class TavilyBackend(WebSearchBackend):
    """
    A backend built on the `Tavily <https://tavily.com>`_ search API.

    :param api_key: The Tavily API key.
    :param timeout: Seconds to wait for the provider before giving up.
    """

    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout: float = 10.0):
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        response = requests.post(
            self.ENDPOINT,
            json={"api_key": self.api_key, "query": query, "max_results": max_results},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return [
            SearchResult(
                title=hit.get("title", ""),
                url=hit.get("url", ""),
                snippet=hit.get("content", ""),
            )
            for hit in response.json().get("results", [])
        ]


@WebSearchBackendConfig.register("tavily")
@dataclass
class TavilyBackendConfig(WebSearchBackendConfig):
    """
    Configuration for building a :class:`TavilyBackend`.
    """

    api_key_env_var: str = "TAVILY_API_KEY"
    timeout: float = 10.0

    def build(self) -> TavilyBackend:
        return TavilyBackend(
            api_key=_require_api_key(self.api_key_env_var, "tavily"), timeout=self.timeout
        )


class SerperBackend(WebSearchBackend):
    """
    A backend built on the `Serper <https://serper.dev>`_ Google search API.

    :param api_key: The Serper API key.
    :param timeout: Seconds to wait for the provider before giving up.
    """

    ENDPOINT = "https://google.serper.dev/search"

    def __init__(self, api_key: str, timeout: float = 10.0):
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        response = requests.post(
            self.ENDPOINT,
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return [
            SearchResult(
                title=hit.get("title", ""),
                url=hit.get("link", ""),
                snippet=hit.get("snippet", ""),
            )
            for hit in response.json().get("organic", [])[:max_results]
        ]


@WebSearchBackendConfig.register("serper")
@dataclass
class SerperBackendConfig(WebSearchBackendConfig):
    """
    Configuration for building a :class:`SerperBackend`.
    """

    api_key_env_var: str = "SERPER_API_KEY"
    timeout: float = 10.0

    def build(self) -> SerperBackend:
        return SerperBackend(
            api_key=_require_api_key(self.api_key_env_var, "serper"), timeout=self.timeout
        )


class StaticBackend(WebSearchBackend):
    """
    A backend that answers from a fixed table, for tests and offline use.

    :param results: Results keyed by query. Queries are matched case-insensitively.
    :param default: What to answer for a query that isn't in the table.
    """

    def __init__(
        self,
        results: Optional[Dict[str, List[SearchResult]]] = None,
        default: Optional[List[SearchResult]] = None,
    ):
        self.results = {query.lower(): hits for query, hits in (results or {}).items()}
        self.default = default or []

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        return self.results.get(query.lower(), self.default)[:max_results]


@WebSearchBackendConfig.register("static")
@dataclass
class StaticBackendConfig(WebSearchBackendConfig):
    """
    Configuration for building a :class:`StaticBackend`.
    """

    results: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)

    def build(self) -> StaticBackend:
        return StaticBackend(
            {query: [SearchResult(**hit) for hit in hits] for query, hits in self.results.items()}
        )


def _require_api_key(env_var: str, provider: str) -> str:
    api_key = os.environ.get(env_var)
    if not api_key:
        raise OLMoConfigurationError(
            f"the '{provider}' web search backend needs an API key, but the environment "
            f"variable '{env_var}' is not set"
        )
    return api_key


class WebSearchTool(Tool):
    """
    A tool for searching the web.

    :param backend: The search provider.
    :param max_results: The most results to return for a query.
    """

    def __init__(self, backend: WebSearchBackend, max_results: int = 5):
        self.backend = backend
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web and return a ranked list of results, each with a title, URL and "
            "short snippet. Use this for facts that may be recent or that you are unsure of."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"How many results to return. Defaults to {self.max_results}.",
                },
            },
            "required": ["query"],
        }

    def call(self, query: str, max_results: Optional[int] = None) -> str:  # type: ignore[override]
        limit = max(1, min(max_results or self.max_results, self.max_results))
        results = self.backend.search(query, limit)
        if not results:
            return f"No results found for '{query}'."
        return "\n\n".join(
            f"{i}. {result.title}\n{result.url}\n{result.snippet}"
            for i, result in enumerate(results, start=1)
        )


@ToolConfig.register("web_search")
@dataclass
class WebSearchToolConfig(ToolConfig):
    """
    Configuration for building a :class:`WebSearchTool`.
    """

    backend: WebSearchBackendConfig = field(default_factory=DdgsBackendConfig)
    max_results: int = 5

    def build(self) -> WebSearchTool:
        return WebSearchTool(backend=self.backend.build(), max_results=self.max_results)
