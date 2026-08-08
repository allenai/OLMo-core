import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.tools import (
    SearchResult,
    SerperBackendConfig,
    StaticBackend,
    StaticBackendConfig,
    TavilyBackendConfig,
    WebSearchTool,
    WebSearchToolConfig,
)

RESULTS = {
    "olmo": [
        SearchResult("OLMo", "https://allenai.org/olmo", "An open language model."),
        SearchResult("OLMo-core", "https://github.com/allenai/OLMo-core", "Training library."),
    ]
}


def test_web_search_formats_results():
    tool = WebSearchTool(StaticBackend(RESULTS))
    output = tool.call(query="OLMo")
    assert "1. OLMo" in output
    assert "https://allenai.org/olmo" in output
    assert "2. OLMo-core" in output


def test_web_search_with_no_results():
    tool = WebSearchTool(StaticBackend())
    assert tool.call(query="nothing") == "No results found for 'nothing'."


def test_web_search_respects_max_results():
    tool = WebSearchTool(StaticBackend(RESULTS), max_results=1)
    assert "2. OLMo-core" not in tool.call(query="OLMo")


def test_web_search_caps_the_models_request():
    """A model asking for more than the configured maximum should not get it."""
    tool = WebSearchTool(StaticBackend(RESULTS), max_results=1)
    assert "2. OLMo-core" not in tool.call(query="OLMo", max_results=100)


def test_static_backend_config_builds_backend():
    config = StaticBackendConfig(
        results={"olmo": [{"title": "OLMo", "url": "https://olmo", "snippet": "..."}]}
    )
    assert config.build().search("olmo", 5)[0].title == "OLMo"


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(TavilyBackendConfig(api_key_env_var="NOT_SET_TAVILY"), id="tavily"),
        pytest.param(SerperBackendConfig(api_key_env_var="NOT_SET_SERPER"), id="serper"),
    ],
)
def test_keyed_backends_fail_without_an_api_key(config, monkeypatch):
    monkeypatch.delenv(config.api_key_env_var, raising=False)
    with pytest.raises(OLMoConfigurationError, match="API key"):
        config.build()


def test_keyed_backend_reads_the_api_key(monkeypatch):
    monkeypatch.setenv("NOT_SET_TAVILY", "secret")
    assert TavilyBackendConfig(api_key_env_var="NOT_SET_TAVILY").build().api_key == "secret"


def test_web_search_tool_config_resolves_backend_from_a_dict():
    config = WebSearchToolConfig.from_dict({"backend": {"type": "static"}, "max_results": 3})
    tool = config.build()
    assert isinstance(tool.backend, StaticBackend)
    assert tool.max_results == 3
