import os

import pytest
from gantry.api import GitRepoState

from olmo_core.launch.beaker import (
    BeakerLaunchConfig,
    OLMoCoreBeakerImage,
    get_beaker_client,
)


@pytest.mark.parametrize("min_runtime", [None, "8h"])
def test_min_runtime_preserves_preemptible_default(monkeypatch, min_runtime):
    config = BeakerLaunchConfig(
        name="test",
        cmd=["echo", "ok"],
        allow_dirty=True,
        min_runtime=min_runtime,
        git=GitRepoState("allenai/OLMo-core", "https://github.com/allenai/OLMo-core", "main"),
    )
    monkeypatch.setattr(config, "_resolve_beaker_image", lambda: "image")

    recipe, _ = config._build_recipe(object(), follow=False, slack_notifications=False)

    assert recipe.min_runtime == min_runtime
    assert recipe.preemptible == (config.preemptible if min_runtime is None else None)


def test_get_beaker_client_caching():
    with get_beaker_client(workspace="ai2/OLMo-core") as beaker1:
        # Should get the same client since we're requesting the same workspace.
        with get_beaker_client(workspace="ai2/OLMo-core") as beaker2:
            assert beaker1 is beaker2
        # Should get the same client since we'll default to the last workspace requested.
        with get_beaker_client(workspace=None) as beaker2:
            assert beaker1 is beaker2
        # Should get different client this time we requested a different workspace.
        with get_beaker_client(workspace="ai2/gantry-testing") as beaker2:
            assert beaker1 is not beaker2

    with get_beaker_client(workspace=None) as beaker1:
        # Should get the same client, but now its default workspace is set.
        with get_beaker_client(workspace="ai2/OLMo-core") as beaker2:
            assert beaker1 is beaker2
            assert beaker1.config.default_workspace == "ai2/OLMo-core"
        # Check same thing again.
        with get_beaker_client(workspace="ai2/OLMo-core") as beaker2:
            assert beaker1 is beaker2
            assert beaker1.config.default_workspace == "ai2/OLMo-core"
        # Should get different client this time we requested a different workspace.
        with get_beaker_client(workspace="ai2/gantry-testing") as beaker2:
            assert beaker1 is not beaker2


@pytest.fixture(scope="session")
def beaker():
    with get_beaker_client(workspace="ai2/OLMo-core") as beaker:
        yield beaker


@pytest.mark.skipif(
    os.environ.get("BEAKER_TOKEN", "") == "", reason="Missing 'BEAKER_TOKEN' env var"
)
@pytest.mark.parametrize("image", list(OLMoCoreBeakerImage))
def test_official_images_exist(image, beaker):
    beaker.image.get(image)
