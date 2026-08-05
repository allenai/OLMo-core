import os

import pytest

from gantry.api import GitRepoState

from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage, get_beaker_client


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


def test_build_recipe_propagates_github_token_secret(monkeypatch):
    monkeypatch.setattr(BeakerLaunchConfig, "_resolve_beaker_image", lambda self: "image-id")
    config = BeakerLaunchConfig(
        name="test",
        cmd=["true"],
        git=GitRepoState(
            repo="OLMo-core",
            repo_url="https://github.com/allenai/OLMo-core.git",
            ref="deadbeef",
        ),
        gh_token_secret="jacobm_GITHUB_TOKEN",
    )

    recipe, _ = config._build_recipe(object(), follow=False)

    assert recipe.gh_token_secret == "jacobm_GITHUB_TOKEN"


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
