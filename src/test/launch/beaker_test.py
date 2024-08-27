import os

import pytest
from beaker import Beaker

from olmo_core.launch.beaker import OLMoCoreBeakerImage


@pytest.mark.skipif("BEAKER_TOKEN" not in os.environ, reason="Missing 'BEAKER_TOKEN' env var")
@pytest.mark.parametrize("image", list(OLMoCoreBeakerImage))
def test_official_images_exist(image):
    beaker = Beaker.from_env(default_workspace="ai2/OLMo-core")
    beaker.image.get(image)
