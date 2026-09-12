import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_import_check(code: str):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(Path(__file__).parents[3]) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )


@pytest.mark.parametrize(
    "module",
    [
        "olmo_core.data.multimodal",
        "olmo_core.data.multimodal.alignment",
        "olmo_core.internal.vision_alignment",
    ],
)
def test_clean_import_does_not_load_legacy_registries(module):
    _run_import_check(
        f"""
import importlib
import sys

importlib.import_module({module!r})
legacy = (
    'native_text_replay',
    'vision_alignment_sources',
    'vision_alignment_perception_sources',
    'vision_alignment_joint_sources',
    'vision_alignment_perception_provenance',
    'vision_alignment_joint_provenance',
)
loaded = [name for name in legacy if 'olmo_core.data.multimodal.' + name in sys.modules]
assert not loaded, loaded
"""
    )


@pytest.mark.parametrize(
    "module,names",
    [
        (
            "native_text_replay",
            [
                "NativeTextReplayDataset",
                "NativeTextReplayDatasetConfig",
                "NativeTextReplayManifest",
                "NativeTextReplaySource",
                "NativeTextReplayVerificationReceipt",
            ],
        ),
        (
            "vision_alignment_perception_sources",
            [
                "VisionAlignmentPerceptionSourceSpec",
                "build_vision_alignment_perception_dataset",
                "build_vision_alignment_perception_dataset_config",
            ],
        ),
    ],
)
def test_legacy_exports_keep_original_identity_and_are_cached(module, names):
    _run_import_check(
        f"""
import importlib
import sys
import olmo_core.data.multimodal as package

module_name = 'olmo_core.data.multimodal.' + {module!r}
assert module_name not in sys.modules
for name in {names!r}:
    assert name in package.__all__
    assert name not in package.__dict__
    namespace = {{}}
    exec('from olmo_core.data.multimodal import ' + name, namespace)
    original = getattr(importlib.import_module(module_name), name)
    assert namespace[name] is original
    assert package.__dict__[name] is original
    assert getattr(package, name) is original
try:
    package.not_a_multimodal_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export did not raise AttributeError')
"""
    )


def test_legacy_star_import_and_replay_config_roundtrip_are_compatible():
    _run_import_check(
        """
import json
from olmo_core.config import Config
import olmo_core.data.multimodal as package

namespace = {}
exec('from olmo_core.data.multimodal import *', namespace)
assert all(name in namespace for name in package.__all__)
cls = namespace['NativeTextReplayDatasetConfig']
config = cls(manifest_path='/unused/manifest.json')
saved = json.loads(json.dumps(config.as_config_dict()))
assert saved['_CLASS_'] == 'olmo_core.data.multimodal.native_text_replay.NativeTextReplayDatasetConfig'
restored = Config.from_dict(saved)
assert type(restored) is cls
assert restored == config
"""
    )
