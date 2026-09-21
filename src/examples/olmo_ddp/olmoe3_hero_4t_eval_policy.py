"""Explicit 4T waiver of inference parity, retaining structural/file integrity gates."""

import hashlib
import json
import shutil
from pathlib import Path

POLICY = "numerical_parity_skipped_by_user_20260916"


def portable_attention_backends(config):
    """Avoid training-only Flash4 construction during CPU weight export.

    Shared and split gains use the same attention reference backend; testing
    the *value* of the gain flag accidentally excluded shared-gain models.
    This does not modify the gain policy, tensor shapes, or exported weights.
    """
    if isinstance(config, dict):
        if "qk_norm_per_head_gains" in config:
            config["backend"] = "torch"
            config["use_flash"] = False
        for child in config.values():
            portable_attention_backends(child)
    elif isinstance(config, list):
        for child in config:
            portable_attention_backends(child)
    return config


def install_conversion(convert):
    """Keep frozen tensor mapping, but do not execute core/HF/cache logit comparisons."""
    import olmo_core.nn.hf.convert_checkpoint as exporter

    original = exporter.convert_checkpoint_to_hf

    def export(*args, **kwargs):
        kwargs["validate"] = False
        return original(*args, **kwargs)

    exporter.convert_checkpoint_to_hf = export
    reference_config = convert.reference_config
    convert.reference_config = lambda config: portable_attention_backends(reference_config(config))
    convert.qualify = structural_check


def structural_check(root, *, full, precise=False):
    """Read every serialized tensor, reject missing/unexpected shapes and nonfinite weights."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes
    from olmo_core.nn.hf.convert_checkpoint import load_config
    from olmo_core.nn.moe.v2.hf import configuration_olmo3moe, modeling_olmo3moe

    hf = root / "hf.partial"
    for module in (configuration_olmo3moe, modeling_olmo3moe):
        shutil.copy2(module.__file__, hf / Path(module.__file__).name)
    _register_olmo3moe_auto_classes()
    config = AutoConfig.from_pretrained(hf)
    saved = load_config(root / "olmo-core")
    assert config.vocab_size == saved["dataset"]["tokenizer"]["vocab_size"]
    gains = set()
    def collect(value):
        if isinstance(value, dict):
            if 'qk_norm_per_head_gains' in value:
                gains.add(bool(value['qk_norm_per_head_gains']))
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(saved['model'])
    assert gains == {bool(config.qk_norm_per_head_gains)}, (gains, config.qk_norm_per_head_gains)
    assert config.latent_moe_dim == 512
    assert config.hidden_size == 1024 and config.num_hidden_layers == 16
    AutoTokenizer.from_pretrained(hf)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    observed = {}
    for path in sorted(hf.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                assert key not in observed, key
                tensor = f.get_tensor(key)
                assert torch.isfinite(tensor).all().item(), key
                observed[key] = tuple(tensor.shape)
    assert observed == expected, {
        "missing": sorted(expected.keys() - observed.keys()),
        "extra": sorted(observed.keys() - expected.keys()),
        "wrong_shape": [k for k in expected.keys() & observed.keys() if expected[k] != observed[k]],
    }
    return dict(
        passed=True,
        checks_scope="serialized_tensor_structure_and_finiteness",
        numerical_qualification=POLICY,
        numerically_qualified=False,
        inference_profile="bf16-grouped-fla-pilot-v1",
        full=False,
        tensors=len(observed),
        source_vocabulary_verified=True,
        checks=["exact_hf_keys_and_shapes", "finite_weights", "tokenizer_vocab", "source_qknorm_gain_policy"],
    )


def validate_export(model, *, hash_weights=False):
    """CPU watchers inspect receipts; inference workers also hash every output file."""
    model = Path(model)
    from olmoe3_tokenizer_policy import require_correct_tokenizer
    require_correct_tokenizer(model)
    assert model.resolve() == model and model.is_dir()
    receipt_path = model / "_HERO_CONVERSION_SUCCESS.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["passed"] and receipt["numerical_qualification"] == POLICY
    assert receipt["numerically_qualified"] is False
    assert json.loads((model.parent / "conversion-success.json").read_text()) == receipt
    for name, digest in receipt["output_sha256"].items():
        path = model / name
        assert path.parent == model and path.is_file() and not path.is_symlink()
        if hash_weights or path.suffix != ".safetensors":
            with path.open("rb") as f:
                assert hashlib.file_digest(f, "sha256").hexdigest() == digest, name
    return hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def install_runtime(runtime):
    def validate(model, fast_pilot):
        assert fast_pilot and model in runtime.FAST_MODELS
        return validate_export(model, hash_weights=True)

    runtime.validate_source = validate
