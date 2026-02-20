import builtins

import pytest

from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.visualize import to_dot

# -- Smoke tests: DOT generation for various factory configs -------------------


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        ("olmo2_1M", {"vocab_size": 32000}),
        ("olmo2_1B", {"vocab_size": 100352}),
        ("smallmoe", {"vocab_size": 50304}),
        ("small_hybrid_moe", {"vocab_size": 50304}),
        ("ngpt_271M", {"vocab_size": 50304}),
        ("gemma3_1B", {}),
        ("llama2_271M", {"vocab_size": 32000}),
    ],
)
@pytest.mark.parametrize("detail", ["overview", "block", "full"])
def test_to_dot_smoke(factory, kwargs, detail):
    """Verify that to_dot produces valid DOT output for various configs and detail levels."""
    config = getattr(TransformerConfig, factory)(**kwargs)
    dot = to_dot(config, detail=detail)
    assert dot.startswith("digraph")
    assert dot.rstrip().endswith("}")
    assert "Embedding" in dot


# -- Content tests: verify components per block type ---------------------------


def test_reordered_norm_block_flow():
    """Verify reordered-norm blocks produce expected DOT structure."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = to_dot(config, detail="block")
    assert "Attention" in dot
    assert "RmsNorm" in dot or "Rms" in dot


def test_moe_block_shows_moe():
    """Verify MoE blocks include MoE-related content."""
    config = TransformerConfig.smallmoe(vocab_size=50304)
    dot = to_dot(config, detail="block")
    assert "MoE" in dot


def test_hybrid_moe_shows_both_paths():
    """Verify hybrid MoE blocks show both dense FFN and sparse MoE paths."""
    config = TransformerConfig.small_hybrid_moe(vocab_size=50304)
    dot = to_dot(config, detail="block")
    assert "Dense path" in dot
    assert "Sparse path" in dot


def test_normalized_block_shows_lerp():
    """Verify nGPT normalized blocks show Lerp operations."""
    config = TransformerConfig.ngpt_271M(vocab_size=50304)
    dot = to_dot(config, detail="block")
    assert "Lerp" in dot
    assert "L2 Normalize" in dot


def test_gemma3_block_pattern():
    """Verify Gemma3 shows both local and global block types."""
    config = TransformerConfig.gemma3_1B()
    dot = to_dot(config, detail="block")
    assert "local" in dot or "Local" in dot
    assert "global" in dot or "Global" in dot


def test_peri_norm_block():
    """Verify peri-norm blocks show pre and post norms."""
    config = TransformerConfig.gemma3_1B()
    dot = to_dot(config, detail="block")
    # Gemma3 uses peri_norm blocks, which should show Pre- and Post- norms.
    assert "Pre-" in dot
    assert "Post-" in dot


# -- Full detail tests ---------------------------------------------------------


def test_full_detail_shows_qkv():
    """Verify full detail shows Q/K/V projections."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = to_dot(config, detail="full")
    assert "w_q" in dot
    assert "w_k" in dot
    assert "w_v" in dot


def test_full_detail_shows_ffn_internals():
    """Verify full detail shows FFN gate structure."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = to_dot(config, detail="full")
    assert "w1" in dot
    assert "w2" in dot
    assert "w3" in dot
    assert "silu" in dot or "SiLU" in dot


def test_full_detail_shows_rope():
    """Verify full detail shows RoPE when configured."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = to_dot(config, detail="full")
    assert "RoPE" in dot


def test_full_detail_shows_sdpa():
    """Verify full detail shows SDPA node."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = to_dot(config, detail="full")
    assert "SDPA" in dot


def test_full_detail_shows_w_out():
    """Verify full detail shows output projection."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = to_dot(config, detail="full")
    assert "w_out" in dot


def test_full_detail_moe_shows_router():
    """Verify full detail MoE shows router and experts."""
    config = TransformerConfig.smallmoe(vocab_size=50304)
    dot = to_dot(config, detail="full")
    assert "Router" in dot
    assert "Experts" in dot


# -- Layer collapsing ----------------------------------------------------------


def test_layer_collapsing():
    """Verify that identical layers are collapsed into a single group."""
    config = TransformerConfig.olmo2_1B(vocab_size=100352)
    dot = to_dot(config, detail="block")
    # 16 identical layers should produce just 1 block subgraph, not 16.
    assert dot.count("cluster_block") < config.n_layers


# -- Dimension annotations ----------------------------------------------------


def test_dimension_annotations():
    """Verify key dimensions appear in the DOT output."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = to_dot(config, detail="full")
    assert str(config.d_model) in dot
    assert "32,000" in dot or "32000" in dot  # vocab_size


# -- Custom title --------------------------------------------------------------


def test_custom_title():
    """Verify custom title appears in DOT output."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = to_dot(config, title="My Custom Model")
    assert "My Custom Model" in dot


def test_default_title_includes_dimensions():
    """Verify default title includes model dimensions."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = to_dot(config)
    assert f"d_model={config.d_model}" in dot
    assert f"n_layers={config.n_layers}" in dot


# -- LM Head -------------------------------------------------------------------


def test_lm_head_shows_norm():
    """Verify LM head norm is shown when present."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = to_dot(config, detail="block")
    if config.lm_head.layer_norm is not None:
        assert "LM Head" in dot
        assert "Linear" in dot


def test_lm_head_projection():
    """Verify LM head projection shows dimensions."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = to_dot(config, detail="block")
    assert "Linear" in dot
    assert "32,000" in dot or "32000" in dot


# -- Convenience method --------------------------------------------------------


def test_visualize_method():
    """Verify the convenience method on TransformerConfig works."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = config.visualize()
    assert dot.startswith("digraph")


def test_visualize_method_passes_args():
    """Verify the convenience method passes through detail and title."""
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    dot = config.visualize(detail="overview", title="Test Title")
    assert "Test Title" in dot


# -- Embedding norm and scale --------------------------------------------------


def test_embedding_norm_shown():
    """Verify embedding norm appears when configured."""
    config = TransformerConfig.gemma3_1B()
    dot = to_dot(config, detail="block")
    # Gemma3 uses embed_scale, which should appear in the output.
    if config.embed_scale is not None:
        assert str(config.embed_scale) in dot or "\\u00d7" in dot


# -- Render tests --------------------------------------------------------------


def test_render_raises_without_graphviz(monkeypatch):
    """Test that render raises ImportError when graphviz is not installed."""
    from olmo_core.nn.transformer.visualize import render

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "graphviz":
            raise ImportError("No module named 'graphviz'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    with pytest.raises(ImportError, match="graphviz"):
        render(config, "/tmp/out")


def test_render_svg(tmp_path):
    """Test rendering to SVG file (skipped if graphviz not installed)."""
    try:
        import graphviz  # noqa: F401
    except ImportError:
        pytest.skip("graphviz package not installed")
    del graphviz

    from olmo_core.nn.transformer.visualize import render

    config = TransformerConfig.olmo2_1M(vocab_size=32000)
    output = render(config, str(tmp_path / "model"), format="svg")
    assert output.endswith(".svg")
