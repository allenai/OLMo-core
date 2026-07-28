"""
Tests for the coverage analysis in run_ci_tests_on_images.py.
"""

import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    "run_ci_tests_on_images", "src/scripts/beaker/run_ci_tests_on_images.py"
)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod  # dataclasses resolve globals via sys.modules[__module__]
spec.loader.exec_module(mod)

Outcome = mod.Outcome


def test_required_gpu_count():
    assert mod._required_gpu_count("requires four GPUs") == 4
    assert mod._required_gpu_count("Requires 4 GPUs") == 4
    assert mod._required_gpu_count("needs 8 gpus for this") == 8
    # >=2 multi-gpu (runs on a 2-GPU node) — no explicit count, not an exclusion.
    assert mod._required_gpu_count("Requires multiple GPUs") is None
    assert mod._required_gpu_count("Requires a GPU") is None
    assert mod._required_gpu_count("") is None


def test_analyze_coverage():
    results = {
        # flash_4 test runs only on the fa4 image; the cu128 image skips it — still covered.
        "cu128": {
            "t_flash4::x": Outcome("skipped", "Requires flash-attn 4"),
            "t_flash3::x": Outcome("passed"),
            "t_4gpu::x": Outcome("skipped", "requires four GPUs"),
            "t_broken::x": Outcome("failed"),
        },
        "cu130-fa4": {
            "t_flash4::x": Outcome("passed"),
            "t_flash3::x": Outcome("skipped", "Requires flash-attn 3"),
            "t_4gpu::x": Outcome("skipped", "requires four GPUs"),
            "t_broken::x": Outcome("passed"),
        },
    }
    report = mod.analyze(results)

    # flash_4 ran on fa4, flash_3 ran on cu128 -> both covered, no gaps.
    assert report.genuine_gaps == []
    # the 4-GPU test skipped everywhere is excluded, not a gap.
    assert report.excluded_gaps == [("t_4gpu::x", 4)]
    # the failing test is surfaced and makes the report not-ok.
    assert report.failures["cu128"] == ["t_broken::x"]
    assert report.ok is False


def test_analyze_flags_genuine_gap():
    # A non-GPU-count test skipped on every image is a real coverage gap.
    results = {
        "cu128": {"t_missing::x": Outcome("skipped", "Requires flash-attn 3")},
        "cu130-fa4": {"t_missing::x": Outcome("skipped", "Requires flash-attn 3")},
    }
    report = mod.analyze(results)
    assert report.genuine_gaps == ["t_missing::x"]
    assert report.ok is False


def test_analyze_all_covered_is_ok():
    results = {
        "cu128": {"a::x": Outcome("passed"), "b::x": Outcome("skipped", "Requires flash-attn 4")},
        "cu130-fa4": {"a::x": Outcome("passed"), "b::x": Outcome("passed")},
    }
    report = mod.analyze(results)
    assert report.ok is True
    assert report.genuine_gaps == []


def test_parse_junit(tmp_path):
    xml = tmp_path / "cu128.xml"
    xml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <testsuites><testsuite name="pytest">
          <testcase classname="src.test.nn.attention.attention_test" name="test_a[flash_2]"/>
          <testcase classname="src.test.nn.attention.attention_test" name="test_b[flash_4]">
            <skipped message="Requires flash-attn 4"/>
          </testcase>
          <testcase classname="src.test.foo" name="test_c">
            <failure message="boom"/>
          </testcase>
        </testsuite></testsuites>
        """
    )
    out = mod.parse_junit(xml)
    assert out["src.test.nn.attention.attention_test::test_a[flash_2]"].status == "passed"
    b = out["src.test.nn.attention.attention_test::test_b[flash_4]"]
    assert b.status == "skipped" and b.skip_reason == "Requires flash-attn 4"
    assert out["src.test.foo::test_c"].status == "failed"
