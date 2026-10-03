"""
Chunked-prefill equivalence over chunk sizes, with and without the FLA length-autotune freeze.

The port failed test_generation_module_chunked_prefill_matches_one_shot at chunk_size=4 (5 and 16
passed); Amanda's original branch, which lacks the freeze, passes. Run on a GPU:

    python debug/gdn_int32_boundary/chunk_matrix.py           # freeze on (the branch as is)
    python debug/gdn_int32_boundary/chunk_matrix.py nofreeze  # freeze patched to a no-op
"""

import sys

import pytest

if len(sys.argv) > 1 and sys.argv[1] == "nofreeze":
    import olmo_core.nn.attention.fla_autotune as fa

    fa.freeze_fla_length_autotune = lambda: 0

T = "src/test/generate/generation_module/transformer/generation_module_test.py"
sys.exit(
    pytest.main(
        [
            "-q",
            "-rfE",
            "-p",
            "no:cacheprovider",
            "--show-capture=no",
            "--tb=line",
            "-k",
            "chunked_prefill",
            T,
        ]
    )
)
