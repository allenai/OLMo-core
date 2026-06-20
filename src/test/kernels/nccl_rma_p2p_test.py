from pathlib import Path

import pytest

from olmo_core.kernels import nccl_rma_p2p


def _write_nccl_install(root: Path, version: int, *, with_lib: bool = True) -> tuple[Path, Path]:
    include_dir = root / "include"
    lib_dir = root / "lib"
    include_dir.mkdir(parents=True)
    lib_dir.mkdir()
    (include_dir / "nccl.h").write_text(
        f"#define NCCL_VERSION_CODE {version}\n",
        encoding="utf-8",
    )
    if with_lib:
        (lib_dir / "libnccl.so.2").write_text("", encoding="utf-8")
    return include_dir, lib_dir


def test_nccl_header_version(tmp_path: Path):
    include_dir, _ = _write_nccl_install(tmp_path / "nccl", 22907)

    assert nccl_rma_p2p._nccl_header_version(include_dir) == 22907


def test_find_nccl_paths_skips_old_header(monkeypatch, tmp_path: Path):
    old_include, old_lib = _write_nccl_install(tmp_path / "old", 22809)
    new_include, new_lib = _write_nccl_install(tmp_path / "new", 22907)

    monkeypatch.setattr(
        nccl_rma_p2p,
        "_candidate_nccl_paths",
        lambda: [
            (old_include, [old_lib], "old"),
            (new_include, [new_lib], "new"),
        ],
    )

    assert nccl_rma_p2p._find_nccl_paths() == (
        new_include,
        new_lib,
        new_lib / "libnccl.so.2",
    )


def test_find_nccl_paths_errors_when_only_old_header(monkeypatch, tmp_path: Path):
    old_include, old_lib = _write_nccl_install(tmp_path / "old", 22809)

    monkeypatch.setattr(
        nccl_rma_p2p,
        "_candidate_nccl_paths",
        lambda: [(old_include, [old_lib], "old")],
    )

    with pytest.raises(RuntimeError, match="NCCL RMA P2P requires NCCL headers"):
        nccl_rma_p2p._find_nccl_paths()


def test_nccl_link_flag_uses_exact_soname_for_versioned_lib():
    assert nccl_rma_p2p._nccl_link_flag(Path("/x/libnccl.so.2")) == "-l:libnccl.so.2"
    assert nccl_rma_p2p._nccl_link_flag(Path("/x/libnccl.so")) == "-lnccl"
