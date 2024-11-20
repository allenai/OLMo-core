import pytest

from olmo_core.io import (
    deserialize_from_tensor,
    file_exists,
    list_directory,
    serialize_to_tensor,
    upload,
)


def test_serde_from_tensor():
    data = {"a": (1, 2)}
    assert deserialize_from_tensor(serialize_to_tensor(data)) == data


def test_list_local_directory(tmp_path):
    (tmp_path / "file1.json").touch()
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2").touch()

    # Should only list immediate children (files and dirs), but not files in subdirs.
    # The paths returned should be full paths.
    assert set(list_directory(tmp_path)) == {f"{tmp_path}/file1.json", f"{tmp_path}/dir1"}


def _run_list_remote_directory(tmp_path, remote_dir):
    (tmp_path / "file1.json").touch()
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2").touch()

    assert not file_exists(f"{remote_dir}/dir1/file2")

    for path in tmp_path.glob("**/*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(tmp_path)
        upload(path, f"{remote_dir}/{rel_path}")
        assert file_exists(f"{remote_dir}/{rel_path}")

    # Should only list immediate children (files and dirs), but not files in subdirs.
    # The paths returned should be full paths.
    assert set(list_directory(remote_dir)) == {
        f"{remote_dir}/file1.json",
        f"{remote_dir}/dir1",
    }


def test_list_remote_directory_s3(tmp_path, s3_checkpoint_dir):
    from botocore.exceptions import NoCredentialsError

    try:
        _run_list_remote_directory(tmp_path, s3_checkpoint_dir)
    except NoCredentialsError:
        pytest.skip("Requires AWS credentials")


def test_list_remote_directory_gcs(tmp_path, gcs_checkpoint_dir):
    from google.auth.exceptions import DefaultCredentialsError

    try:
        _run_list_remote_directory(tmp_path, gcs_checkpoint_dir)
    except DefaultCredentialsError:
        pytest.skip("Requires authentication with Google Cloud")
