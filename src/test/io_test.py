from glob import glob

import pytest

from olmo_core.io import (
    copy_dir,
    copy_file,
    deserialize_from_tensor,
    file_exists,
    glob_directory,
    list_directory,
    serialize_to_tensor,
    upload,
)


def test_serde_from_tensor():
    data = {"a": (1, 2)}
    assert deserialize_from_tensor(serialize_to_tensor(data)) == data


def test_local_functionality(tmp_path):
    (tmp_path / "file1.json").touch()
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2").touch()

    # Should only list immediate children (files and dirs), but not files in subdirs.
    # The paths returned should be full paths.
    assert set(list_directory(tmp_path)) == {f"{tmp_path}/file1.json", f"{tmp_path}/dir1"}
    assert set(list_directory(tmp_path, recurse=True)) == {
        f"{tmp_path}/file1.json",
        f"{tmp_path}/dir1",
        f"{tmp_path}/dir1/file2",
    }

    (tmp_path / "dir1" / "subdir1").mkdir()
    (tmp_path / "dir1" / "subdir1" / "file1").touch()

    copy_dir(tmp_path / "dir1", tmp_path / "dir2")
    assert set(list_directory(tmp_path / "dir2", recurse=True)) == {
        f"{tmp_path}/dir2/file2",
        f"{tmp_path}/dir2/subdir1",
        f"{tmp_path}/dir2/subdir1/file1",
    }


def _run_remote_functionality(tmp_path, remote_dir):
    (tmp_path / "file1.json").touch()
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2.json").touch()

    assert not file_exists(f"{remote_dir}/dir1/file2.json")

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

    # Should list all children.
    assert set(list_directory(remote_dir, recurse=True)) == {
        f"{remote_dir}/file1.json",
        f"{remote_dir}/dir1",
        f"{remote_dir}/dir1/file2.json",
    }

    # Should list top-level json files.
    assert set(glob_directory(f"{remote_dir}/*.json")) == {
        f"{remote_dir}/file1.json",
    }

    # Should list all json files.
    assert set(glob_directory(f"{remote_dir}/**/*.json")) == {
        f"{remote_dir}/file1.json",
        f"{remote_dir}/dir1/file2.json",
    }

    # Try copying to a file that already exists.
    with pytest.raises(FileExistsError):
        copy_file(f"{remote_dir}/dir1/file2.json", tmp_path / "dir1/file2.json")
    copy_file(f"{remote_dir}/dir1/file2.json", tmp_path / "dir1/file2.json", save_overwrite=True)

    # Copy to a new file that doesn't exist.
    copy_file(f"{remote_dir}/dir1/file2.json", tmp_path / "dir2/file2.json")
    assert (tmp_path / "dir2/file2.json").is_file()

    # Copy dir.
    copy_dir(f"{remote_dir}", tmp_path / "dir3")
    assert (tmp_path / "dir3/dir1/file2.json").is_file()


def test_s3_functionality(tmp_path, s3_checkpoint_dir):
    from botocore.exceptions import NoCredentialsError

    try:
        _run_remote_functionality(tmp_path, s3_checkpoint_dir)
    except NoCredentialsError:
        pytest.skip("Requires AWS credentials")


def test_gcs_functionality(tmp_path, gcs_checkpoint_dir):
    from google.auth.exceptions import DefaultCredentialsError

    try:
        _run_remote_functionality(tmp_path, gcs_checkpoint_dir)
    except DefaultCredentialsError:
        pytest.skip("Requires authentication with Google Cloud")


def test_glob_directory():
    assert set(glob("*.md")) == set(glob_directory("*.md"))
    assert set(glob("src/examples/**/*.py", recursive=True)) == set(
        glob_directory("src/examples/**/*.py")
    )
