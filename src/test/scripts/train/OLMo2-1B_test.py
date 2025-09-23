import subprocess


def test_olmo2_1b_dry_run():
    # Run the dry_run command which should print config and exit cleanly
    result = subprocess.run(
        [
            "python",
            "src/scripts/train/OLMo2-1B.py",
            "dry_run",
            "my-test-run",
            "ai2/nerd-cluster",
        ],
        capture_output=True,
        text=True,
    )

    # Check that the command exited successfully (exit code 0)
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Optionally check that some expected output is present
    assert len(result.stdout) > 0, "Expected some output from dry_run command"
