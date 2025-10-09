from olmo_core.launch.utils import parse_git_remote_url


def test_parse_git_remote_url():
    # HTTPS format.
    assert parse_git_remote_url("https://github.com/allenai/OLMo-core.git") == (
        "allenai",
        "OLMo-core",
    )
    # SSH format.
    assert parse_git_remote_url("git@github.com:allenai/OLMo-core.git") == (
        "allenai",
        "OLMo-core",
    )
    # Username+password format.
    assert parse_git_remote_url("https://USERNAME:PASSWORD@github.com/allenai/OLMo-core.git") == (
        "allenai",
        "OLMo-core",
    )
