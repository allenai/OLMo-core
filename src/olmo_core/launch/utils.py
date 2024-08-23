from typing import Tuple


def parse_git_remote_url(url: str) -> Tuple[str, str]:
    """
    Parse a git remote URL into a GitHub (account, repo) pair.

    :raises InvalidRemoteError: If the URL can't be parsed correctly.
    """
    try:
        account, repo = (
            url.split("https://github.com/")[-1]
            .split("git@github.com:")[-1]
            .split(".git")[0]
            .split("/")
        )
    except ValueError:
        raise ValueError(f"Failed to parse GitHub repo path from remote '{url}'")
    return account, repo


def ensure_repo(allow_dirty: bool = False) -> Tuple[str, str, str, bool]:
    import requests
    from git.repo import Repo

    repo = Repo(".")
    if repo.is_dirty() and not allow_dirty:
        raise RuntimeError("You have uncommitted changes! Use --allow-dirty to force.")
    git_ref = str(repo.commit())
    account, repo = parse_git_remote_url(repo.remote().url)
    response = requests.get(f"https://github.com/{account}/{repo}")
    if response.status_code not in {200, 404}:
        response.raise_for_status()
    is_public = response.status_code == 200
    return account, repo, git_ref, is_public
