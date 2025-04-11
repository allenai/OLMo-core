from dataclasses import dataclass
from typing import Optional, Tuple

from olmo_core.config import Config


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


@dataclass
class GitConfig(Config):
    repo_url: str
    ref: str
    is_public: bool
    is_dirty: bool
    branch: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GitConfig":
        import requests
        from git.repo import Repo

        repo = Repo(".")
        #  if repo.is_dirty() and not allow_dirty:
        #  raise RuntimeError("You have uncommitted changes! Use --allow-dirty to force.")

        git_ref = str(repo.commit())
        remote = repo.remote()

        # Try to find a remote based on the current tracking branch.
        try:
            branch = repo.active_branch
        except TypeError:
            branch = None

        if branch is not None:
            branch = branch.tracking_branch()

        branch_name: Optional[str] = None
        if branch is not None:
            remote = repo.remote(branch.remote_name)
            assert branch.name.startswith(branch.remote_name + "/")
            branch_name = branch.name.replace(branch.remote_name + "/", "", 1)

        account, repo_name = parse_git_remote_url(remote.url)
        repo_url = f"https://github.com/{account}/{repo_name}"

        # Check if repo is public.
        response = requests.get(repo_url)
        if response.status_code not in {200, 404}:
            response.raise_for_status()
        is_public = response.status_code == 200

        return cls(
            repo_url=repo_url,
            ref=git_ref,
            is_public=is_public,
            is_dirty=repo.is_dirty(),
            branch=branch_name,
        )
