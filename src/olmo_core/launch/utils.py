import os
from dataclasses import dataclass
from typing import Optional, Tuple

import requests

from olmo_core.config import Config

GIT_REPO_URL_ENV_VAR = "REPO_URL"
GIT_REF_ENV_VAR = "GIT_REF"
GIT_BRANCH_ENV_VAR = "GIT_BRANCH"


def parse_git_remote_url(url: str) -> Tuple[str, str]:
    """
    Parse a git remote URL into a GitHub (account, repo) pair.

    :raises InvalidRemoteError: If the URL can't be parsed correctly.
    """
    if "github.com" not in url:
        raise ValueError(f"Remote ('{url}') must point to a GitHub repo")
    try:
        account, repo = url.split("github.com", 1)[-1].strip("/:").split(".git")[0].split("/")
    except ValueError:
        raise ValueError(f"Failed to parse GitHub repo path from remote '{url}'")
    return account, repo


@dataclass
class GitConfig(Config):
    repo_url: str
    ref: str
    branch: Optional[str] = None

    @property
    def is_dirty(self) -> bool:
        from git.exc import InvalidGitRepositoryError
        from git.repo import Repo

        try:
            repo = Repo(".")
            return repo.is_dirty()
        except InvalidGitRepositoryError:
            return False

    @property
    def is_public(self) -> bool:
        response = requests.get(self.repo_url)
        if response.status_code not in {200, 404}:
            response.raise_for_status()
        return response.status_code == 200

    @classmethod
    def from_env(cls) -> Optional["GitConfig"]:
        from git.exc import InvalidGitRepositoryError
        from git.repo import Repo

        try:
            repo = Repo(".")
        except InvalidGitRepositoryError:
            return None

        git_ref = os.environ.get(GIT_REF_ENV_VAR, str(repo.commit()))
        remote = repo.remote()

        # Try to find a remote based on the current tracking branch.
        try:
            branch = repo.active_branch
        except TypeError:
            branch = None

        if branch is not None:
            branch = branch.tracking_branch()

        branch_name = os.environ.get(GIT_BRANCH_ENV_VAR)
        if branch is not None:
            remote = repo.remote(branch.remote_name)
            if branch_name is None:
                assert branch.name.startswith(branch.remote_name + "/")
                branch_name = branch.name.replace(branch.remote_name + "/", "", 1)

        if (repo_url := os.environ.get(GIT_REPO_URL_ENV_VAR)) is None:
            account, repo_name = parse_git_remote_url(remote.url)
            repo_url = f"https://github.com/{account}/{repo_name}"

        return cls(
            repo_url=repo_url,
            ref=git_ref,
            branch=branch_name,
        )
