#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beaker-py>2.0,<3.0",
#     "huggingface-hub",
# ]
# ///

import argparse
from contextlib import ExitStack
import dataclasses as dt
from typing import Self, Literal
import os
import subprocess
from pathlib import Path

from beaker import Beaker
from beaker.exceptions import BeakerSecretNotFound
from huggingface_hub import get_token as get_hugging_face_hub_token


def get_aws_config_file(file_type: Literal["config", "credentials"]) -> str | None:
    """Get AWS config or credentials as raw string, using CLI to find location.

    Args:
        file_type: Either "config" or "credentials"
    """

    # Map file type to what appears in CLI output
    location_markers = {
        "config": "config-file",
        "credentials": "shared-credentials-file",
    }
    marker = location_markers.get(file_type, file_type)

    # Try to get file location from AWS CLI
    try:
        result = subprocess.run(
            ["aws", "configure", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.split("\n"):
            if marker in line:
                config_path = Path(line.split()[-1]).expanduser()
                if config_path.exists():
                    return config_path.read_text()
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        pass

    # Fallback to default locations
    default_paths = {
        "config": os.environ.get("AWS_CONFIG_FILE", Path.home() / ".aws" / "config"),
        "credentials": os.environ.get(
            "AWS_SHARED_CREDENTIALS_FILE", Path.home() / ".aws" / "credentials"
        ),
    }
    config_path = Path(default_paths.get(file_type, ""))
    if config_path.exists():
        return config_path.read_text()

    return None


def get_beaker_token(client: Beaker | None = None) -> str:
    with ExitStack() as stack:
        client = stack.enter_context(Beaker.from_env() if client is None else client)
        assert client is not None, "Client shouldn't be None here"

        return client.config.user_token


@dt.dataclass(frozen=True)
class EnvironmentVariables:
    wandb_api_key: str
    aws_config: str
    aws_credentials: str
    hugging_face_hub_token: str
    google_credentials: str
    beaker_token: str
    comet_api_key: str | None = None
    r2_endpoint_url: str | None = None
    weka_endpoint_url: str | None = None

    @staticmethod
    def get_secret_name(field: dt.Field, client: Beaker | None = None) -> str:
        with ExitStack() as stack:
            client = stack.enter_context(Beaker.from_env() if client is None else client)
            assert client is not None, "Client shouldn't be None here"

            if field.name != "google_credentials":
                return f"{client.user_name.upper()}_{field.name.upper()}"
            else:
                return "GOOGLE_CREDENTIALS"

    @classmethod
    def from_workspace(cls, workspace_name: str) -> Self:
        env_vars: dict[str, str] = {}

        with Beaker.from_env() as client:
            workspace = client.workspace.get(workspace_name)

            for field in dt.fields(cls):
                secret_name = cls.get_secret_name(field, client)
                try:
                    secret_object = client.secret.get(secret_name, workspace=workspace)
                except BeakerSecretNotFound as e:
                    if field.default is not dt.MISSING or field.default_factory is not dt.MISSING:
                        raise ValueError(
                            f"Secret {secret_name} not found in workspace {workspace.name}"
                        ) from e
                    else:
                        print(
                            f"Secret {secret_name} not found in workspace {workspace.name}; skipping..."
                        )
                        continue

                secret_value = client.secret.read(secret_object, workspace=workspace)
                env_vars[field.name] = secret_value

            # in case it's missing
            env_vars.setdefault("beaker_token", get_beaker_token(client))

        return cls(**env_vars)

    @classmethod
    def from_environment(
        cls,
    ) -> Self:
        env_vars: dict[str, str] = {}

        if (wandb_api_key := os.getenv("WANDB_API_KEY")) is not None:
            env_vars["wandb_api_key"] = wandb_api_key

        if (aws_config := get_aws_config_file("config")) is not None:
            env_vars["aws_config"] = aws_config

        if (aws_credentials := get_aws_config_file("credentials")) is not None:
            env_vars["aws_credentials"] = aws_credentials

        if (r2_endpoint_url := os.getenv("R2_ENDPOINT_URL")) is not None:
            env_vars["r2_endpoint_url"] = r2_endpoint_url

        if (weka_endpoint_url := os.getenv("WEKA_ENDPOINT_URL")) is not None:
            env_vars["weka_endpoint_url"] = weka_endpoint_url

        if (hugging_face_hub_token := get_hugging_face_hub_token()) is not None:
            env_vars["hugging_face_hub_token"] = hugging_face_hub_token

        if (google_credentials := os.getenv("GOOGLE_CREDENTIALS")) is not None:
            env_vars["google_credentials"] = google_credentials

        env_vars.setdefault("beaker_token", get_beaker_token())

        return cls(**env_vars)

    def push_to_workspace(self, workspace_name: str, overwrite: bool = False) -> None:
        with Beaker.from_env() as client:
            workspace = client.workspace.get(workspace_name)
            for field in dt.fields(self):
                secret_name = self.get_secret_name(field, client)

                try:
                    client.secret.get(secret_name, workspace=workspace)
                    if not overwrite:
                        print(f"Skipping {secret_name} because it already exists...")
                        continue
                except BeakerSecretNotFound:
                    pass

                if (secret_value := getattr(self, field.name)) is None:
                    print(f"Skipping {secret_name} because I did not find it...")
                    continue

                print(
                    f"Pushing {secret_name} to workspace {workspace.name} with value {secret_value[:6]}***..."
                )
                client.secret.write(name=secret_name, value=secret_value, workspace=workspace)


def prepare_environment():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--from-workspace",
        type=str,
        default=None,
        help="Beaker workspace to read secrets from; if not provided, secrets will be read from the environment.",
    )
    parser.add_argument(
        "-w", "--workspace", type=str, required=True, help="Beaker workspace to push secrets to."
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing secrets in the workspace.",
    )
    args = parser.parse_args()

    env_vars = (
        EnvironmentVariables.from_workspace(args.from_workspace)
        if args.from_workspace is not None
        else EnvironmentVariables.from_environment()
    )

    env_vars.push_to_workspace(workspace_name=args.workspace, overwrite=args.overwrite)


if __name__ == "__main__":
    prepare_environment()
