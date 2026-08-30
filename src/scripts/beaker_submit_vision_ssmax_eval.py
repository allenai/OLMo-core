"""Validate and submit only a reviewed dense-SSMax downstream evaluation spec.

The wrapper fixes the workspace to ``ai2/scaling-ladders`` and rejects specs that weaken urgent
priority, the eight-hour minimum runtime, Holmes-only placement, immutable checkpoint identity,
or the exact evaluator protocol. Paired submission remains the default; ``--single-arm`` permits
an explicitly named arm to be staged while its matched checkpoint is still training. Template
validation is read-only and never invokes Beaker.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

BEAKER_WORKSPACE = "ai2/scaling-ladders"
BEAKER_BUDGET = "ai2/oe-other"
BEAKER_CLUSTER = "ai2/holmes"
LMMS_EVAL_REVISION = "cb45ac4d4a667ea5ef89c7a148bff69b3489b981"
SSMAX_VARIANTS = ("ssmax_head_qknorm", "ssmax_no_qknorm")
SSMAX_PHASES = ("bridge", "perception", "joint")
EVAL_SCRIPT = "src/scripts/eval/vision_alignment_ssmax_downstream.py"
EVAL_ROOT = (
    "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/"
    "vision-alignment/evals"
)
CHECKPOINT_ROOT = (
    "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/"
    "vision-alignment/checkpoints"
)

OUTPUT_FORMATS = ("text", "json", "yaml", "csv")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SAFE_PATH_PART = re.compile(r"[A-Za-z0-9._ <>-]+")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_PLACEHOLDER = re.compile(r"<[A-Z0-9_]+>")
_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")
_EXPECTED_ARGUMENT_FLAGS = (
    "--checkpoint",
    "--expected-model-variant",
    "--expected-phase",
    "--expected-global-step",
    "--expected-config-sha256",
    "--expected-marker-sha256",
    "--expected-dcp-metadata-sha256",
    "--expected-checkpoint-identity-sha256",
    "--max-sequence-length",
    "--max-crops-total",
    "--sequence-bucket-size",
    "--checkpoint-load-threads",
    "--output",
)
_ALLOWED_SECRETS = {"BEAKER_TOKEN": "RUSTINS_BEAKER_TOKEN"}
_FORBIDDEN_AUTH_ENV = frozenset(
    {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
)


class SpecValidationError(ValueError):
    """Raised when a direct Beaker spec differs from the reviewed contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("-n", "--name", help="Optional Beaker experiment name")
    parser.add_argument("--format", choices=OUTPUT_FORMATS)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate without invoking Beaker"
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Validate a .template file structurally; requires --validate-only",
    )
    parser.add_argument(
        "--single-arm",
        choices=SSMAX_VARIANTS,
        help="Explicitly validate and submit exactly one staged model arm",
    )
    return parser


def _validated_path(parser: argparse.ArgumentParser, spec: Path, *, template: bool) -> Path:
    suffixes = spec.suffixes
    if template:
        if suffixes[-2:] not in ([".yaml", ".template"], [".yml", ".template"]):
            parser.error("placeholder validation requires a .yaml.template or .yml.template file")
    elif spec.suffix.lower() not in {".yaml", ".yml"}:
        parser.error("submission spec must have a .yaml or .yml suffix")
    if any(part == ".." for part in spec.parts):
        parser.error(f"spec path traversal is not allowed: {spec}")
    for part in spec.parts:
        if part in {spec.anchor, "."}:
            continue
        if part.startswith("-") or _SAFE_PATH_PART.fullmatch(part) is None:
            parser.error(f"spec contains an unsafe path component: {part!r}")
    absolute = Path(os.path.abspath(spec))
    try:
        resolved = spec.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        parser.error(f"could not resolve spec {spec}: {error}")
    if absolute != resolved:
        parser.error(f"spec must not use symlinks: {spec}")
    if not resolved.is_file():
        parser.error(f"spec does not exist or is not a file: {spec}")
    return resolved


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{name} must be an object")
    return value


def _duration_seconds(value: Any, *, name: str) -> int:
    if not isinstance(value, str) or (match := _DURATION.fullmatch(value)) is None:
        raise SpecValidationError(f"{name} must be a Beaker h/m/s duration")
    if not any(match.groups()):
        raise SpecValidationError(f"{name} must not be empty")
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _env(task: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = task.get("envVars")
    if not isinstance(rows, list):
        raise SpecValidationError("task envVars must be a list")
    output = {}
    for row in rows:
        item = _mapping(row, name="env var")
        name = item.get("name")
        if not isinstance(name, str) or name in output:
            raise SpecValidationError("environment names must be unique strings")
        sources = [field for field in ("value", "secret") if field in item]
        if len(sources) != 1:
            raise SpecValidationError(f"environment {name} must set exactly one value or secret")
        if sources[0] == "secret":
            secret = item["secret"]
            if not isinstance(secret, str) or not secret:
                raise SpecValidationError(f"environment {name} has an invalid secret reference")
            output[name] = {"secret": secret}
        else:
            output[name] = item["value"]
    return output


def _argument_value(arguments: Sequence[Any], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise SpecValidationError(f"task must provide exactly one {flag}")
    value = arguments[positions[0] + 1]
    if not isinstance(value, str) or value.startswith("--"):
        raise SpecValidationError(f"{flag} must have one string value")
    return value


def _is_placeholder(value: str) -> bool:
    return _PLACEHOLDER.fullmatch(value) is not None


def validate_spec(
    spec: Mapping[str, Any],
    *,
    allow_placeholders: bool,
    expected_single_arm: str | None = None,
) -> None:
    """Validate the complete paired or explicitly staged Beaker experiment contract."""
    if spec.get("version") != "v2":
        raise SpecValidationError("spec version must be v2")
    if spec.get("budget") != BEAKER_BUDGET:
        raise SpecValidationError(f"spec budget must be {BEAKER_BUDGET}")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list):
        raise SpecValidationError("spec tasks must be a list")
    if expected_single_arm is None:
        expected_variants = set(SSMAX_VARIANTS)
        if len(tasks) != 2:
            raise SpecValidationError("spec must contain exactly the two model-arm tasks")
    else:
        expected_variants = {expected_single_arm}
        if len(tasks) != 1:
            raise SpecValidationError("single-arm staging must contain exactly one model-arm task")

    seen_variants: set[str] = set()
    seen_outputs: set[str] = set()
    seen_refs: set[str] = set()
    seen_phases: set[str] = set()
    seen_steps: set[str] = set()
    for task_value in tasks:
        task = _mapping(task_value, name="task")
        if task.get("command") != ["bash", "/gantry/entrypoint.sh"]:
            raise SpecValidationError("task command must use the pinned Gantry entrypoint")
        arguments = task.get("arguments")
        if not isinstance(arguments, list) or arguments[:2] != ["python", EVAL_SCRIPT]:
            raise SpecValidationError(f"task must invoke python {EVAL_SCRIPT}")
        if arguments[2::2] != list(_EXPECTED_ARGUMENT_FLAGS) or len(arguments) != 2 + 2 * len(
            _EXPECTED_ARGUMENT_FLAGS
        ):
            raise SpecValidationError(
                "task must use the exact reviewed evaluator argument surface and order"
            )
        variant = _argument_value(arguments, "--expected-model-variant")
        if variant not in expected_variants:
            raise SpecValidationError(f"unexpected SSMax variant {variant!r}")
        if variant in seen_variants:
            raise SpecValidationError(f"duplicate model-arm task {variant}")
        seen_variants.add(variant)
        phase = _argument_value(arguments, "--expected-phase")
        if phase not in SSMAX_PHASES:
            raise SpecValidationError(f"expected phase must be one of {SSMAX_PHASES}")
        seen_phases.add(phase)
        global_step = _argument_value(arguments, "--expected-global-step")
        if allow_placeholders:
            if not _is_placeholder(global_step):
                raise SpecValidationError("template global step must be a placeholder")
        elif not global_step.isdecimal() or str(int(global_step)) != global_step:
            raise SpecValidationError(
                "expected global step must be a canonical non-negative integer"
            )
        seen_steps.add(global_step)
        if _argument_value(arguments, "--max-sequence-length") != "8192":
            raise SpecValidationError("max sequence length must be 8192")
        if _argument_value(arguments, "--max-crops-total") != "8":
            raise SpecValidationError("the document-interface crop budget must be shared total 8")
        if _argument_value(arguments, "--sequence-bucket-size") != "128":
            raise SpecValidationError("sequence bucket size must be 128")
        if _argument_value(arguments, "--checkpoint-load-threads") != "8":
            raise SpecValidationError("checkpoint hash/load thread count must be 8")

        checkpoint = _argument_value(arguments, "--checkpoint")
        output = _argument_value(arguments, "--output")
        hashes = [
            _argument_value(arguments, flag)
            for flag in (
                "--expected-config-sha256",
                "--expected-marker-sha256",
                "--expected-dcp-metadata-sha256",
                "--expected-checkpoint-identity-sha256",
            )
        ]
        if allow_placeholders:
            if not (_is_placeholder(checkpoint) and _is_placeholder(output)):
                raise SpecValidationError("template checkpoint and output must be placeholders")
            if not all(_is_placeholder(value) for value in hashes):
                raise SpecValidationError("template checkpoint hashes must be placeholders")
        else:
            if _PLACEHOLDER.search(" ".join(str(value) for value in arguments)):
                raise SpecValidationError("runnable spec contains unresolved placeholders")
            if not checkpoint.startswith(CHECKPOINT_ROOT + "/"):
                raise SpecValidationError(f"checkpoint must be below {CHECKPOINT_ROOT}")
            if Path(checkpoint).name != f"step{global_step}":
                raise SpecValidationError(
                    f"checkpoint path must end in the declared step{global_step}"
                )
            if not output.startswith(EVAL_ROOT + "/"):
                raise SpecValidationError(f"output must be below {EVAL_ROOT}")
            if not all(_HEX64.fullmatch(value) for value in hashes):
                raise SpecValidationError("checkpoint identity arguments must be SHA-256 digests")
        if output in seen_outputs:
            raise SpecValidationError("model arms must write distinct result paths")
        seen_outputs.add(output)

        resources = _mapping(task.get("resources"), name="resources")
        if resources.get("gpuCount") != 1:
            raise SpecValidationError("each dense 1.4B fast-suite task must request one GPU")
        context = _mapping(task.get("context"), name="context")
        if context.get("priority") != "urgent":
            raise SpecValidationError("task priority must be urgent")
        if _duration_seconds(context.get("minRuntime"), name="minRuntime") < 8 * 3600:
            raise SpecValidationError("task minRuntime must be at least eight hours")
        constraints = _mapping(task.get("constraints"), name="constraints")
        if constraints.get("cluster") != [BEAKER_CLUSTER]:
            raise SpecValidationError(f"task cluster must be [{BEAKER_CLUSTER!r}]")
        if any(name in constraints for name in ("hostname", "hostnames")):
            raise SpecValidationError("exact host selection is forbidden")
        if task.get("hostNetworking") is not False:
            raise SpecValidationError("hostNetworking must remain disabled")

        environment = _env(task)
        secrets = {
            name: value["secret"]
            for name, value in environment.items()
            if isinstance(value, Mapping)
        }
        if secrets != _ALLOWED_SECRETS:
            raise SpecValidationError(
                f"task secret surface must be exactly {_ALLOWED_SECRETS}, got {secrets}"
            )
        forbidden_auth = sorted(_FORBIDDEN_AUTH_ENV.intersection(environment))
        if forbidden_auth:
            raise SpecValidationError(
                f"public, judge-free tasks forbid auth environment variables {forbidden_auth}"
            )
        if environment.get("GITHUB_REPO") != "allenai/OLMo-core":
            raise SpecValidationError("GITHUB_REPO must be allenai/OLMo-core")
        git_ref = environment.get("GIT_REF")
        if not isinstance(git_ref, str):
            raise SpecValidationError("GIT_REF must be a string")
        if allow_placeholders:
            if not _is_placeholder(git_ref):
                raise SpecValidationError("template GIT_REF must be a placeholder")
        elif _HEX40.fullmatch(git_ref) is None:
            raise SpecValidationError("GIT_REF must be an immutable 40-hex commit")
        seen_refs.add(git_ref)
        setup = environment.get("GANTRY_PRE_SETUP_CMD")
        if not isinstance(setup, str) or LMMS_EVAL_REVISION not in setup:
            raise SpecValidationError("pre-setup must install the pinned lmms-eval revision")
        if "datasets==3.6.0" not in setup or "pyarrow==19.0.1" not in setup:
            raise SpecValidationError("pre-setup must pin datasets and pyarrow")
        if "GANTRY_USE_TORCHRUN" in environment:
            raise SpecValidationError("single-process dense eval must not enable torchrun")

        datasets = task.get("datasets")
        if not isinstance(datasets, list) or not any(
            isinstance(row, Mapping)
            and row.get("mountPath") == "/weka/oe-training-default"
            and row.get("source") == {"weka": "oe-training-default"}
            for row in datasets
        ):
            raise SpecValidationError("task must mount oe-training-default Weka")

    if seen_variants != expected_variants:
        if expected_single_arm is None:
            raise SpecValidationError("spec must cover QK-norm and no-QK-norm exactly once")
        raise SpecValidationError(f"single-arm spec must cover only {expected_single_arm!r}")
    if len(seen_phases) != 1:
        raise SpecValidationError("paired model arms must evaluate the same declared phase")
    if len(seen_steps) != 1:
        raise SpecValidationError("paired model arms must evaluate the same declared global step")
    if len(seen_refs) != 1:
        raise SpecValidationError("paired model arms must run the same evaluator commit")


def _command(
    spec: Path,
    *,
    name: str | None,
    output_format: str | None,
    quiet: bool,
) -> list[str]:
    command = ["beaker", "experiment", "create", str(spec), f"--workspace={BEAKER_WORKSPACE}"]
    if name is not None:
        command.append(f"--name={name}")
    if output_format is not None:
        command.append(f"--format={output_format}")
    if quiet:
        command.append("--quiet")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.allow_placeholders and not args.validate_only:
        raise SpecValidationError("--allow-placeholders requires --validate-only")
    if args.name is not None and _SAFE_NAME.fullmatch(args.name) is None:
        raise SpecValidationError("unsafe Beaker experiment name")
    path = _validated_path(_parser(), args.spec, template=args.allow_placeholders)
    with path.open() as file_handle:
        payload = yaml.safe_load(file_handle)
    validate_spec(
        _mapping(payload, name="spec"),
        allow_placeholders=args.allow_placeholders,
        expected_single_arm=args.single_arm,
    )
    if args.validate_only:
        return 0
    subprocess.run(
        _command(path, name=args.name, output_format=args.format, quiet=args.quiet),
        check=True,
        shell=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
