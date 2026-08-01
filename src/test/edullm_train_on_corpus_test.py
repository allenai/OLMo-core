"""What ``.edullm/train_on_corpus.py`` refuses, and what it lets through.

The entry point is not importable as a package -- it sits in ``.edullm/`` because that is what
the platform's image build copies and runs -- so it is loaded by path here. The alternative
was to leave the file untested, and the things it checks are precisely the ones that produce a
working run on wrong data rather than an error.

``edullm_data`` is not installed in this repository's CI, only in the built image. That is why
the module imports the reader inside ``resolve_corpus`` and why everything tested below is
reachable without it.
"""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


def _load():
    path = Path(__file__).parent.parent.parent / ".edullm" / "train_on_corpus.py"
    spec = importlib.util.spec_from_file_location("edullm_train_on_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


entry = _load()


@dataclass
class FakeManifest:
    """The shape ``edullm_data.read.dataset_paths`` returns, with the fields that matter.

    Defaults describe a healthy corpus -- headerless, little-endian, uint32 -- so each test
    below changes exactly the one thing it is about.
    """

    paths: List[str] = field(default_factory=lambda: ["s3://edullm-data/x/v1/tokens/a.u32le.bin"])
    dtype: Optional[str] = "uint32"
    byte_order: Optional[str] = "little"
    header_bytes: int = 0
    rows: Optional[int] = 1000


def resolve(manifest: FakeManifest, tokenizer: str = "tokenizer/dolma2-bpe"):
    return entry.corpus_from_manifest(
        manifest, dataset_id="pretrain/regmix-10b", version="v1", tokenizer_id=tokenizer
    )


class ReaderProtocolStub:
    """The four methods ``edullm_data.read`` calls on whatever it is handed.

    A boto3 client has none of them, which is the entire subject of the two tests below.
    """

    def get(self, bucket, key):
        ...

    def get_range(self, bucket, key, start, length):
        ...

    def head(self, bucket, key):
        ...

    def list(self, bucket, prefix):
        ...


@pytest.fixture
def reader(monkeypatch):
    """A stand-in for the installed reader, recording what ``resolve_corpus`` hands it.

    ``edullm_data`` is not installed in this repository's CI, so the modules are built here
    and put in ``sys.modules`` before the import inside ``resolve_corpus`` runs.
    """
    import types

    handed: Dict[str, Any] = {}
    adapter = ReaderProtocolStub()

    class Boto3S3:
        @classmethod
        def default(cls, region="us-east-1"):
            handed["region"] = region
            return adapter

    def dataset_paths(dataset_id, version, *, s3, **_):
        handed["s3"] = s3
        return FakeManifest()

    def resolve_latest(dataset_id, *, s3, **_):
        handed["resolve_latest_s3"] = s3
        return "v7"

    # Typed Any because these are modules being built rather than imported, and mypy is
    # right that a fresh ModuleType has no such attributes until this assigns them.
    read_module: Any = types.ModuleType("edullm_data.read")
    read_module.dataset_paths = dataset_paths
    read_module.resolve_latest = resolve_latest
    s3_module: Any = types.ModuleType("edullm_data.s3")
    s3_module.Boto3S3 = Boto3S3
    package = types.ModuleType("edullm_data")

    monkeypatch.setitem(sys.modules, "edullm_data", package)
    monkeypatch.setitem(sys.modules, "edullm_data.read", read_module)
    monkeypatch.setitem(sys.modules, "edullm_data.s3", s3_module)
    return handed


def test_the_reader_is_handed_its_own_adapter_and_not_a_boto3_client(reader):
    """Mutation: pass ``boto3.client("s3")``, which is what this did and what it cost.

    The reader's ``s3`` parameter is typed against a four-method protocol and a boto3 client
    implements none of it, so ``_require_validated`` calls ``s3.head(bucket, key)`` and the
    run dies with an AttributeError before a byte leaves the account. Nothing catches it
    earlier: the parameter is named ``s3``, the annotation is a Protocol, and the traceback
    names a missing attribute rather than a wrong argument.

    On a GPU job that is a container which starts, exits 1 in under a second, and writes its
    only explanation to a log stream nobody on the platform side is allowed to read.
    """
    entry.resolve_corpus(
        dataset_id="pretrain/regmix-10b", version="v1", tokenizer_id="tokenizer/dolma2-bpe"
    )

    for method in ("get", "get_range", "head", "list"):
        assert callable(getattr(reader["s3"], method, None)), (
            f"the reader was handed something with no {method}(), which is what a boto3 "
            "client is"
        )


def test_resolving_the_latest_version_uses_the_same_adapter(reader):
    # The other call into the reader, and a second place a raw client could be passed.
    entry.resolve_corpus(
        dataset_id="pretrain/regmix-10b", version="latest", tokenizer_id="tokenizer/dolma2-bpe"
    )

    assert reader["resolve_latest_s3"] is reader["s3"]


def test_a_healthy_corpus_keeps_the_width_the_manifest_declared():
    # The whole reason this file exists. OLMo-core's own fallback would look at dolma2's
    # 100,278-token vocab, conclude uint16 fits it, and read a uint32 corpus two bytes at a
    # time -- producing in-range ids, no error, and a loss curve that is merely bad.
    corpus = resolve(FakeManifest())
    assert str(corpus.dtype) == "uint32"
    assert corpus.tokenizer.vocab_size == 100278


def test_a_corpus_with_a_header_is_refused_rather_than_read_from_offset_zero():
    with pytest.raises(SystemExit, match="header"):
        resolve(FakeManifest(header_bytes=128))


def test_a_big_endian_corpus_is_refused_on_a_little_endian_host():
    other = "big" if sys.byteorder == "little" else "little"
    with pytest.raises(SystemExit, match="endian"):
        resolve(FakeManifest(byte_order=other))


def test_a_corpus_that_declares_no_width_is_refused_rather_than_guessed_at():
    with pytest.raises(SystemExit, match="no dtype"):
        resolve(FakeManifest(dtype=None))


def test_no_trainable_shards_is_an_error_and_not_an_empty_run():
    # A corpus whose only splits are held out resolves to nothing. Training on zero shards is
    # not a shorter run, it is a run whose loss means nothing.
    with pytest.raises(SystemExit, match="no trainable shards"):
        resolve(FakeManifest(paths=[]))


def test_an_unknown_tokenizer_names_the_ones_this_image_has():
    # Rather than defaulting. A default here trains on ids that mean something other than what
    # they meant when the corpus was tokenized, and nothing downstream can tell.
    with pytest.raises(SystemExit, match="dolma2-bpe"):
        resolve(FakeManifest(), tokenizer="tokenizer/bytes-utf8")


def test_the_whole_config_builds_from_a_corpus_without_touching_s3(monkeypatch):
    """The check that would otherwise cost an A10G to run.

    Every mistake in the config below -- a TrainerConfig field that was renamed, a callback
    argument that does not exist, a model factory that is gone -- raises here in a second.
    Without this, the first thing that discovers it is a twelve-hour submission that reached a
    GPU, pulled a three-gigabyte image, and died before the first step.
    """
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifest(),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=s3://outputs/teams/platform/runs/a-run-id/checkpoints/",
            "--steps=25",
        ]
    )
    config = entry.build_config(opts, overrides)

    assert config.dataset.dtype == "uint32"
    assert config.dataset_id == "pretrain/regmix-10b"
    assert config.trainer.save_folder.endswith("/a-run-id/checkpoints/")
    # A retry must resume rather than overwrite what the first attempt left, which is the only
    # thing that makes a second attempt cheaper than a second run.
    assert config.trainer.save_overwrite is False
    # Pruning is off because the workload role has no delete permission. At OLMo-core's
    # default of three, the fourth save fails a run that is most of a day old.
    assert config.trainer.callbacks["checkpointer"].max_checkpoints is None
    # Serializing is what the config saver does beside the checkpoint; a config that cannot be
    # written is one whose record of what ran does not exist.
    assert config.as_config_dict()["dataset_version"] == "v1"


def test_an_override_on_the_command_line_reaches_the_config(monkeypatch):
    # The escape hatch researchers actually use: everything after the flags is merged into the
    # config, so a person can change the learning rate without a new entry point.
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifest(),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=/tmp/x",
            "train_module.compile_model=false",
        ]
    )
    config = entry.build_config(opts, overrides)
    assert config.train_module.compile_model is False


def test_the_platform_variables_are_required_and_named_when_missing(monkeypatch, capsys):
    for name in (
        "EDULLM_DATASET_ID",
        "EDULLM_DATASET_VERSION",
        "EDULLM_DATASET_TOKENIZER",
        "EDULLM_CHECKPOINT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["train_on_corpus", "some-run"])

    with pytest.raises(SystemExit) as refusal:
        entry.main()

    # Naming all four at once rather than the first one missing: a person fixing a submission
    # should not have to discover them one failed run at a time.
    message = str(refusal.value)
    for name in ("EDULLM_DATASET_ID", "EDULLM_CHECKPOINT_DIR"):
        assert name in message


class Boom(Exception):
    """Whatever the reader raises, wrapped in a message like the ones botocore produces."""


def resolve_with_a_reader_that_raises(monkeypatch, exc: BaseException):
    import types

    class Boto3S3:
        @classmethod
        def default(cls, region="us-east-1"):
            return ReaderProtocolStub()

    def dataset_paths(dataset_id, version, *, s3, **_):
        raise exc

    read_module: Any = types.ModuleType("edullm_data.read")
    read_module.dataset_paths = dataset_paths
    read_module.resolve_latest = lambda dataset_id, *, s3, **_: "v1"
    s3_module: Any = types.ModuleType("edullm_data.s3")
    s3_module.Boto3S3 = Boto3S3
    monkeypatch.setitem(sys.modules, "edullm_data", types.ModuleType("edullm_data"))
    monkeypatch.setitem(sys.modules, "edullm_data.read", read_module)
    monkeypatch.setitem(sys.modules, "edullm_data.s3", s3_module)

    with pytest.raises(entry.Refusal) as refusal:
        entry.resolve_corpus(
            dataset_id="pretrain/regmix-10b", version="v1", tokenizer_id="tokenizer/dolma2-bpe"
        )
    return refusal.value


def test_a_role_that_may_not_read_the_corpus_is_not_the_same_number_as_a_bad_run(monkeypatch):
    """Mutation: give every reader failure one code, which is what exit 1 already did.

    A missing ``s3:GetObject`` on ``edullm-data`` and a registry entry pointing at a prefix
    nobody published both arrive here as a failed read, and they have nothing in common: the
    first is an IAM change, the second is a dataset that is not there. Told apart at the
    exit code, the first question after a dead container is already answered.
    """
    denied = resolve_with_a_reader_that_raises(
        monkeypatch,
        Boom("An error occurred (AccessDenied) when calling the HeadObject operation"),
    )
    assert denied.stage is entry.Stage.THE_ROLE_MAY_NOT_READ_THE_CORPUS

    absent = resolve_with_a_reader_that_raises(
        monkeypatch, Boom("An error occurred (NoSuchKey) when calling the GetObject operation")
    )
    assert absent.stage is entry.Stage.THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS

    # And something neither, which must not be filed as either -- a reader that changed under
    # us is a third thing, and calling it AccessDenied would send somebody to write a policy.
    other = resolve_with_a_reader_that_raises(monkeypatch, Boom("manifest is not valid JSON"))
    assert other.stage is entry.Stage.THE_READER_FAILED_IN_SOME_OTHER_WAY


def test_a_denial_wrapped_in_the_readers_own_exception_is_still_a_denial(monkeypatch):
    # The reader does not re-raise botocore's errors bare; it raises its own with the original
    # attached. Reading only the outermost message would file every denial as unrecognised.
    wrapped = Boom("could not read the seal")
    wrapped.__cause__ = Boom("AccessDenied")
    assert (
        resolve_with_a_reader_that_raises(monkeypatch, wrapped).stage
        is entry.Stage.THE_ROLE_MAY_NOT_READ_THE_CORPUS
    )


def test_every_stage_has_a_number_of_its_own_and_none_collides_with_the_shell():
    """Mutation: number a stage 127, or reuse one.

    126, 127 and 128+n belong to the shell and the signal convention -- "cannot execute",
    "not found", "killed by signal n" -- and a stage sharing one of those is a stage that
    reads as an infrastructure failure forever.
    """
    numbers = [int(stage) for stage in entry.Stage]
    assert len(numbers) == len(set(numbers))
    assert all(64 <= number <= 78 for number in numbers)


def test_the_stage_survives_the_boundary_that_turns_it_into_an_exit_status(monkeypatch, capsys):
    """Mutation: let main's SystemExit reach the interpreter directly.

    ``SystemExit("a message")`` exits 1 and prints the message, which is exactly the
    indistinguishable failure this exists to end. The number only appears if something turns
    the refusal into one.
    """
    for name in ("EDULLM_DATASET_ID", "EDULLM_DATASET_VERSION", "EDULLM_DATASET_TOKENIZER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("EDULLM_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)
    monkeypatch.setattr(sys, "argv", ["train_on_corpus", "some-run"])

    assert entry.cli() == int(entry.Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT)
    printed = capsys.readouterr().err
    assert "EDULLM_DATASET_ID" in printed
    assert "edullm-stage: THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT" in printed


def test_a_diagnostic_that_cannot_reach_wandb_does_not_replace_the_error_it_reports(monkeypatch):
    """Mutation: let the reporter's own failure propagate.

    W&B is reached over a network, and a container broken enough to die in startup may be
    broken in exactly that way. A reporter that raises turns "the role cannot read the
    corpus" into "connection refused", which is a worse answer than no answer.
    """
    import types

    monkeypatch.setenv("EDULLM_WANDB_PROJECT", "edullm-platform-smoke")
    exploding: Any = types.ModuleType("wandb")

    def refuse(*args, **kwargs):
        raise RuntimeError("no route to host")

    exploding.init = refuse
    exploding.run = None
    monkeypatch.setitem(sys.modules, "wandb", exploding)

    entry.leave_the_reason_in_wandb(
        run_name="run_x", stage=entry.Stage.THE_ROLE_MAY_NOT_READ_THE_CORPUS, explanation="denied"
    )


def test_nothing_is_sent_to_wandb_when_the_platform_named_no_project(monkeypatch):
    # Running the image by hand must not fail on a missing WANDB_API_KEY, which is the same
    # reason the trainer's own callback is enabled only when the project is set.
    import types

    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)
    tripwire: Any = types.ModuleType("wandb")

    def never(*args, **kwargs):
        raise AssertionError("W&B was reached without a project")

    tripwire.init = never
    tripwire.run = None
    monkeypatch.setitem(sys.modules, "wandb", tripwire)

    entry.leave_the_reason_in_wandb(
        run_name="run_x", stage=entry.Stage.TRAINING_ITSELF_FAILED, explanation="whatever"
    )


class FakeModel:
    def __init__(self, parameters):
        self._parameters = parameters

    def parameters(self):
        return self._parameters


class FakeTrainModule:
    def __init__(self, parameters):
        self.model = FakeModel(parameters)


class FakeTrainer:
    def __init__(self, parameters, step):
        self.train_module = FakeTrainModule(parameters)
        self.global_step = step


@dataclass
class FakeOptions:
    run_name: str = "run_0"
    save_folder: str = "s3://bucket/teams/platform/runs/run_0/checkpoints/"


@dataclass
class FakeConfig:
    dataset_id: str = "pretrain/regmix-10b"
    dataset_version: str = "v1"


class FakeParameter:
    def __init__(self, count):
        self._count = count

    def numel(self):
        return self._count


def test_the_first_and_last_loss_are_kept_and_the_ones_between_are_not():
    """The summary reports both ends. Steps with no loss in their metrics are ignored."""
    watcher = entry.LossWatcher()

    watcher.log_metrics(1, {"throughput/device/TPS": 1000.0})
    watcher.log_metrics(2, {"train/CE loss": 6.9})
    watcher.log_metrics(3, {"train/CE loss": 6.5})
    watcher.log_metrics(4, {"train/CE loss": 6.1})

    assert watcher.first == 6.9
    assert watcher.last == 6.1


def test_the_summary_is_one_json_object_carrying_what_only_this_process_knows(capsys):
    """The platform reads this back out of the log stream, so it has to parse on its own."""
    import json

    watcher = entry.LossWatcher()
    watcher.log_metrics(1, {"train/CE loss": 6.9})
    watcher.log_metrics(2, {"train/CE loss": 6.1})

    entry.summarise(
        opts=FakeOptions(),
        config=FakeConfig(),
        trainer=FakeTrainer([FakeParameter(100), FakeParameter(90)], step=50),
        losses=watcher,
        seconds=12.5,
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["parameters"] == 190
    assert printed["steps"] == 50
    assert printed["first_loss"] == 6.9
    assert printed["last_loss"] == 6.1
    assert printed["seconds"] == 12.5
    assert printed["dataset_id"] == "pretrain/regmix-10b"
    assert printed["checkpoint_uri"].endswith("/checkpoints/")


def test_a_summary_is_printed_even_when_no_step_reported_a_loss(capsys):
    """A run that printed nothing cannot be told apart from one that never started."""
    import json

    entry.summarise(
        opts=FakeOptions(),
        config=FakeConfig(),
        trainer=FakeTrainer([FakeParameter(1)], step=0),
        losses=entry.LossWatcher(),
        seconds=0.5,
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["first_loss"] is None
    assert printed["last_loss"] is None


def test_the_config_print_names_how_many_shards_rather_than_all_of_them(monkeypatch):
    """olmo-150b-dolma2 resolves to 6,851 objects and the dtype must stay readable."""
    printed = []
    monkeypatch.setattr(entry.rich, "print", lambda value: printed.append(value))
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifest(
                paths=[f"s3://edullm-data/x/v1/tokens/train-{n:05}.u32le.bin" for n in range(9)]
            ),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=s3://outputs/teams/platform/runs/a-run-id/checkpoints/",
        ]
    )
    config = entry.build_config(opts, overrides)
    paths = list(config.dataset.paths)
    entry.show(config)

    assert len(printed) == 1
    assert printed[0].dataset.paths == [f"<{len(paths)} objects>"]
    # The config itself is untouched, because the run trains on it after this prints.
    assert list(config.dataset.paths) == paths


def test_the_wandb_url_is_read_while_the_run_still_has_one(monkeypatch):
    """WandBCallback.post_train finishes the run, so reading it in summarise gets None."""
    import types

    watcher = entry.LossWatcher()
    fake = types.SimpleNamespace(run=types.SimpleNamespace(url="https://wandb.ai/o/p/runs/abc"))
    monkeypatch.setitem(sys.modules, "wandb", fake)

    watcher.log_metrics(1, {"train/CE loss": 6.9})
    assert watcher.wandb_url == "https://wandb.ai/o/p/runs/abc"

    # Once the run is finished the url is kept rather than overwritten with a blank.
    fake.run = None
    watcher.log_metrics(2, {"train/CE loss": 6.1})
    assert watcher.wandb_url == "https://wandb.ai/o/p/runs/abc"


def test_a_run_with_no_wandb_reports_a_blank_url_rather_than_failing(monkeypatch):
    watcher = entry.LossWatcher()
    monkeypatch.setitem(sys.modules, "wandb", None)

    watcher.log_metrics(1, {"train/CE loss": 6.9})

    assert watcher.wandb_url == ""
    assert watcher.first == 6.9
