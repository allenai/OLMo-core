from typing import cast
from unittest.mock import Mock, call, patch

from olmo_core.train.callbacks.checkpointer import (
    CheckpointerCallback,
    CheckpointRemovalStrategy,
)


def _build_callback(
    *,
    step: int,
    checkpoints,
    permanent_checkpoints=None,
    max_checkpoints: int = 2,
    remove: CheckpointRemovalStrategy = cast(
        CheckpointRemovalStrategy, CheckpointRemovalStrategy.ephemeral_only
    ),
) -> tuple[CheckpointerCallback, Mock]:
    trainer = Mock()
    trainer.global_step = step
    trainer.checkpoint_loaded = True
    trainer.save_folder = "/checkpoints"
    trainer.checkpointer.process_group = None
    if permanent_checkpoints is None:
        permanent_checkpoints = checkpoints

    def find_checkpoints(_, ephemeral=None):
        if ephemeral is False:
            return permanent_checkpoints
        return checkpoints

    trainer.checkpointer.find_checkpoints.side_effect = find_checkpoints

    callback = CheckpointerCallback(
        save_interval=100,
        ephemeral_save_interval=25,
        fixed_steps=[75],
        max_checkpoints=max_checkpoints,
        remove=remove,
        save_async=False,
    )
    callback.trainer = trainer
    return callback, trainer


def _pre_train(callback: CheckpointerCallback) -> None:
    with (
        patch("olmo_core.train.callbacks.checkpointer.is_distributed", return_value=False),
        patch("olmo_core.train.callbacks.checkpointer.get_rank", return_value=0),
        patch(
            "olmo_core.train.callbacks.checkpointer.broadcast_object",
            side_effect=lambda value: value,
        ),
    ):
        callback.pre_train()


def test_pre_train_discovers_and_trims_permanent_checkpoints_deterministically():
    callback, trainer = _build_callback(
        step=350,
        checkpoints=[
            (400, "/checkpoints/step400"),
            (300, "/checkpoints/step300"),
            (300, "/checkpoints/step300"),
            (325, "/checkpoints/step325"),
            (125, "/checkpoints/step125"),
            (0, "/checkpoints/step0"),
            (75, "/checkpoints/step75"),
            (200, "/checkpoints/step200"),
        ],
        permanent_checkpoints=[
            (400, "/checkpoints/step400"),
            (300, "/checkpoints/step300"),
            (300, "/checkpoints/step300"),
            (325, "/checkpoints/step325"),
            (0, "/checkpoints/step0"),
            (75, "/checkpoints/step75"),
            (200, "/checkpoints/step200"),
        ],
    )

    _pre_train(callback)

    assert trainer.checkpointer.find_checkpoints.call_args_list == [
        call("/checkpoints", ephemeral=False),
        call("/checkpoints"),
    ]
    assert callback._checkpoints_to_remove == [
        "/checkpoints/step0",
        "/checkpoints/step75",
        "/checkpoints/step200",
    ]
    assert callback._checkpoints == ["/checkpoints/step300", "/checkpoints/step325"]
    assert callback._ephemeral_checkpoints == ["/checkpoints/step125"]


def test_resume_removes_excess_checkpoint_before_next_permanent_save():
    callback, trainer = _build_callback(
        step=300,
        checkpoints=[
            (300, "/checkpoints/step300"),
            (100, "/checkpoints/step100"),
            (200, "/checkpoints/step200"),
        ],
    )
    _pre_train(callback)

    events = []

    def remove_checkpoint(path):
        events.append(("remove", path))

    def save_checkpoint(**kwargs):
        events.append(("save", kwargs))
        return "/checkpoints/step400"

    trainer.save_checkpoint.side_effect = save_checkpoint
    trainer.global_step = 400

    with patch.object(callback, "_remove_checkpoint", side_effect=remove_checkpoint):
        callback.post_train_batch()

    assert events == [
        ("remove", "/checkpoints/step100"),
        ("save", {"ephemeral": False}),
    ]
    assert callback._checkpoints == [
        "/checkpoints/step300",
        "/checkpoints/step400",
    ]
    assert callback._checkpoints_to_remove == ["/checkpoints/step200"]


def test_remove_never_does_not_discover_or_trim_checkpoints():
    callback, trainer = _build_callback(
        step=350,
        checkpoints=[(100, "/checkpoints/step100")],
        max_checkpoints=1,
        remove=cast(CheckpointRemovalStrategy, CheckpointRemovalStrategy.never),
    )
    callback._checkpoints = ["/checkpoints/step200", "/checkpoints/step300"]

    _pre_train(callback)
    callback._trim_checkpoints()

    trainer.checkpointer.find_checkpoints.assert_not_called()
    assert callback._checkpoints == ["/checkpoints/step200", "/checkpoints/step300"]
    assert callback._checkpoints_to_remove == []
