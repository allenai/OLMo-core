from types import SimpleNamespace
from unittest.mock import MagicMock

from olmo_core.train.callbacks.beaker import BeakerCallback
from olmo_core.train.callbacks.wandb import WandBCallback


class MockBeaker:
    def __init__(self, url: str):
        self.workload = SimpleNamespace(get=lambda experiment_id: experiment_id, url=lambda _: url)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback


def test_beaker_metadata_can_change_when_wandb_run_resumes(tmp_path, monkeypatch):
    experiment_id = "new-experiment"
    experiment_url = f"https://beaker.org/ex/{experiment_id}"
    run = MagicMock()
    run.get_url.return_value = None
    wandb = WandBCallback()
    wandb._wandb = SimpleNamespace(run=run)

    callback = BeakerCallback(
        enabled=True,
        experiment_id=experiment_id,
        result_dir=str(tmp_path),
    )
    callback.trainer = SimpleNamespace(callbacks={"wandb": wandb})
    callback._update = MagicMock()

    monkeypatch.setattr("olmo_core.train.callbacks.beaker.get_rank", lambda: 0)
    monkeypatch.setattr(
        "olmo_core.launch.beaker.get_beaker_client",
        lambda: MockBeaker(experiment_url),
    )
    monkeypatch.setattr(
        "olmo_core.train.callbacks.beaker.subprocess.call", lambda *args, **kwargs: 0
    )

    callback.pre_train()

    run.config.update.assert_called_once_with(
        {
            "beaker_experiment_url": experiment_url,
            "beaker_experiment_id": experiment_id,
        },
        allow_val_change=True,
    )
