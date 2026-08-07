from types import SimpleNamespace

from olmo_core.train.callbacks.wandb import WANDB_API_KEY_ENV_VAR, WandBCallback


class MockWandB:
    def __init__(self, run_id: str = "run-123"):
        self.run = SimpleNamespace(id=run_id, path=f"entity/project/{run_id}")
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


def _callback(tmp_path, wandb, *, step: int = 25) -> WandBCallback:
    callback = WandBCallback(
        name="stage1",
        project="molmo2-stage1",
        entity="test-entity",
        auto_resume=True,
    )
    callback._wandb = wandb
    callback.trainer = SimpleNamespace(
        work_dir=tmp_path,
        global_step=step,
        checkpoint_loaded=False,
    )
    return callback


def test_wandb_run_id_and_checkpoint_step_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv(WANDB_API_KEY_ENV_VAR, "test-key")
    callback = _callback(tmp_path, MockWandB(), step=25)
    callback.pre_train()

    state = callback.state_dict()

    assert state == {
        "run_id": "run-123",
        "step": 25,
        "name": "stage1",
        "project": "molmo2-stage1",
        "entity": "test-entity",
    }


def test_wandb_auto_resume_rewinds_to_checkpoint_step(tmp_path, monkeypatch):
    monkeypatch.setenv(WANDB_API_KEY_ENV_VAR, "test-key")
    wandb = MockWandB()
    callback = _callback(tmp_path, wandb, step=25)
    callback.load_state_dict(
        {
            "run_id": "run-123",
            "step": 20,
            "name": "stage1",
            "project": "molmo2-stage1",
            "entity": "test-entity",
        }
    )

    callback.pre_train()

    assert wandb.init_calls[0]["resume_from"] == "run-123?_step=20"
    assert wandb.init_calls[0]["allow_val_change"] is True
    assert "resume" not in wandb.init_calls[0]


def test_wandb_auto_resume_rejects_different_run_identity(tmp_path, monkeypatch):
    monkeypatch.setenv(WANDB_API_KEY_ENV_VAR, "test-key")
    wandb = MockWandB(run_id="new-run")
    callback = _callback(tmp_path, wandb)
    callback.load_state_dict(
        {
            "run_id": "old-run",
            "step": 20,
            "name": "different-name",
            "project": "molmo2-stage1",
            "entity": "test-entity",
        }
    )

    callback.pre_train()

    assert "resume_from" not in wandb.init_calls[0]
    assert callback.state_dict()["run_id"] == "new-run"


def test_wandb_explicit_run_id_migrates_legacy_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv(WANDB_API_KEY_ENV_VAR, "test-key")
    wandb = MockWandB(run_id="legacy-run")
    callback = _callback(tmp_path, wandb, step=4000)
    callback.run_id = "legacy-run"
    callback.trainer.checkpoint_loaded = True
    callback.load_state_dict({})

    callback.pre_train()

    assert wandb.init_calls[0]["resume_from"] == "legacy-run?_step=4000"
