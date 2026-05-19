from olmo_core.train.callbacks.checkpointer import CheckpointerCallback
from olmo_core.train.callbacks.comet import CometCallback
from olmo_core.train.callbacks.wandb import WandBCallback


def test_external_loggers_initialize_before_checkpointer():
    # Pre-train checkpoint saves can flush already-recorded metrics, so external
    # metric sinks must be initialized before the checkpointer runs pre_train().
    assert WandBCallback.priority > CheckpointerCallback.priority
    assert CometCallback.priority > CheckpointerCallback.priority


class MockWandB:
    run = object()

    def __init__(self):
        self.finish_calls = []

    def finish(self, **kwargs):
        self.finish_calls.append(kwargs)


def test_wandb_finalizes_on_close_not_post_train():
    # Final checkpoint saves happen from CheckpointerCallback.post_train(), after
    # higher-priority callbacks have run their own post_train(). Keep W&B open
    # until close(), because close() runs after final checkpoint metrics drain.
    wandb = MockWandB()
    callback = WandBCallback()
    callback._wandb = wandb

    callback.post_train()
    assert not callback.finalized
    assert wandb.finish_calls == []

    callback.close()
    assert callback.finalized
    assert wandb.finish_calls == [{"exit_code": 0, "quiet": True}]
