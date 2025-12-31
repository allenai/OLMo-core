import torch

from olmo_core.train.callbacks import DataMixtureMonitorCallback


class FakeTrainer:
    def __init__(self):
        # metric_name -> list of values
        self.logged = {}

    def record_metric(self, name, value, **kwargs):
        self.logged.setdefault(name, []).append(value)


def test_data_mixture_monitor_basic_counts():
    """
    3 sequences, length 4:
    - source A: 2 sequences -> 8 tokens
    - source B: 1 sequence -> 4 tokens
    """
    cb = DataMixtureMonitorCallback(log_interval=1)
    trainer = FakeTrainer()
    cb.trainer = trainer

    input_ids = torch.ones(3, 4, dtype=torch.long)

    metadata = [
        {"source": "A"},
        {"source": "B"},
        {"source": "A"},
    ]

    batch = {
        "input_ids": input_ids,
        "metadata": metadata,
    }

    cb.pre_train()
    cb.pre_step(batch)

    # Check token counts
    tokens_A = trainer.logged["data_mixture/tokens/source=A"][-1]
    tokens_B = trainer.logged["data_mixture/tokens/source=B"][-1]

    assert tokens_A == 8.0
    assert tokens_B == 4.0

    # Check sequence counts
    seqs_A = trainer.logged["data_mixture/sequences/source=A"][-1]
    seqs_B = trainer.logged["data_mixture/sequences/source=B"][-1]

    assert seqs_A == 2.0
    assert seqs_B == 1.0

    # Check token share
    share_A = trainer.logged["data_mixture/token_share/source=A"][-1]
    share_B = trainer.logged["data_mixture/token_share/source=B"][-1]

    assert abs(share_A - 8 / 12) < 1e-6
    assert abs(share_B - 4 / 12) < 1e-6


def test_data_mixture_monitor_respects_log_interval():
    """
    With log_interval=2, the first step should not log anything.
    """
    cb = DataMixtureMonitorCallback(log_interval=2)
    trainer = FakeTrainer()
    cb.trainer = trainer

    input_ids = torch.ones(1, 4, dtype=torch.long)
    metadata = [{"source": "A"}]
    batch = {"input_ids": input_ids, "metadata": metadata}

    cb.pre_train()
    cb.pre_step(batch)  # step 1 -> should NOT log yet

    # No metrics expected yet
    assert trainer.logged == {}

    cb.pre_step(batch)  # step 2 -> should log now

    assert "data_mixture/tokens/source=A" in trainer.logged
    assert trainer.logged["data_mixture/tokens/source=A"][-1] == 8.0  # 2 steps * 4 tokens
