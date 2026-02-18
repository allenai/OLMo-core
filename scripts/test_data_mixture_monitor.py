import torch

from olmo_core.train.callbacks import DataMixtureMonitorCallback


class FakeTrainer:
    def __init__(self):
        self.logged = {}

    def record_metric(self, name, value, **kwargs):
        self.logged.setdefault(name, []).append(value)
        print(f"[metric] {name} = {value}")


def main():
    cb = DataMixtureMonitorCallback(log_interval=1)

    trainer = FakeTrainer()
    cb.trainer = trainer

    # 3 sequences, length 4
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


if __name__ == "__main__":
    main()
