import torch

from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.train.train_module import MultimodalOLMoDDPTrainModule


def test_multimodal_eval_uses_explicit_labels_and_summed_weighted_loss():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module._cp_config = None
    train_module._tp_config = None
    train_module._pp_config = None
    train_module.response_logits_only = True
    train_module.label_ignore_index = -100

    class ModelPart:
        def __init__(self):
            self.eval_calls = 0
            self.reset_calls = 0

        def eval(self):
            self.eval_calls += 1

        def reset_auxiliary_metrics(self):
            self.reset_calls += 1

    model_part = ModelPart()
    object.__setattr__(train_module, "model_parts", [model_part])
    captured = {}

    def model_forward(input_ids, labels=None, **kwargs):
        captured.update(input_ids=input_ids, labels=labels, kwargs=kwargs)
        return LMOutputWithLoss(None, torch.tensor(3.0), torch.tensor(3.0), None)

    train_module.model_forward_no_pipeline = model_forward
    explicit_labels = torch.tensor([[7, 8, -100]])
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": explicit_labels,
        "loss_masks": torch.tensor([[0.0, 1.0, 0.0]]),
        "router_token_mask": torch.tensor([[True, True, False]]),
    }

    output = train_module.eval_batch(batch, labels=torch.tensor([[2, 3, -100]]))

    assert output.ce_loss.item() == 3.0
    assert captured["labels"] is explicit_labels
    assert captured["kwargs"]["loss_reduction"] == "sum"
    assert captured["kwargs"]["return_logits"] is False
    assert captured["kwargs"]["response_logits_only"] is True
    assert batch["input_ids"].shape == (1, 3)
    assert batch["labels"] is explicit_labels
    assert model_part.eval_calls == 1
    assert model_part.reset_calls == 1
