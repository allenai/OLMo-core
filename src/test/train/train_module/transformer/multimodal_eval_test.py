import contextlib

import torch

from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.train.train_module import (
    MultimodalOLMoDDPTrainModule,
    OLMoDDPTrainModule,
)


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
            self.lm = self

        @staticmethod
        def routed_blocks():
            yield from ()

        def eval(self):
            self.eval_calls += 1

        def reset_auxiliary_metrics(self):
            self.reset_calls += 1

    model_part = ModelPart()
    object.__setattr__(train_module, "model_parts", [model_part])
    captured = {}

    def model_forward(input_ids, labels=None, **kwargs):
        captured.update(input_ids=input_ids, labels=labels, kwargs=kwargs)
        loss = torch.tensor(3.0, requires_grad=True)
        return LMOutputWithLoss(None, loss, loss, None)

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
    assert not output.loss.requires_grad
    assert not output.ce_loss.requires_grad
    assert captured["labels"] is explicit_labels
    assert captured["kwargs"]["loss_reduction"] == "sum"
    assert captured["kwargs"]["return_logits"] is False
    assert captured["kwargs"]["response_logits_only"] is True
    assert batch["input_ids"].shape == (1, 3)
    assert batch["labels"] is explicit_labels
    assert model_part.eval_calls == 1
    assert model_part.reset_calls == 1


def test_multimodal_text_eval_delegates_to_standard_full_logits_path(monkeypatch):
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    captured = {}
    loss = torch.ones(1, 2, requires_grad=True)
    expected = LMOutputWithLoss(torch.ones(1, 2, 3, requires_grad=True), loss, loss, None)

    def standard_eval(self, batch, labels=None):
        captured.update(self=self, batch=batch, labels=labels)
        return expected

    monkeypatch.setattr(OLMoDDPTrainModule, "eval_batch", standard_eval)
    batch = {"input_ids": torch.tensor([[1, 2]])}
    labels = torch.tensor([[2, -100]])

    output = train_module.eval_batch(batch, labels=labels)

    assert torch.equal(output.logits, expected.logits)
    assert not output.logits.requires_grad
    assert not output.loss.requires_grad
    assert not output.ce_loss.requires_grad
    assert captured == {"self": train_module, "batch": batch, "labels": labels}


def test_response_only_logits_are_limited_to_loss_mask_batches(monkeypatch):
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module.response_logits_only = True

    def standard_prepare(self, batch, labels=None):
        return (
            batch["input_ids"],
            labels,
            {key: value for key, value in batch.items() if key != "input_ids"},
        )

    monkeypatch.setattr(OLMoDDPTrainModule, "_prepare_batch", standard_prepare)
    input_ids = torch.tensor([[1, 2]])

    _, _, text_kwargs = train_module._prepare_batch({"input_ids": input_ids})
    _, _, multimodal_kwargs = train_module._prepare_batch(
        {"input_ids": input_ids, "loss_masks": torch.ones_like(input_ids)}
    )

    assert "response_logits_only" not in text_kwargs
    assert multimodal_kwargs["response_logits_only"] is True


def test_multimodal_olmo_ddp_eval_uses_training_routing_dispatch_only():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)

    class Block:
        training = False

    block = Block()

    class LM:
        @staticmethod
        def routed_blocks():
            yield block

    class ModelPart:
        lm = LM()

    object.__setattr__(train_module, "model_parts", [ModelPart()])

    with torch.no_grad():
        with train_module._multimodal_eval_batch_context():
            assert block.training is True
            assert torch.is_grad_enabled()

    assert block.training is False


def test_multimodal_olmo_ddp_text_eval_forces_eager_with_grad_enabled(monkeypatch):
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    events = []

    @contextlib.contextmanager
    def set_stance(stance):
        events.append(("enter", stance))
        try:
            yield
        finally:
            events.append(("exit", stance))

    monkeypatch.setattr(torch.compiler, "set_stance", set_stance)

    with torch.no_grad():
        with train_module._eval_batch_context():
            assert torch.is_grad_enabled()

    assert events == [("enter", "force_eager"), ("exit", "force_eager")]
