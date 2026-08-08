"""What :func:`validate_precision_support` refuses, and what it lets through.

Every card is faked. The decision rests on the compute capability the driver reports and on
nothing a container, a CUDA version or a torch build could change, so patching the three calls
that read it is the whole of the hardware -- and it means both directions of the refusal can be
proven on a laptop and in CI, on the cards that matter rather than on whichever one is present.
"""

from typing import Optional, Tuple

import pytest
import torch

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
    validate_precision_support,
)
from olmo_core.utils import get_devices_without_bfloat16

T4 = ("Tesla T4", (7, 5))
A10G = ("NVIDIA A10G", (8, 6))
L4 = ("NVIDIA L4", (8, 9))
H100 = ("NVIDIA H100 80GB HBM3", (9, 0))


def on_a(card: str, capability: Tuple[int, int], *, count: int, monkeypatch):
    """
    Answer as a host carrying ``count`` of this card would, without one being present.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: card)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)


def train_module_config(param_dtype: DType = DType.bfloat16) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=256,
        max_sequence_length=256,
        optim=AdamWConfig(),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=param_dtype, reduce_dtype=DType.float32
        ),
    )


def single_device_config(precision: Optional[DType]) -> TransformerTrainModuleConfig:
    """The same config with and without the request, and no parallelism to need a world.

    ``autocast_precision`` rather than ``dp_config.param_dtype`` only because a data parallel
    config refuses to build outside a process group, and these two tests are about what
    ``build`` does before it gets that far.
    """
    return TransformerTrainModuleConfig(
        rank_microbatch_size=256,
        max_sequence_length=256,
        optim=AdamWConfig(),
        autocast_precision=precision,
    )


def model(dtype: DType = DType.float32):
    return TransformerConfig.llama_like(
        d_model=64,
        vocab_size=128,
        n_layers=2,
        n_heads=2,
        dtype=dtype,
        feed_forward=FeedForwardConfig(hidden_size=128, bias=False),
    ).build(init_device="meta")


def test_bfloat16_is_refused_on_a_card_whose_silicon_has_none(monkeypatch):
    """Mutation: return without raising when the devices lack the format.

    The direction the check exists for. Nothing in the config names a T4 and nothing in the
    hardware names a dtype, so this is the only place the two facts meet.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)

    with pytest.raises(OLMoConfigurationError, match="dp_config.param_dtype"):
        validate_precision_support(train_module_config())


def test_bfloat16_is_allowed_on_every_card_that_has_it(monkeypatch):
    """Mutation: raise whenever the config asks for bfloat16, whatever the hardware says.

    The direction that matters more, because there is no waiver past this refusal: a false
    positive does not cost somebody a message they can ignore, it stops a run that would have
    worked. Ampere, Ada and Hopper all have bfloat16, and so does a host with no CUDA at all,
    which is a laptop and is also this test suite.
    """
    config = train_module_config()

    for card, capability in (A10G, L4, H100):
        on_a(card, capability, count=8, monkeypatch=monkeypatch)
        assert get_devices_without_bfloat16() == []
        validate_precision_support(config)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    validate_precision_support(config)


def test_float16_on_a_card_without_bfloat16_is_left_alone(monkeypatch):
    """Mutation: refuse any half precision on a pre-Ampere card, not just bfloat16.

    fp16 on a T4 is a real recipe that people run -- Turing has the format, it is bfloat16 it
    lacks -- so widening this to "narrow dtypes are unsafe here" would refuse working jobs.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)

    for chosen in (DType.float16, DType.float32):
        validate_precision_support(train_module_config(chosen))


def test_a_rocm_build_is_left_alone_because_its_numbers_mean_something_else(monkeypatch):
    """Mutation: compare the capability without checking for HIP first.

    AMD reports its own architecture numbering through the same call, where 7.5 is not Turing
    and the comparison is meaningless.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)
    monkeypatch.setattr(torch.version, "hip", "6.2.0")

    validate_precision_support(train_module_config())


def test_a_device_that_cannot_be_read_gets_out_of_the_way(monkeypatch, caplog):
    """Mutation: let whatever the driver raises propagate.

    This now runs in front of every train module built anywhere in the repository. Missing a
    Turing card costs what today already costs; raising on a host it did not anticipate stops
    runs that were fine, which is strictly worse and is the failure a startup check is most
    likely to have.
    """

    def unreadable(index=None):
        raise RuntimeError("no CUDA-capable device is detected")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unreadable)

    validate_precision_support(train_module_config())
    assert "bfloat16 check is not running" in caplog.text


def test_a_field_that_merely_holds_the_word_is_not_a_precision_request(monkeypatch):
    """Mutation: match the value anywhere in the config and ignore what the key is called.

    There is no waiver past this refusal, so a false positive is a run nobody can start. A
    string field whose value happens to be "bfloat16" is not somebody asking a T4 for bfloat16,
    and the name of the field is what tells them apart.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)

    config = train_module_config(DType.float32)
    config.load_key_mapping = {"bfloat16": "bfloat16"}

    validate_precision_support(config)


def test_a_model_already_in_bfloat16_is_caught_although_no_field_says_so(monkeypatch):
    """Mutation: read the config only and ignore the model.

    ``TransformerConfig.dtype`` is reachable from a command line and is not a field of any
    train module config, so a config walk on its own would let a bfloat16 model through onto a
    card that cannot multiply one.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)

    with pytest.raises(OLMoConfigurationError, match="the model's parameters"):
        validate_precision_support(train_module_config(DType.float32), model(DType.bfloat16))


def test_the_refusal_names_the_card_the_reason_and_both_ways_out(monkeypatch):
    """Mutation: say "bfloat16 is not supported on this device" and stop.

    Whoever reads this has one question: what do I do now. The card and the capability say it
    is the hardware rather than the image, so nobody goes looking for a driver, and the two
    remedies are the whole of what can be done about it.

    ``is_bf16_supported`` is named because a reviewer who does not know it returns ``True`` on
    this card will conclude the check is redundant and delete it.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)

    with pytest.raises(OLMoConfigurationError) as refused:
        validate_precision_support(train_module_config())

    message = str(refused.value)
    assert "Tesla T4" in message and "7.5" in message
    assert "is_bf16_supported" in message
    assert "float32" in message and "float16" in message
    assert "8.0 or newer" in message


def test_building_a_train_module_is_what_applies_the_check(monkeypatch):
    """Mutation: drop the ``validate_precision_support`` call from ``build``.

    The reason the check sits in the library at all. An entry point does not have to know this
    exists, does not have to call it, and does not have to be the one script that used to: it
    builds a train module, which every training and evaluation entry point in this repository
    does, and the refusal arrives before the model is placed on a device or a step is taken.
    """
    built = model()
    on_a(*T4, count=8, monkeypatch=monkeypatch)

    with pytest.raises(OLMoConfigurationError, match="Tesla T4"):
        single_device_config(DType.bfloat16).build(built, device=torch.device("cpu"))


def test_building_a_train_module_on_a_card_that_has_bfloat16_is_untouched(monkeypatch):
    """Mutation: raise from ``build`` unconditionally rather than only when refusing.

    The same call, the same request, one card newer. It has to reach a train module, which is
    what makes the test above a guard rather than a wall.
    """
    built = model()
    on_a(*A10G, count=8, monkeypatch=monkeypatch)

    train_module = single_device_config(DType.bfloat16).build(built, device=torch.device("cpu"))
    assert isinstance(train_module, TransformerTrainModule)
