import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.lm_head import LMHeadConfig, LMHeadType


def test_lm_head_builder_config():
    lm_head = LMHeadConfig(name=LMHeadType.default).build(d_model=64, vocab_size=128)
    assert lm_head.w_out.bias is not None

    lm_head = LMHeadConfig(name=LMHeadType.default, bias=False).build(d_model=64, vocab_size=128)
    assert lm_head.w_out.bias is None

    with pytest.raises(OLMoConfigurationError):
        LMHeadConfig(name=LMHeadType.normalized, bias=True).build(d_model=64, vocab_size=128)
