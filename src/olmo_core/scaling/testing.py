from typing import List, Optional, Union

from mup import get_shapes, make_base_shapes
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.data import TokenizerConfig
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.utils import get_default_device, prepare_cli_environment, seed_all


# from olmo_core.model import OLMo



def load_mu_model(config: TransformerConfig):
    config.use_mup = True
    tokenizer_config = TokenizerConfig.dolma2()
    device = get_default_device()
    return TransformerConfig.olmo2_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        compile=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
    ).build(device=device)

tokenizer_config = TokenizerConfig.dolma2()
device = get_default_device()
config_use = TransformerConfig.olmo2_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        compile=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
    )

def save_base_shapes(
    model_config: Union[str, TransformerConfig], dims_to_scale: int = 768
):
    tokenizer_config = TokenizerConfig.dolma2()
    config_use = TransformerConfig.olmo2_7B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            compile=True,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.hsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            ),
        )

    base_shapes = get_shapes(load_mu_model(config_use))
    setattr(model_config, dims_to_scale, getattr(model_config, dims_to_scale) * 2)

    delta_shapes = get_shapes(load_mu_model(config_use))
    make_base_shapes(base_shapes, delta_shapes)#, savefile=output_path)

x = save_base_shapes(config_use)
print(x)
