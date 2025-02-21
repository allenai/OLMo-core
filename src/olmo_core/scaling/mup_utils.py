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
from olmo_core.utils import get_default_device



def load_mu_model(config: TransformerConfig):
    config.use_mup = True
    device = get_default_device()
    return config.build(device=device)


def save_base_shapes(output_path: str, d_model: int = 768):

    tokenizer_config = TokenizerConfig.dolma2()
    config_use = TransformerConfig.olmo2_7B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            compile=True,
            d_model = d_model,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.hsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            ),
        )
    
    
    base_shapes = get_shapes(load_mu_model(config_use))

    config_scaled = TransformerConfig.olmo2_7B(
        vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
        compile=True,
        d_model=d_model * 2,  # Double d_model
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
    )

    delta_shapes = get_shapes(load_mu_model(config_scaled))
    make_base_shapes(base_shapes, delta_shapes, savefile=output_path)
