from typing import List, Optional, Union
import os
from mup import get_shapes, make_base_shapes
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.data import TokenizerConfig
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)
import torch
from olmo_core.distributed.utils import OLMO_LOCAL_WORLD_SIZE_ENV_VAR

def save_base_shapes(output_path: str, d_model: int = 768):
    os.environ[OLMO_LOCAL_WORLD_SIZE_ENV_VAR] = "1"
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

    tokenizer_config = TokenizerConfig.dolma2()
    config_use = TransformerConfig.olmo2_190M(
        mup=True,
        vocab_size=tokenizer_config.padded_vocab_size(),
        compile=True,
        d_model=d_model,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
    )
    # config_use.use_mup = True
    device = torch.device('cuda')
    base_model = config_use.build(device=device)
    base_shapes = get_shapes(base_model)
    config_scaled = TransformerConfig.olmo2_190M(
        mup=True,
        vocab_size=tokenizer_config.padded_vocab_size(),
        compile=True,
        d_model=d_model * 2,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
    )
    # config_scaled.use_mup = True
    scaled_model = config_scaled.build(device=device)
    delta_shapes = get_shapes(scaled_model)
    make_base_shapes(base_shapes, delta_shapes, savefile=output_path)
