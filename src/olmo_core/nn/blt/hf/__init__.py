from transformers import AutoModelForCausalLM, AutoModel, AutoConfig

from olmo_core.nn.blt.hf.configuration_bolmo import BolmoConfig
from olmo_core.nn.blt.hf.modeling_bolmo import BolmoForCausalLM, BolmoModel

AutoConfig.register("bolmo", BolmoConfig)
AutoModelForCausalLM.register(BolmoConfig, BolmoForCausalLM)
AutoModel.register(BolmoConfig, BolmoModel)