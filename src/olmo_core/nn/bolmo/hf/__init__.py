from transformers import AutoModelForCausalLM, AutoModel, AutoConfig

from olmo_core.nn.bolmo.hf.configuration_bolmo import BolmoConfig
from olmo_core.nn.bolmo.hf.modeling_bolmo import BolmoForCausalLM, BolmoModel

AutoConfig.register("bolmo", BolmoConfig)
AutoModelForCausalLM.register(BolmoConfig, BolmoForCausalLM)
AutoModel.register(BolmoConfig, BolmoModel)
BolmoConfig.register_for_auto_class("AutoConfig")
BolmoForCausalLM.register_for_auto_class("AutoModelForCausalLM")
BolmoModel.register_for_auto_class("AutoModel")