from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.generate.generation_module.transformer import (
    TransformerGenerationModule,
)
from olmo_core.generate.attention.backends import AttentionBackendName
from transformers import AutoTokenizer
import torch

class OlmoCoreModel():
    def __init__(self, model_name: str, tokenizer_name = "allenai/dolma2-tokenizer"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"

        # use the default if possible, append if necessary
        stop_token_ids = []
        self.stop_token_ids = stop_token_ids

        self.generation_config = GenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=10_000,
            max_new_tokens=1,
            temperature=1.0,
            top_p=1.0,
            do_sample=False,
            use_cache=True,
            stop_token_ids=self.stop_token_ids,
        )

        print(f"Loading OlmoCoreModel from {model_name}")


        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        generation_module = TransformerGenerationModule.from_checkpoint(
            checkpoint_dir=model_name,
            generation_config=self.generation_config,
            device=self.device,
            dtype=torch.bfloat16,
            attention_backend=AttentionBackendName.torch,
        )

        print("Initialized OlmoCoreModel")

        self.disable_prefill = False

def __main__():
    pass

    # TODO: get some text (maybe govreport docs?) and tokenize 10 examples
    from datasets import load_dataset


    # TODO: then, run inference with output_attention set to true 
