from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.generate.generation_module.transformer import (
    TransformerGenerationModule,
)
from olmo_core.nn.attention.backend import AttentionBackendName
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



        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.generation_module = TransformerGenerationModule.from_checkpoint(
            checkpoint_dir=model_name,
            generation_config=self.generation_config,
            device=self.device,
            dtype='bfloat16',
            attention_backend=AttentionBackendName.torch,
        )


        self.disable_prefill = False


from sink_inference import OlmoCoreModel
for model_name in [ "/weka/oe-training-default/ai2-llm/checkpoints/amandab/olmo29_7b_140B-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preannealv2-f1a781d3/step2385", \
        "/weka/oe-training-default/ai2-llm/checkpoints/amandab/olmo28_140B_lc_64k-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preannealv2-988d396f/step2385/",
        "/weka/oe-training-default/ai2-llm/checkpoints/amandab/olmo25_140B_lc_64k-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preannealv2-b9609b3f/step2385",
        "/weka/oe-training-default/ai2-llm/checkpoints/amandab/llamalike_140B_lc_64k-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preannealv2-1623f603/step2385/",
        "/weka/oe-training-default/ai2-llm/checkpoints/amandab/llamalike_140B_lc_64k-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preannealv2-ffc378a3/step2385/"]:
        #"/weka/oe-training-default/ai2-llm/checkpoints/amandab/Meta-Llama-3-8B-Base-redone",
        #"/weka/oe-training-default/ai2-llm/checkpoints/amandab/Marin-8B-Base",
        #"/weka/oe-training-default/ai2-llm/checkpoints/amandab/olmo25_float8_rerun_for_init_140B_lc_64k-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preannealv2-d65cb2d/step2385/"]:
   
    try:
        model = OlmoCoreModel(model_name=model_name)
    except Exception as e:
        print("error on model load:", model_name)
        print(e.message)
        continue 

    with open('random_text_3.txt', 'r') as f:
        input_text = f.read()

    inputs = model.tokenizer([input_text], return_tensors='pt', padding=False)
    print(model_name)
    model.generation_module.model_forward(**inputs)

