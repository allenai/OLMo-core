"""Convert an OLMo-core Qwen3 MoE checkpoint to native Hugging Face format."""

from olmo_core.nn.moe.v2.qwen_hf_export import main

if __name__ == "__main__":
    main()
