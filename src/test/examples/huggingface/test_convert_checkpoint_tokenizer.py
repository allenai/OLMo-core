"""Tokenizer precedence for convert_checkpoint_to_hf (allenai/OLMo-core#609)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.hf.convert_checkpoint import convert_checkpoint_to_hf
from olmo_core.nn.transformer.config import TransformerConfig


def test_explicit_tokenizer_id_overrides_sibling_tokenizer_dir(tmp_path: Path):
    """Explicit tokenizer_id must win when a sibling tokenizer/ directory exists."""
    checkpoint_dir = tmp_path / "run" / "step1000"
    (checkpoint_dir / "model_and_optim").mkdir(parents=True)
    sibling = checkpoint_dir.parent / "tokenizer"
    sibling.mkdir()
    (sibling / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    tokenizer_config = TokenizerConfig.dolma2()
    transformer_config = TransformerConfig.olmo2_190M(
        tokenizer_config.padded_vocab_size(), n_layers=2
    )
    model = transformer_config.build()
    # Minimal CPU state for the convert path that accepts model_state_dict.
    model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    fake_tok = MagicMock()
    fake_tok.model_max_length = 256
    fake_tok.pad_token_id = tokenizer_config.pad_token_id
    fake_tok.bos_token_id = tokenizer_config.bos_token_id
    fake_tok.eos_token_id = tokenizer_config.eos_token_id

    override_id = "allenai/olmo-test-tokenizer-override"
    output_dir = tmp_path / "hf-out"

    with (
        patch(
            "olmo_core.nn.hf.convert_checkpoint.AutoTokenizer.from_pretrained",
            return_value=fake_tok,
        ) as from_pretrained,
        patch("olmo_core.nn.hf.convert_checkpoint.save_hf_model"),
        patch("olmo_core.nn.hf.convert_checkpoint.validate_conversion"),
    ):
        convert_checkpoint_to_hf(
            original_checkpoint_path=str(checkpoint_dir),
            output_path=str(output_dir),
            transformer_config_dict=transformer_config.as_config_dict(),
            tokenizer_config_dict=tokenizer_config.as_config_dict(),
            model_state_dict=model_state_dict,
            tokenizer_id=override_id,
            max_sequence_length=256,
            validate=False,
            dtype=None,
            device=torch.device("cpu"),
        )

    assert from_pretrained.call_count >= 1
    assert from_pretrained.call_args_list[0].args[0] == override_id
    fake_tok.save_pretrained.assert_called()
