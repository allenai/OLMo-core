"""Fail-closed production-tokenizer gates for hero exports and legacy launchers."""

import hashlib
import json
from pathlib import Path

PRODUCTION_ID = "allenai/Olmo-3-7B-Think-SFT"
PRODUCTION_REVISION = "6ff857587e040d6d523a3d5f3a56e918f5401d66"
PRETOKENIZER_SHA256 = "ba2cf544ccca9022f2033fe108be1c340633f0f30f50c0dd6835bb49371ac6a1"


def require_correct_tokenizer(path, *, runtime=False):
    """Reject legacy segmentation/class routing; never repair incorrectly trained SFTs."""
    path = Path(path)
    raw = json.loads((path / "tokenizer.json").read_text())
    config = json.loads((path / "tokenizer_config.json").read_text())
    digest = hashlib.sha256(json.dumps(raw["pre_tokenizer"], sort_keys=True).encode()).hexdigest()
    if digest != PRETOKENIZER_SHA256:
        raise ValueError(
            f"Incorrect production tokenizer at {path}; retokenization/retraining required"
        )
    if config.get("tokenizer_class") not in ("TokenizersBackend", "PreTrainedTokenizerFast"):
        raise ValueError(f"Unsafe AutoTokenizer class routing at {path}")
    if runtime:
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer

        reference = Tokenizer.from_file(str(path / "tokenizer.json"))
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if tok.encode("9078563412", add_special_tokens=False) != [23505, 25505, 16546, 17]:
            raise ValueError("Incorrect inference number splitting")
        for text in ("12345678901234567890", "def f():\n    return 123456\n", "中文 123456"):
            if (
                tok.encode(text, add_special_tokens=False)
                != reference.encode(text, add_special_tokens=False).ids
            ):
                raise ValueError("Inference tokenizer does not preserve the serialized backend")
        if (len(tok), tok.bos_token_id, tok.eos_token_id, tok.pad_token_id) != (
            100278,
            None,
            100257,
            100277,
        ):
            raise ValueError("Incorrect production vocabulary/special IDs")
    return digest
