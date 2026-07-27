"""
Produce a Llama-3 tokenizer copy whose reserved slots 128002/128003 are NAMED
``<|box_start|>`` / ``<|box_end|>``, so the document-chunk data path works unchanged.

Why this exists: the document-chunk converter
(:mod:`src.scripts.data.convert_unified_to_document_landmark`) and the native evaluators wrap every
context document with the literal strings ``<|box_start|>`` / ``<|box_end|>`` and then verify that
``tok.convert_tokens_to_ids(...)`` returns the ids in ``RESERVED_IDS[family]``. Qwen3/Qwen3.5 happen
to ship tokens with exactly those names. Llama 3 does not -- it ships 250 anonymous
``<|reserved_special_token_N|>`` slots instead. Rather than fork the string constants (which would
have to be threaded through the converter, the trainer AND both evaluators, i.e. four more places a
train/eval mismatch could hide), we RENAME two of Llama's reserved slots in a local tokenizer copy.

Nothing about the vocabulary changes: ids, merges and the BPE model are untouched, only two
``added_tokens`` *content* strings. Id 128002 (``<|reserved_special_token_0|>``) becomes
``<|box_start|>`` and 128003 (``<|reserved_special_token_1|>``) becomes ``<|box_end|>``, matching
``RESERVED_IDS["llama"]``. Those rows are untrained in the base checkpoint, which is precisely why
``fix_marker_embeddings.py --family llama`` must still be run on the base -- see
``records/document-chunked-marker-embeddings.md``.

Usage::

    python src/scripts/data/make_llama_marker_tokenizer.py \\
        --base /scratch/users/prasann/hf_models/Llama-3.2-3B \\
        --out  /scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok
"""

import argparse
import json
import os
import shutil

from olmo_core.data.document_chunk_landmark import (  # canonical ids -- never retype
    DOC_END_STR,
    DOC_START_STR,
    RESERVED_IDS,
)

#: Tokenizer files that define the vocabulary/specials. Weights are deliberately NOT copied: the
#: output is a tokenizer-only directory, so it can never be mistaken for a model checkpoint.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")

#: Chat template for the Llama **base** checkpoints. ``segment_prompt_to_chunks`` renders every
#: prompt through ``tok.apply_chat_template``; Qwen3.5-*-Base ships one, Llama-3.2-3B(-Base) does
#: NOT, so the data path dies with "tokenizer.chat_template is not set" without this.
#:
#: It deliberately uses **plain text** rather than Llama's chat specials
#: (``<|start_header_id|>``/``<|eot_id|>``/...). Those rows are barely-or-never trained in a base
#: checkpoint -- exactly the out-of-distribution-embedding failure mode that
#: ``records/n100-chunked-marker-position-bug.md`` documents -- and putting three of them in every
#: single prompt would inject that noise everywhere, not just at document boundaries. ``bos_token``
#: is kept because it IS trained and Llama depends on it.
#:
#: The template must satisfy the harness's invariant that the generation prompt is a **prefix** of
#: the full conversation, which is why the ``### Response:`` header is emitted with the user turn
#: rather than gated on ``add_generation_prompt``. :func:`main` asserts this.
LLAMA_BASE_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "### Instruction:\n{{ message['content'] }}\n\n### Response:\n"
    "{% else %}{{ message['content'] }}{% endif %}"
    "{% endfor %}"
)


def _rename_in_tokenizer_json(path: str, renames: dict) -> int:
    """Rename ``added_tokens`` contents by id in a ``tokenizer.json``.

    :param path: Path to ``tokenizer.json``.
    :param renames: ``{id: new_content}``.

    :returns: Number of entries renamed.
    """
    with open(path) as f:
        data = json.load(f)
    n = 0
    for entry in data.get("added_tokens", []):
        if entry["id"] in renames:
            entry["content"] = renames[entry["id"]]
            n += 1
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    return n


def _rename_in_tokenizer_config(path: str, renames: dict) -> int:
    """Rename ``added_tokens_decoder`` contents by id in a ``tokenizer_config.json``.

    :param path: Path to ``tokenizer_config.json``.
    :param renames: ``{id: new_content}``.

    :returns: Number of entries renamed.
    """
    with open(path) as f:
        data = json.load(f)
    n = 0
    for tid, new in renames.items():
        key = str(tid)
        if key in data.get("added_tokens_decoder", {}):
            data["added_tokens_decoder"][key]["content"] = new
            n += 1
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="source HF Llama model/tokenizer dir")
    ap.add_argument("--out", required=True, help="destination tokenizer-only dir")
    ap.add_argument("--family", default="llama", help="RESERVED_IDS key (default llama)")
    args = ap.parse_args()

    ids = RESERVED_IDS[args.family]
    renames = {ids.doc_start: DOC_START_STR, ids.doc_end: DOC_END_STR}

    os.makedirs(args.out, exist_ok=True)
    for name in TOKENIZER_FILES:
        src = os.path.join(args.base, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, name))

    # Refuse to rename anything that is NOT an untrained reserved slot: silently renaming a real
    # token (e.g. <|eot_id|>) would corrupt every prompt built with this tokenizer.
    with open(os.path.join(args.out, "tokenizer_config.json")) as f:
        before = json.load(f)["added_tokens_decoder"]
    for tid in renames:
        content = before[str(tid)]["content"]
        if "reserved_special_token" not in content:
            raise SystemExit(
                f"id {tid} is {content!r}, not an untrained <|reserved_special_token_N|> slot -- "
                f"refusing to rename it. Fix RESERVED_IDS[{args.family!r}]."
            )
        print(f"  {tid}: {content} -> {renames[tid]}")

    n1 = _rename_in_tokenizer_json(os.path.join(args.out, "tokenizer.json"), renames)
    n2 = _rename_in_tokenizer_config(os.path.join(args.out, "tokenizer_config.json"), renames)
    if n1 != len(renames) or n2 != len(renames):
        raise SystemExit(f"renamed {n1} in tokenizer.json and {n2} in tokenizer_config.json")

    cfg_path = os.path.join(args.out, "tokenizer_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["chat_template"] = LLAMA_BASE_CHAT_TEMPLATE
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Verify by actually loading it: this is the same call the converter/evaluators make.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.out)
    got_start = tok.convert_tokens_to_ids(DOC_START_STR)
    got_end = tok.convert_tokens_to_ids(DOC_END_STR)
    if (got_start, got_end) != (ids.doc_start, ids.doc_end):
        raise SystemExit(
            f"verification failed: {DOC_START_STR}->{got_start} (want {ids.doc_start}), "
            f"{DOC_END_STR}->{got_end} (want {ids.doc_end})"
        )
    # ...and that the markers survive a real round-trip through the string-level wrapping the
    # converter does (a marker that re-tokenizes into pieces would silently break chunking).
    round_trip = tok(f"{DOC_START_STR}hello{DOC_END_STR}", add_special_tokens=False).input_ids
    if round_trip[0] != ids.doc_start or round_trip[-1] != ids.doc_end:
        raise SystemExit(f"marker round-trip failed: {round_trip}")

    # The data path (segment_prompt_to_chunks) renders the prompt and the prompt+answer separately
    # and REQUIRES the former to be a prefix of the latter -- it derives the answer label mask from
    # that boundary. A template that emits a turn terminator only in one of the two breaks it.
    msgs = [{"role": "user", "content": "QUESTION"}]
    prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": "ANSWER"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full.startswith(prompt_str) or not full.endswith("ANSWER"):
        raise SystemExit(
            "chat template broken: the generation prompt must be a prefix of the full "
            f"conversation.\n  prompt={prompt_str!r}\n  full={full!r}"
        )
    # No untrained chat specials may leak into the rendering (see LLAMA_BASE_CHAT_TEMPLATE).
    bad = [t for t in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>") if t in full]
    if bad:
        raise SystemExit(f"chat template emits untrained special tokens: {bad}")
    print(f"chat template OK; prompt renders as {prompt_str!r}")
    print(
        f"OK: {args.out} maps {DOC_START_STR}->{got_start} {DOC_END_STR}->{got_end}; "
        f"round-trip {round_trip}; len(tok)={len(tok)}"
    )


if __name__ == "__main__":
    main()
