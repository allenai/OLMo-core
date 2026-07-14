"""
Stage 2c -- Document-chunked attention results + a token/mask visualization.

Emits ``outputs/docchunk.json`` with:
  * ``results`` -- the OOLONG eval scores for the dense (DocumentChunkedAttention) and landmark
    (DocumentLandmarkAttention) document-chunked SFT runs (read from outputs/eval_results/*.json if
    present, else the recorded numbers).
  * ``dense`` / ``landmark`` -- a small synthetic example rendered through the REAL olmo_core data
    primitives (``olmo_core.data.document_chunk_landmark`` + ``chunked_mask``): the emitted token
    sequence (with per-token role + chunk id) and the materialized attention mask, so the site can
    draw exactly what the model sees. Each OOLONG item line is one document; the landmark variant
    first-fit packs documents into landmark windows.
"""

import json
import os

try:
    from . import config
except ImportError:
    import config

# Special / reserved ids (match olmo_core.data.document_chunk_landmark + the converter).
BOX_S, BOX_E, LM, PAD, EOS = 151648, 151649, 151860, 151863, 151643

# A small, legible synthetic example: instruction (FREE) + 3 short documents (context) + query (FREE).
# Each content token gets a unique id so we can label it unambiguously; markers/landmark/pad use the
# real reserved ids so build_chunk_ids_from_tokens classifies them exactly as at train/eval time.
_LABELS = {BOX_S: "<box_start>", BOX_E: "<box_end>", LM: "<landmark>", PAD: "<pad>"}
MEM_FREQ_VIZ = 6  # block size 7 -- small so packing + landmarks are visible


def _content(label, start_id):
    """Return (ids, labels) for a run of content tokens with unique ids."""
    ids = list(range(start_id, start_id + len(label)))
    for i, lab in zip(ids, label):
        _LABELS[i] = lab
    return ids


def _segments():
    from olmo_core.data.document_chunk_landmark import ChunkSegment

    nid = 1
    segs = []

    def free(labels):
        nonlocal nid
        ids = _content(labels, nid)
        nid += len(labels)
        return ChunkSegment(ids, [False] * len(ids), False)

    def doc(labels):
        nonlocal nid
        inner = _content(labels, nid)
        nid += len(labels)
        ids = [BOX_S] + inner + [BOX_E]
        return ChunkSegment(ids, [False] * len(ids), True)

    segs.append(free(["You", "answer:"]))  # instruction (FREE)
    segs.append(doc(["docA-1", "docA-2"]))  # document 0
    segs.append(doc(["docB-1"]))  # document 1
    segs.append(doc(["docC-1", "docC-2"]))  # document 2
    segs.append(free(["Question", "->"]))  # query/answer (FREE)
    return segs


def _role(tid, cid):
    if tid == LM:
        return "landmark"
    if tid == PAD:
        return "pad"
    if cid == -2:
        return "pad"
    if tid in (BOX_S, BOX_E):
        return "marker"
    if cid is not None and cid >= 0:
        return "ctx"
    return "free"


def _render_variant(out_ids, mem_freq):
    import torch

    from olmo_core.nn.attention.chunked_mask import (
        AttentionPattern,
        build_chunk_ids_from_tokens,
        build_chunked_allowed_mask,
    )

    pad_id = PAD if mem_freq else None
    chunk_ids = build_chunk_ids_from_tokens(
        torch.tensor([out_ids]), BOX_S, BOX_E, EOS, pad_id=pad_id
    )[0].tolist()
    allowed = build_chunked_allowed_mask(AttentionPattern(name="chunked"), torch.tensor([chunk_ids]))[
        0
    ]
    tokens = []
    for pos, (tid, cid) in enumerate(zip(out_ids, chunk_ids)):
        tokens.append(
            {
                "pos": pos,
                "label": _LABELS.get(tid, str(tid)),
                "role": _role(tid, cid),
                "chunk": cid if cid is not None and cid >= 0 else None,
            }
        )
    block = (mem_freq + 1) if mem_freq else None
    is_mem = (
        [bool((p % block) == (block - 1)) for p in range(len(out_ids))] if block else None
    )
    return {
        "mem_freq": mem_freq,
        "block_size": block,
        "tokens": tokens,
        "mask": allowed.int().tolist(),
        "is_mem": is_mem,
    }


def _results():
    rows = [
        {
            "variant": "full",
            "label": "Full attention — baseline (no chunked mask)",
            "desc": "Standard causal attention on the same data; documents are NOT isolated.",
            "out": "q4b-docfull-oolong-ctx2048_docnative.json",
            "score": 0.511,
            "em": 0.425,
            "groups": {"user": 0.64, "timeline": 0.45, "counting": 0.43},
        },
        {
            "variant": "dense",
            "label": "Dense — DocumentChunkedAttention",
            "desc": "Ordinary full attention, restricted by the chunked mask (no landmarks).",
            "out": "q4b-docdense-oolong-ctx2048_docnative.json",
            "score": 0.527,
            "em": 0.463,
            "groups": {"user": 0.61, "timeline": 0.49, "counting": 0.48},
        },
        {
            "variant": "landmark",
            "label": "Landmark — DocumentLandmarkAttention",
            "desc": "Grouped-softmax landmark memory + the same chunked mask; docs packed into windows.",
            "out": "q4b-doclandmark-oolong-ctx2048_docnative.json",
            "score": 0.536,
            "em": 0.463,
            "groups": {"user": 0.68, "timeline": 0.49, "counting": 0.43},
        },
        {
            "variant": "landmark",
            "label": "Landmark — top-25% inference",
            "desc": "Same landmark model, hard top-k landmark retrieval at decode (attend the top 25% "
            "of document blocks). Matches exact accuracy at a fraction of the attention.",
            "out": "q4b-doclandmark-oolong-ctx2048_topk0.25.json",
            "score": 0.537,
            "em": 0.463,
            "groups": {"user": 0.68, "timeline": 0.49, "counting": 0.43},
        },
    ]
    eval_dir = os.path.join(config.CR_REPO_ROOT, "outputs", "eval_results")
    for r in rows:
        p = os.path.join(eval_dir, r["out"])
        if os.path.exists(p):
            try:
                with open(p) as f:
                    o = json.load(f).get("oolong", {})
                if o.get("score") is not None:
                    r["score"] = round(o["score"], 3)
                    r["em"] = round(o.get("exact_match", r["em"]), 3)
                    r["groups"] = {k: round(v, 2) for k, v in (o.get("per_task_group") or {}).items()}
                    # eval-set size. Legacy jsons stored this as "n"; `n` now means CORPUS
                    # size only, so read both and emit the unambiguous name.
                    _o = json.load(open(p))
                    r["eval_size"] = _o.get("eval_size", _o.get("n"))
            except Exception:
                pass
    return {"ctx": "ctx2048", "eval_size": 80, "rows": rows}


def main():
    config.ensure_out_dir()
    from olmo_core.data.document_chunk_landmark import (
        emit_document_chunk_dense,
        emit_document_chunk_landmark,
    )

    segs = _segments()
    dense_ids, _ = emit_document_chunk_dense(segs)
    landmark_ids, _ = emit_document_chunk_landmark(
        segs, mem_freq=MEM_FREQ_VIZ, mem_id=LM, pad_id=PAD
    )
    out = {
        "results": _results(),
        "dense": _render_variant(dense_ids, None),
        "landmark": _render_variant(landmark_ids, MEM_FREQ_VIZ),
    }
    path = os.path.join(config.OUT_DIR, "docchunk.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(
        f"[docchunk] wrote {path} "
        f"(dense {len(dense_ids)} tok, landmark {len(landmark_ids)} tok)"
    )


if __name__ == "__main__":
    main()
