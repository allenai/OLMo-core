"""Measure the per-instance padding tail of the dense 256k SFT packing.

Confirms (or refutes) that PackingInstanceSource emits windows whose padding tail alone exceeds the
~61.4k EOS-derived 'documents' at which fla's causal_conv varlen grid (grid.y = NT ~ 4096 + n_docs)
crosses the CUDA 65535 limit.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "src")

from olmo_core.data import TokenizerConfig  # noqa: E402
from olmo_core.data.composable import (  # noqa: E402
    LongDocStrategy,
    MixingDocumentSourceConfig,
    MixingDocumentSourceSpecConfig,
    NumpyDocumentSourceConfig,
    PackingInstanceSourceConfig,
)
from olmo_core.data.utils import get_document_lengths  # noqa: E402

from dataclasses import replace  # noqa: E402

DATA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/shards_full"
DOLCI = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen35"
SEQ = 262144
WORK_DIR = "/weka/oe-training-default/ai2-llm/checkpoints/amandab/dataset-cache"

tok = TokenizerConfig.qwen3_5()
doc_tok = replace(tok, bos_token_id=None)
print(f"eos={tok.eos_token_id} pad={tok.pad_token_id} -> pad==eos: {tok.pad_token_id == tok.eos_token_id}")

_W = {"contra": 2.0, "rerank": 1.5, "outlier": 1.5, "nq": 1.0, "oolong": 1.0}
S = sum(_W.values())


def src(root):
    r = root.rstrip("/")
    return NumpyDocumentSourceConfig(
        source_paths=[f"{r}/token_ids_part_*.npy"],
        tokenizer=doc_tok,
        label_mask_paths=[f"{r}/labels_mask_*.npy"],
        expand_glob=True,
    )


five = [
    MixingDocumentSourceSpecConfig(source=src(f"{DATA_ROOT}/contradiction_train"), ratio=_W["contra"] / S, max_repetition_factor=8.0, label="contradiction"),
    MixingDocumentSourceSpecConfig(source=src(f"{DATA_ROOT}/nq_train"), ratio=_W["nq"] / S, max_repetition_factor=8.0, label="nq_retrieval"),
    MixingDocumentSourceSpecConfig(source=src(f"{DATA_ROOT}/oolong_train"), ratio=_W["oolong"] / S, max_repetition_factor=8.0, label="oolong"),
    MixingDocumentSourceSpecConfig(source=src(f"{DATA_ROOT}/rerank_train"), ratio=_W["rerank"] / S, max_repetition_factor=8.0, label="rerank"),
    MixingDocumentSourceSpecConfig(source=src(f"{DATA_ROOT}/outlier_train"), ratio=_W["outlier"] / S, max_repetition_factor=8.0, label="outlier"),
]
specs = [
    MixingDocumentSourceSpecConfig(source=MixingDocumentSourceConfig(source_specs=five), ratio=0.75, label="five_task_mix"),
    MixingDocumentSourceSpecConfig(source=src(DOLCI), ratio=0.25, max_repetition_factor=8.0, label="dolci_instruct_sft"),
]

cfg = PackingInstanceSourceConfig(
    sources=[MixingDocumentSourceConfig(source_specs=specs)],
    sequence_length=SEQ,
    tokenizer=doc_tok,
    long_doc_strategy=LongDocStrategy.exclude,
)
source = cfg.build(WORK_DIR)
n = len(source)
print(f"instances: {n}")

LIMIT = 65535
BT = 64
n_over = 0
worst = 0
worst_i = -1
ndocs_list = []
step = max(1, n // 1200)  # sample ~1200 instances spread across the source
idxs = list(range(0, n, step))
for c, i in enumerate(idxs):
    ids = np.asarray(source[i]["input_ids"])
    doc_lens = get_document_lengths(ids, tok.eos_token_id, bos_token_id=None)
    ndocs = int(doc_lens.numel())
    # NT as fla computes it: sum over docs of ceil(len/BT)
    nt = int(np.ceil(doc_lens.numpy().astype(np.float64) / BT).sum())
    ndocs_list.append((ndocs, nt))
    if nt > worst:
        worst, worst_i = nt, i
    if nt > LIMIT:
        n_over += 1
    if c % 200 == 0:
        print(f"  [{c}/{len(idxs)}] i={i} ndocs={ndocs} NT={nt}", flush=True)

nts = np.array([x[1] for x in ndocs_list])
nds = np.array([x[0] for x in ndocs_list])
print("\n=== RESULT ===")
print(f"sampled {len(idxs)} of {n} instances")
print(f"docs/instance : min={nds.min()} median={int(np.median(nds))} max={nds.max()}")
print(f"NT (grid.y)   : min={nts.min()} median={int(np.median(nts))} max={nts.max()}")
print(f"CUDA grid.y limit = {LIMIT}")
print(f"instances over limit: {n_over}/{len(idxs)} ({100.0*n_over/len(idxs):.2f}%)  worst NT={worst} at i={worst_i}")
print("VERDICT:", "CONFIRMED - varlen grid exceeds CUDA limit" if n_over else "NOT reproduced in this sample")
