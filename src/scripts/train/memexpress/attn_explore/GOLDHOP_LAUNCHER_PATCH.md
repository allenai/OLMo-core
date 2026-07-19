# Launcher patch — `gold_hop_controlled` (Approach A: the multi-hop gold-routing ladder)

Applies to **`Qwen3-0.6B-docchunk-mask-mix-contradiction-SFT-local.py`** (you hold that file, so it is
handed over as a patch rather than edited). Everything below is additive: no existing arm's behaviour
changes, and with `--cross-doc-mode` unset from `gold_hop_controlled` not one line of this runs.

Library side is landed and tested (`src/olmo_core/nn/attention/gold_hop_mask.py`,
`src/test/nn/attention/gold_hop_mask_test.py`, 144 CPU tests green).

---

## 0. Prerequisite — the pair-preserving sidecar (already built)

```
/scratch/users/prasann/longctx_sft_qwen/contra_n50_v2_orig/gold_pairs.json
```

2000/2000 examples, 6000 pairs, verified to flatten **exactly** onto the shard's existing
`gold_fingerprints.json`. Rebuild with:

```bash
python src/scripts/data/build_gold_sidecar_from_shard.py \
  --shard-dir /scratch/users/prasann/longctx_sft_qwen/contra_n50_v2_orig \
  --emit pairs \
  --verify-against /scratch/users/prasann/longctx_sft_qwen/contra_n50_v2_orig/gold_fingerprints.json
```

⚠ The flat `gold_fingerprints.json` is **rejected** by the mask (it is an unordered set and cannot say
which doc contradicts which). That is deliberate — it is the defect that invalidated the first goldgrad
arms.

---

## 1. New CLI flags

**Anchor** (unique, `main()` — verified: line ~498, the closing `)` of the `--random-doc-per-example`
`add_argument`):

```python
        "example) instead of one graph shared by all examples. Ablates whether a STABLE mask "
        "matters or whether sparse-but-varied connectivity suffices.",
    )
```

**Insert after it:**

```python
    # ---- gold_hop_controlled (Approach A: the multi-hop gold-routing ladder) ----
    # The one GOLD-AWARE mask: it deletes the direct gold<->gold edge and forces a path of a controlled
    # length, which is the literal form of "can the model use two documents that never attend to each
    # other?". Base graph is the gold-AGNOSTIC random_doc p=--doc-keep-prob, PER EXAMPLE.
    ap.add_argument(
        "--gold-hops",
        type=str,
        default="2",
        choices=("1", "2", "3", "inf"),
        help="gold_hop_controlled arm: 1 = gold edge forced present (upper reference); 2 / 3 = gold "
        "edge DELETED and the shortest gold path forced to exactly that length (2 is the test); "
        "inf = gold edge deleted AND every path cut (the leak-matched control). Read the ladder as "
        "hop2 vs hop_inf -- NOT vs chunked, which is not leak-matched.",
    )
    ap.add_argument(
        "--gold-pairs",
        type=str,
        default=None,
        help="gold_pairs.json for the training shard ({fingerprint: [[a, b], ...]}). REQUIRED by "
        "--cross-doc-mode gold_hop_controlled. Build with `build_gold_sidecar_from_shard.py "
        "--emit pairs`; the flat gold_fingerprints.json is rejected.",
    )
    ap.add_argument(
        "--gold-decoys",
        type=int,
        default=12,
        help="⚠ THE LEAK FIX -- distance-matched NON-gold pairs per gold pair given the IDENTICAL "
        "edit, so the arm's structural signature stops naming the gold pair. MEASURED "
        "(debug/gold_hop/leak_probe.py, graph-only classifier, no text, out-of-sample): at 0, "
        "hop_inf reaches precision@3 16.2% vs 0.245% chance (66x) while hop2 reaches 10x -- NOT "
        "leak-matched, which breaks the hop2-hop_inf contrast. At 12: hop_inf 8.2x, hop2 7.3x "
        "(matched), gold convergence 150/150, edge drift still exactly 0. ⚠ hop3 is incompatible "
        "(gold convergence 147/150, drift +3.9) -- use 0 for hop3 and declare its 26x leak.",
    )
```

`--doc-keep-prob`, `--random-doc-seed` and `--random-doc-per-example` already exist and are **reused**
as the base-graph knobs. ⚠ Do **not** hard-code 0.25: if Stage 1's `random_doc p=0.25` arm pins to the
chunked floor, the whole ladder is unresolvable on that base and moves to `p=0.5` (whose larger residual
leak `hop_inf` cancels anyway).

---

## 2. Model config

**Anchor** (unique) — the `elif` chain in `build_and_fit()`:

```python
    elif opts.cross_doc_mode == "summary_attention":
```

**Insert a new branch immediately before it** (or after the block, anywhere in the chain):

```python
    elif opts.cross_doc_mode == "gold_hop_controlled":
        # The base graph is random_doc at --doc-keep-prob, PER EXAMPLE, built inside gold_hop_mask
        # with the same seeded hash the random_doc pattern uses (asserted bit-identical in
        # src/test/nn/attention/gold_hop_mask_test.py). gold_hops is recorded on the attention config
        # so config.json names the arm; the per-example gold-edited graph arrives at runtime via the
        # fingerprint hook, which refuses to install if the two disagree.
        qwen_kwargs["gold_hops"] = GOLD_HOPS_INF if opts.gold_hops == "inf" else int(opts.gold_hops)
        # Recorded in config.json so EVAL rebuilds the identical graph: decoys change the graph, so a
        # train/eval mismatch would score the model on a mask it never saw -- silently.
        qwen_kwargs["gold_decoys"] = opts.gold_decoys
```

**Import anchor** (unique):

```python
from olmo_core.nn.attention.chunked_mask import mask_mix_standard_prob
```

**Replace with:**

```python
from olmo_core.nn.attention.chunked_mask import mask_mix_standard_prob
from olmo_core.nn.attention.gold_hop_mask import (
    GOLD_HOPS_INF,
    install_gold_hop_mask,
    make_fingerprint_gold_hop_fn,
)
```

---

## 3. Validation — fail at launch, not after a wasted run

**Anchor** (unique, `build_and_fit()` — verified: lines 85-86):

```python
    meta = json.load(open(f"{opts.data_dir}/metadata.json"))
    n_examples = int(meta["num_instances"])
```

**Insert after it:**

```python
    # ⚠ --random-doc-per-example is MANDATORY for the gold-hop ladder, and it is a deviation from every
    # prior random_doc run (which used one graph shared across all layers AND all examples). A graph
    # that fixed is memorizable -- and against a memorized graph a MISSING edge announces which doc is
    # gold, amplifying the exact leak the camouflage base exists to suppress.
    if opts.cross_doc_mode == "gold_hop_controlled":
        if not opts.gold_pairs:
            raise SystemExit(
                "--cross-doc-mode gold_hop_controlled requires --gold-pairs "
                "(gold_pairs.json). Build it with build_gold_sidecar_from_shard.py --emit pairs."
            )
        if not opts.random_doc_per_example:
            raise SystemExit(
                "--cross-doc-mode gold_hop_controlled requires --random-doc-per-example. A base graph "
                "shared across examples is memorizable, and then the DELETED gold edge is a beacon "
                "pointing at the gold pair -- the mask would leak the answer it is testing."
            )
```

---

## 4. Install the hook after `model = ...` is built

The mask needs per-example gold identity, and it must **never** enter the token stream. So the graph is
built in a `forward_pre_hook` that fingerprints the live `input_ids` — the same mechanism
`gold_grad_mask` already uses.

**Anchor** (unique, `build_and_fit()` — verified: lines 354-355):

```python
    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
```

**Insert after it** (⚠ *after* `train_module_config.build`, not after `model_config.build`: the model is
built on the **meta** device and the train module is what materializes/wraps it, so
`train_module.model` is the live module the hook must attach to):

```python
    if opts.cross_doc_mode == "gold_hop_controlled":
        gold_pairs_table = json.load(open(opts.gold_pairs))
        gold_hop_fn = make_fingerprint_gold_hop_fn(
            gold_pairs_table,
            doc_start_id=DOC_START_ID,
            doc_end_id=DOC_END_ID,
            eos_id=EOS_TOKEN_ID,
            hops=GOLD_HOPS_INF if opts.gold_hops == "inf" else int(opts.gold_hops),
            doc_keep_prob=opts.doc_keep_prob,
            seed=opts.random_doc_seed,
            per_example=opts.random_doc_per_example,
            n_decoys=opts.gold_decoys,
        )
        gold_hop_holder = install_gold_hop_mask(train_module.model, gold_hop_fn)
        print(
            f"[gold-hop] arm={opts.gold_hops} keep_prob={opts.doc_keep_prob} "
            f"seed={opts.random_doc_seed} decoys={opts.gold_decoys} "
            f"sidecar={len(gold_pairs_table)} examples "
            f"attached_layers={gold_hop_holder.n_attached}",
            flush=True,
        )
```

`install_gold_hop_mask` already raises if no layer reads the graph, or if the hook's arm disagrees with
the arm recorded in the model config (so a launcher typo cannot train `hop_inf` under a run named
`hop2`).

**Log the REALIZED structure at the end of the run** — the config states intent, the holder states fact:

```python
    # after trainer.fit()
    if opts.cross_doc_mode == "gold_hop_controlled":
        print(gold_hop_holder.summary(), flush=True)
        if get_rank() == 0:
            # Stage 4 joins this on the fingerprint to stratify per-example f1 by REALIZED hop
            # distance -- hop2's headline is a ~96/4 mixture over it (the adjacent pairs), so the
            # routable subset is the mixture-free number.
            n = dump_realized_hops(gold_hop_holder.stats, f"{save_folder}/realized_hops.json")
            print(f"[gold-hop] wrote realized_hops.json ({n} examples)", flush=True)
```

(add `dump_realized_hops` to the `gold_hop_mask` import in §2.)

Prints e.g.
`[gold-hop] examples=2000 pairs=6000 realized_hops: 2=0.965(96.5%) inf=0.035(3.5%) unroutable=210(3.50%) edge_drift mean=+0.000 max_abs=0`

---

## 5. Inherited constraints — all already satisfied by this family

| constraint | status |
|---|---|
| **`--no-compile`** | required — the per-forward fingerprint hook is not `torch.compile`-capturable, exactly like the mask-mix curriculum. `build_chunked_mask_mod` deliberately declines this pattern, so every arm stays on the one dense eager path. |
| **`--num-workers 0`** | already the flag; 2 workers deadlock when several torchrun jobs share a node. |
| **`.contiguous()` on edited K/V** | **not applicable** — this is a *mask*, it never touches K/V. (That constraint belongs to `gold_grad_mask`'s detach, which this reuses the *lookup* of, not the detach.) |
| **norm-repaired base** | `q06b-dense-cpt-modelonly-trainedmark` only. |
| **`p_standard` anneal → 0** | the launcher's existing hard-fail guard covers it; it must stay live for every arm. |
| **data** | `contra_n50_v2_orig` only. ⚠ `contra_n50_v2_7k` is poisoned. |

---

## 6. Suggested arms

Base `random_doc p=0.25` **per-example**, seed 42, everything else per the standing Stage-1 recipe.
`chunked` / `standard` come from the existing Stage 1 arms (floor 0.408, ceiling 0.943).

Common: `--cross-doc-mode gold_hop_controlled --doc-keep-prob 0.25 --random-doc-per-example
--gold-decoys 12 --gold-pairs /scratch/users/prasann/longctx_sft_qwen/contra_n50_v2_orig/gold_pairs.json`

⚠ **Stay at `--doc-keep-prob 0.25`** — Stage 1 Gate 2 passed (`random_doc` p=0.25 scored f1 0.558 vs the
0.408 floor / 0.943 ceiling), so the base routes and the p=0.5 fallback is not needed.

| arm | flags |
|---|---|
| `hop1` | ` … --gold-hops 1 … ` |
| `hop2` | ` … --gold-hops 2 … ` **(the test; 2 seeds)** |
| `hop_inf` | ` … --gold-hops inf … ` **(the control; 2 seeds)** |
| `hop3` | ` … --gold-hops 3 --gold-decoys 0 … ` ⚠ decoys over-constrain hop3 (gold convergence 147/150). Its 26x graph-only leak must be declared, or drop the arm. |

### Eval (the sidecar is BUILT and verified)

```bash
# already built: 488/488 unique fingerprints, 1464 pairs
python src/scripts/data/build_gold_pairs_for_eval.py \
  --contra-data /scratch/users/prasann/corpus-reasoning/data/contradiction_eval_pubmed_both_n50_k3.jsonl \
  --out /scratch/users/prasann/longctx_sft_qwen/contra_n50_v2_orig/gold_pairs_eval_n50.json \
  --tokenizer Qwen/Qwen3-0.6B --cot-mode none

torchrun ... src/corpus_reasoning/eval/eval_lc_native_docchunk_contra.py \
  --variant dense --model-path <ckpt> --cot-mode none --eos-token-id 151643 --max-length 8192 \
  --contra-data .../contradiction_eval_pubmed_both_n50_k3.jsonl --max-test-samples 0 \
  --gold-pairs .../gold_pairs_eval_n50.json --gold-hops 2
```

⚠ **The eval sidecar is a DIFFERENT FILE from the training one and they are not interchangeable** — the
eval prefill is prompt-only (no answer/EOS/pad), so the two key spaces share **0 of 488** fingerprints
(measured). Passing the training `gold_pairs.json` makes every row fall back to plain causal and score
near the ceiling; the eval now `SystemExit`s instead (demonstrated). `doc_keep_prob` / `random_doc_seed`
/ `gold_decoys` are read from the checkpoint's `config.json`, not from eval flags.

**Read:** `hop2 > hop_inf` ⇒ channel (c) is real. `hop2 ≈ hop_inf` ⇒ everything above the chunked floor
is just the FREE bridge. The contrast is leak-cancelled by construction — both arms have identical
gold-edge-absent structure.

⚠ A null is only interpretable if `hop1` lands near `full` and `hop_inf` near `chunked`. If that bracket
collapses, the ladder had no resolution and nothing `hop2` does means anything.
