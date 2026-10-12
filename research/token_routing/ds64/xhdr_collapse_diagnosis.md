# Why the outlier HEADER-REAL arms (`xhdr00/17/33`) collapse

**Date** 2026-09-14 · **Author** diagnostic pass on records/ds64-handoff.md §8
**Verdict** NOT a code bug. The `--st-header-stop-id 5491` path does exactly what it is documented to
do (verified token-by-token on CPU). The collapse is a **construction-level confound that is unique
to outlier**: header-real + gold-blind together hand the model the complete answer vocabulary for
documents it cannot read, and the loss-minimising policy becomes *"emit three ids sampled from the
visible id list"*. Every xhdr number sits **exactly on the uniform-random-guess floor**.
**Recommendation: kill the remaining xhdr runs. The arm family is uninformative as specified.**

---

## 1. The numbers (eval_size = 500 per rung, all rungs; SE ≈ 0.019 at f1≈0.24, ≈0.010 at f1≈0.05)

The outlier grader is set-F1 over `k=3` gold ids out of `n` documents. A model that emits three
**in-range but content-blind** ids scores `E[f1] = k/n` exactly. Measured `n` on the very rung files
this eval used (`outlier_lengthmix/eval_rungs/outlier/rung_*.jsonl`, 500 rows each):

| rung | n_docs (mean, min–max) | k | **uniform-guess floor = k/n** |
|---|---|---|---|
| 2k  | 13.4 (11–14)    | 3 | **0.224** |
| 8k  | 56.3 (54–57)    | 3 | **0.053** |
| 16k | 110.4 (108–111) | 3 | **0.027** |
| 32k | 219.3 (217–220) | 3 | **0.014** |

Observed (weka `.../ds64-outlier-*/eval/outlier_multirung.json`, eval_size=500/rung):

| run | 2k | 8k | 16k | 32k |
|---|---|---|---|---|
| **guess floor** | **0.224** | **0.053** | **0.027** | **0.014** |
| xhdr00-u16M | 0.267 | 0.046 | 0.035 | 0.011 |
| xhdr00-u32M | 0.241 | 0.047 | 0.032 | 0.011 |
| xhdr00-u64M | 0.223 | 0.052 | — | — |
| xhdr17-u16M | 0.233 | 0.052 | 0.033 | 0.014 |
| xhdr17-u32M | 0.239 | 0.045 | — | — |
| xhdr33-u16M | 0.237 | 0.042 | 0.031 | — |
| xhdr33-u32M | 0.232 | — | — | — |
| kvgb-u16M   (keep 1/3, NO header) | 0.684 | 0.139 | 0.052 | 0.018 |
| kvgb50-u16M (keep 1/2, NO header) | 0.871 | 0.188 | 0.066 | 0.015 |
| dense-u16M  | 0.911 | 0.483 | 0.153 | 0.054 |

Three independent facts, each on its own damning:

1. **Every xhdr cell is within noise of `k/n`.** Not "degraded" — *exactly chance*, at every rung.
2. **`keep_prob` does nothing.** 0.0 / 1/6 / 1/3 all land on the same number. Yet the same knob moves
   the header-free arms a lot (kvgb 0.684 → kvgb50 0.871 at 2k).
3. **Budget does nothing.** xhdr00 at 16M / 32M / 64M: 0.267 / 0.241 / 0.223 at 2k (drifting *down*,
   i.e. flat). dense goes 0.911 → 0.965 → 0.972 over the same budgets.

The training loss says the same thing. 28-step CE curves (`train/CE loss`, 4 GPUs, 16M):

```
xhdr00  1.069 0.679 0.563 ... 0.483 0.496 0.503      final ≈ 0.50
xhdr33  1.158 0.734 0.611 ... 0.463 0.467 0.480      final ≈ 0.48
kvgb50  1.134 0.731 0.576 ... 0.351 0.346 0.374      final ≈ 0.35
```

xhdr00 (**no** real bodies) and xhdr33 (**1/3** real bodies) are within 0.01–0.02 of each other from
step 8 onwards. Giving the model a third of the documents' text back buys nothing, because it has
already found a policy whose loss does not depend on the text. That is the signature of a shortcut,
not of a hard task. (Cf. the standing `ce-vs-chance-floor` note: a plateaued CE is often *at* a floor.)

## 2. The mechanism, shown mechanically

`research/token_routing/ds64/repro_xhdr_compaction.py` (new, CPU-only) rebuilds exactly what the trainer feeds the
model — same `segment_prompt_to_chunks` + `emit_document_chunk_dense` as the ds64 tokenizer, then
`build_chunk_ids_from_tokens` → `mark_doc_headers_free` → `resolve_keep_docs` → `compact_pooled_rows`
— and decodes the compacted row. One real outlier row, 22 docs, gold `[1,2,19]`, answer
`Outliers: [2], [3], [20]`:

**Without header-real (`kvgb`), keep_prob 0.0** — 3460 → 156 tokens:
```
...Outliers: [id1], [id2], ...<SLOT><SLOT><SLOT><SLOT>...<SLOT>Can you find passages that are about...
```
**With header-real (`xhdr00`), keep_prob 0.0** — 3460 → 279 tokens:
```
...Outliers: [id1], [id2], ...

Document [1]:<SLOT>

Document [2]:<SLOT>

Document [3]:<SLOT> ... Document [22]:<SLOT>Can you find passages that are about...
```
**With header-real, keep_prob 1/3** — pooled and kept docs interleave, *all 22 ids still visible*:
```
...Document [1]:<SLOT><BS>

Document [2]: Ludmilla ""Lucy"" Diakovska ( ; born 2 April 1976) is ...<|box_end|>

Document [3]:<SLOT> ...
```

That is the whole story:

* **Without the header, a pooled document's id is gone.** On the `(1-p)` of rows where the gold is
  pooled (gold-blind!), the target id is *not present anywhere in the context*, so no copy policy can
  produce it; the only way to lower loss is to read the `p` fraction of real bodies. Supervision stays
  content-grounded, and more `p` ⇒ better (kvgb 0.684 → kvgb50 0.871).
* **With the header, every id 1..N is real text, including for documents whose entire body is a single
  detached soft token.** Now on the `(1-p)` gold-pooled rows the target *is* in context and is
  perfectly copyable, but nothing in the context justifies picking it. Gradient descent's best
  response is the marginal: emit three ids from the visible list with roughly the empirical position
  prior. That policy is (a) available at every `keep_prob`, (b) already optimal, so extra real bodies
  and extra tokens change nothing — exactly what the flat CE and flat f1 show.

In short: **header-real converts an "impossible, must-ignore" target into a "copyable but
unjustifiable" one, and the shortcut wins.**

The repro also confirms the header path itself is clean — ruling out every mechanical hypothesis:

| check | result |
|---|---|
| header span freed | exactly `\n\nDocument [N]:` — 123 tokens over 22 docs = 5.6/doc (cap 32 never hit) |
| stop id 5491 present in every header | yes (handoff §8's decode is correct; id 25 would have hit the cap) |
| header tokens carrying a LABEL | **0** — labels are answer-only (`labels_mask_part_*.npy`), unchanged by freeing |
| label count through compaction | 17 → 17 (both with and without header-real) |
| `position_ids` | original, monotonic, no duplicates |
| query / answer region | untouched: `chunk_ids >= 0` guard in `mark_doc_headers_free` excludes FREE text (chunked_mask.py:331) |
| `]:` inside bodies or the question | irrelevant — the scan stops at the **first** occurrence after `<|doc_start|>` |
| gold indexing | `gold_doc_indices=[1,2,19]` → answer `[2],[3],[20]`; grader does `int(g)+1`. Consistent, no off-by-one |
| compaction ratio | sane: xhdr00 154x, xhdr33 26.9x, kvgb50 23.1x (whole-batch, includes pad removal) |

## 3. Why the SAME mechanism worked on contradiction and oolong

| task | header | answer is… | keep policy | outcome |
|---|---|---|---|---|
| contradiction `hdr33` | `Claim N:` (stop 25) | claim **ids** | `--st-keep-frac` + `--st-keep-mode gold_plus_random` → **gold is ALWAYS real** (`select_keep_docs`, gold_grad_mask.py:194) | 0.855 ✔ |
| oolong `ohdr17/33` | `Date: … \|\| User: … \|\| Instance:` (stop 25, count 3) | a **phrase** (`'less common than'`) — never a line id | gold-blind | 0.606 ✔ |
| outlier `xhdr*` | `Document [N]:` (stop 5491) | document **ids** | **gold-blind** | chance ✘ |

Outlier is the only cell with *gold-blind* **×** *the answer IS the document id* **×** *header-real*.
Contradiction dodges it by forcing the gold real (there is never an unjustifiable target); oolong
dodges it because exposing `Date:`/`User:` exposes the **match keys**, not the **answer space**.
Nothing about stop-id 5491, the leading `\n\n`, or the fused `']:'` token is at fault — the handoff's
stop-id analysis was right, and switching it back to 25 would only make the arm cheaper-to-wrong.

## 4. Secondary finding (real, but not the cause)

`mark_doc_headers_free` (chunked_mask.py:325-337) uses `dist >= 1` and `input_ids != doc_end_id`, so
a **pooled** document's `<|box_start|>` and `<|box_end|>` stay chunk-owned and get dropped by
`compact_pooled_rows`. The training stream therefore contains marker-less headers
(`\n\nDocument [N]:<SLOT>`) next to marker-wrapped kept docs (`<BS>\n\nDocument [M]: text<|box_end|>`),
while at eval (dense, full attention) *every* header is marker-wrapped. This train/eval layout
inconsistency is introduced by header-real and applies to contradiction/oolong too (where it is
evidently harmless). If header-real is ever revisited, `dist >= 0` plus keeping `doc_end` would make
the pooled-doc layout match inference. **It is not why outlier collapses** — it cannot explain
chance-level output or the total insensitivity to `keep_prob`.

## 5. What to do

**Kill all remaining xhdr work.** Nothing at 32M/64M can change the reading: xhdr00 already has
16M/32M/64M all at the floor, and xhdr17/xhdr33 already have 16M+32M at the floor.

Still running (cancel with `beaker job cancel <job-id>`; experiment stop/cancel are no-ops):

| run | kind | experiment | **job id** |
|---|---|---|---|
| `ds64-outlier-xhdr33-b128f3-u64M` | train | `01M2GX2YZY4SJCYQP37Q6ND0R5` | `01M2GX2Z48ZFQX3PQD58APBWKH` |
| `ds64-outlier-xhdr17-b128f3-u64M` | train | `01M2GX5N3X0DVE8Z4H9FWGW2GM` | `01M2GX5N7GV2WA9QP05EJ142QJ` |
| `ev-…-xhdr33-…-u16M` | eval | `01M2H1SVAZ29KKQAQXS3G917N6` | `01M2H1SVER3ZSWG8CACM7S34XW` |
| `ev-…-xhdr33-…-u32M` | eval | `01M2H40Z06VC5W5GKXNF4XSNRX` | `01M2H40ZK2D57AMSKN7ZC31TCR` |
| `ev-…-xhdr17-…-u16M` | eval | `01M2H15NC62B7V8CH44YMYJ1RK` | `01M2H15NFR9QJEWC0C21SGTC1D` |
| `ev-…-xhdr17-…-u32M` | eval | `01M2H3P9DFVA84N0ATWDWKDTWB` | `01M2H3P9S7JV4SDPNXZG2SYTXR` |
| `ev-…-xhdr00-…-u16M` | eval | `01M2GZP33V86EGPFJRB8JBV5SJ` | `01M2GZP37D8SB6HD7EJ3KBZ51F` |
| `ev-…-xhdr00-…-u32M` | eval | `01M2H16EWK3M4VAZ8SFHCN33J3` | `01M2H16EZZV05WVNAX2HJXTPM5` |
| `ev-…-xhdr00-…-u64M` | eval | `01M2H3C41833BEG3CM07FZ3QZ1` | `01M2H3C45JAWET31C9S8EC9647` |

Also **stop the orchestrator from relaunching them**: drop `xhdr33/xhdr17/xhdr00` from
`research/token_routing/ds64/soft_arms.json` (and leave `ARM_EXTRA` in place with the warning below) before restarting
it, or it will re-queue the 64M rungs.

**There is no code fix**, because there is no code bug. If the *question* ("is a real document id
enough to name the odd one out?") is still worth answering, the confound has to be removed, not the
implementation:

* **(a) Gold-forced header-real** — `--st-keep-mode gold_plus_random` instead of `--st-gold-blind`,
  i.e. contradiction's exact recipe. Removes the unjustifiable-target rows. **But** outlier's
  `kv17`/`kv33` already showed gold-forcing creates its own shortcut ("gold is always among the few
  real docs", 0.00–0.02 at every rung), so this probably just trades one shortcut for a worse one.
* **(b) Blind the id, keep the delimiter** — free only the structural part of the header
  (`\n\nDocument [` / `]:`) and leave the **number** pooled, so the layout is real but the answer
  vocabulary is not. `mark_doc_headers_free` cannot express this today; it would need a second
  stop/skip span.
* **(c) Renumber per forward** — shuffle the id↔document assignment each epoch so a copied id carries
  no positional prior. Data-side change.
* **Cheapest honest answer: don't.** kvgb50 (0.871 / 0.188 / 0.066 at 2k/8k/16k, eval_size=500) is
  still the best gold-blind soft arm on outlier and §1 already calls outlier done at parity.

### Guard for the future

Header-real is unsafe on **any** gold-blind task whose answer is drawn from the header itself
(outlier, and `nq`/`rerank` if they are ever given `--st-header-stop-id` with `--st-gold-blind` —
their answers are document ids too). Add the check to the arm definition, not to the code: before
pairing `--st-header-stop-id` with `--st-gold-blind`, confirm the freed header span does **not**
contain the answer string.

## 6. Repro / artefacts

* `research/token_routing/ds64/repro_xhdr_compaction.py` — the CPU repro above.
  `python research/token_routing/ds64/repro_xhdr_compaction.py --tokenizer /net/horton/data/prasann/hf_models/Qwen3.5-4B-Base --input /scratch/users/prasann/cpt_data/eval500_v2/outlier/outlier_wiki100w_n22_k3_eval_600.jsonl`
* `research/token_routing/ds64/xhdr_diag_weka.sh` — the weka-side CPU job (rung doc counts + every outlier result JSON);
  run as a gantry CPU job, `ai2/flex2` / `oe-other` / urgent / unallocated.
* `research/token_routing/ds64/xhdr_logs/` — the fetched train + eval Beaker logs.
* Note: `eval_lc_native_docchunk_ladder.py` writes **only** the summary JSON — it never saves
  generations (no `KEEP_RESPONSES` path, unlike the other drivers). That is why this diagnosis is
  built from the `k/n` floor rather than from raw generations; worth fixing in that driver.
