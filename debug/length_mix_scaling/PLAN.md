# Does short-context data contribute to 32k performance? — PLAN (awaiting go/no-go)

**Question.** Holding the long-context training data **fixed**, does adding progressively more
short-context data improve **32k** performance? And if so, does [modest long + lots of cheap short]
beat [lots of long] at **lower wall-clock**?

**Design decisions (locked with user):** Qwen3.5-**4B**, **from base**, **two long-pool levels**.

---

## 0. Baseline (established; do not re-measure)

`ctc-s5-contra-full-4b` (4B, full attn, uniform 2k–32k joint mix), graded with **vLLM**:

| rung | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| set_f1 | 0.849 | 0.766 | 0.690 | 0.620 | **0.335** |

eval_size 500 → binomial SE ≈ **±0.021** at f1≈0.335; seed noise is additional.

⚠ **vLLM only.** The same checkpoint reads 0.571/0.219/0.038 at 2k/8k/32k on the **native**
backend — a degraded harness, not a model property. Never mix backends in one comparison.

## 1. Cost structure (why short data is the interesting lever)

Measured 4B throughput from this repo's logs:

| seq_len | 5632 | 8192 | 40960 |
|---|---|---|---|
| TPS/device | 6.2–7.9k | 8.2–8.5k | 8.3k |

Tokens/sec is **flat** 8k→40k: at 4B the parameter term rivals attention, and with packing +
document masking attention cost tracks *document* length, not sequence length. So wall-clock is
essentially **proportional to tokens consumed**, whatever the mix — which makes the cost side of
this experiment simple and predictable rather than something to calibrate.

Consequence: a 4k example costs ~1/8 of a 32k example. If short data carries even a fraction of a
long example's value for 32k performance, it is a bargain.

## 2. Design — hold long fixed, add short

Let `L` = a **fixed** pool of long (16k–32k) examples, **identical across every arm** (same
examples, same order seed). Let `S` = short (≤8k) examples added on top.

Each arm trains for `max_steps ∝ (tokens_L + tokens_S)`, so **every arm sees the same number of
long examples** — that is what "long held constant" requires. Short exposure grows across arms, and
so does wall-clock. **Wall-clock is a measured output, not a control.**

> This is the key departure from a mix-fraction sweep: there, adding short data *displaces* long
> data and you cannot attribute a change to either. Here, only one thing varies.

### Run matrix (10 runs, all 1-node, parallel, urgent)

**Row A — full long pool (`L`, ≈40 M long tokens):**

| arm | S/L (tokens) | total tokens | est. wall-clock |
|---|---|---|---|
| A0 | 0 (long only) | 40 M | ~10 min |
| A1 | 0.5 | 60 M | ~15 min |
| A2 | 1 | 80 M | ~20 min |
| A3 | 2 | 120 M | ~30 min |
| A4 | 4 | 200 M | ~50 min |

**Row B — half long pool (`L/2`), to get the interaction:**

| arm | S/L | total | est. |
|---|---|---|---|
| B0 | 0 | 20 M | ~5 min |
| B2 | 1 | 40 M | ~10 min |
| B4 | 4 | 100 M | ~25 min |

Row B answers the question that actually decides a production mix: **is short data a substitute for
long data, or only a complement?** If short data helps *more* when long data is scarce (B rises
faster than A), it substitutes; if it helps equally, it is adding something long data never had.

**Row C — iso-wall-clock reference (the "is this actually cheaper?" control):**

| arm | what | matched to |
|---|---|---|
| C3 | uniform 2k–32k mix, trained to the same wall-clock as A3 | A3 (~30 min) |
| C4 | uniform 2k–32k mix, trained to the same wall-clock as A4 | A4 (~50 min) |

Row C is what converts this into your original question: at *equal* wall-clock, does
[fixed long + piled-on short] beat the production uniform mix at 32k?

## 3. Measurement

Every checkpoint evaluated with **vLLM at 2k / 8k / 32k**. Primary metric **f1@32k**; the 3-rung
curve shows *where* short data pays (if it only lifts 2k/8k and not 32k, that is a clean negative).

**Fit.** With long fixed, model
`f1@32k(N_S) = f∞ − (f∞ − f0)·exp(−N_S / τ)` (or a power-law saturation).
- **slope at N_S = 0** → the marginal value of the first short tokens;
- **τ** → where short data stops paying;
- comparing Row A vs Row B τ/asymptote → substitute vs complement.

Combined with the measured cost (~tokens), the slope directly yields **f1 gained per GPU-minute**
for short vs long data — the number that decides the mix.

## 4. Schedule (≤5 h)

| phase | what | wall-clock |
|---|---|---|
| 0 | build fixed long pool + large short pool | ~45 min |
| 1 | 10 runs, parallel, longest ~50 min | ~50 min + ~20 min queue |
| 2 | export 10 ckpts → HF | ~20 min (parallel) |
| 3 | vLLM eval, 10 ckpts × 3 rungs | ~60 min (parallel over GPUs) |
| 4 | fit + writeup | ~20 min |
| | **total** | **≈3 h 15 m**, ~1.7 h slack |

Total compute ≈ 700 M tokens ≈ **3 node-hours**, but wall-clock is set by the longest arm (~50 min)
since all 10 run concurrently.

### Phase 0 — the one real cost
Short-heavy arms need far more short data than exists. `contradiction_train` is 19,366 examples /
419 M tokens over 1,986–40,957 tokens; its ≤8k slice is only ~60 M tokens, but arm A4 alone wants
160 M short tokens. Without new data, A4 would silently loop its pool ~3× while A0 did one pass —
an **epoch-count confound that would look exactly like a short-data effect**.

So Phase 0 generates additional short examples (`generate_pubmed_contradiction_data.py
--expand-from-train`, PubMedQA fillers, cached) so every arm draws its short tokens **without
repetition**. The long pool is sliced from the existing shard — no re-tokenization of long data.

**Pre-flight assertion:** every arm logs `unique_short_tokens_consumed / short_pool_tokens ≤ 1.0`
and the run aborts otherwise. Repetition must never be silent.

## 5. Honest risks

1. **Undertraining.** A0 is only ~10 min from base and may be near-floor, compressing the whole
   curve. Mitigation: A0/B0 are the *controls* — if they are at floor, the curve still reads, but
   absolute values sit below the 0.335 baseline. This buys a **direction**, not a production mix.
2. **Resolution.** SE ±0.021 plus seed noise → arms within ~0.04 are not separable on one seed. The
   design is powered for a large effect, which is the only kind that would justify changing the mix.
3. **Confound removed, one remains.** Long exposure is identical across arms, but arms differ in
   *total optimizer steps*, so a long-arm-only effect of "more steps" is entangled with "more short
   data". Row C (uniform mix at matched wall-clock/steps) is the control that catches this.
4. **Single task.** Contradiction only — the one task with a trustworthy vLLM baseline and a built
   32k rung.
5. **Cluster contention.** 10 concurrent nodes on jupiter, alongside the running dense-128k job
   (2 nodes). May queue; urgent priority assumed.
