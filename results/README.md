# results/

One markdown file per experiment family, holding the **current, bug-free** tables for that family.

Rules for this folder:

- **Only trustworthy numbers go in the main tables.** A number earned on a setup that was later found
  broken is either deleted or moved to an explicitly-labelled *Retracted* section — never left sitting
  in a results table where someone will quote it.
- Every table states its **setup line** (data shard, base checkpoint, steps, mask-mixing, eval set) so
  two tables are never silently incomparable.
- Claims that were made and then **overturned** are recorded, with what killed them. The point of this
  folder is that you can trust what's in it, which means the corrections have to be visible.
- `eval_size` is the number of EVAL examples; `n` is CORPUS size (n=100 documents). See CLAUDE.md.

| file | family |
|---|---|
| [masks-n100.md](masks-n100.md) | attention masks at n=100: chunked / hierarchical / random / full, + full-attention-layer hybrids |
| [goldgrad-n100.md](goldgrad-n100.md) | gold-gradient (sparse backward): which documents keep gradient |
| [gold-connectivity-stage0.md](gold-connectivity-stage0.md) | n=100 ablation: hier-K50's win over random-k0.5 IS gold-pair connectivity, not dilation-as-bias |
| [hopgold-n50-stage1.md](hopgold-n50-stage1.md) | multi-hop gold routing, n=50 gates: floor / ceiling / can a p=0.25 base route at all |
| [hopgold-n50-connectivity-preview.md](hopgold-n50-connectivity-preview.md) | zero-GPU preview of the hop question, by conditioning the p=0.25 arm on realized per-pair connectivity |
| [hopgold-n50-summary-ladder.md](hopgold-n50-summary-ladder.md) | summary-attention relay: does a dedicated 2-hop relay carry information, and how much bandwidth does it need |

Longer prose diagnoses of *bugs* (as opposed to results) live in `records/`.
