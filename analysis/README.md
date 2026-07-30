# Attention-graph analysis

Count how many "connections" / information-flow paths exist between two documents given a
transformer's attention pattern, for different attention types, corpus sizes, and depths.

## The idea

Treat an attention mask as the adjacency matrix of a directed graph over tokens:

```
M[i, j] = 1   <=>   token i attends to token j   (i can read from j)
```

All masks here are **causal** (`M[i,j]` can be 1 only if `j <= i`).

An `L`-layer transformer that reuses this connectivity at every layer lets information
flow along **length-`L` walks** in that graph. The number of length-`L` walks from token
`a` to token `b` is exactly `(M ** L)[a, b]`. So the number of "direct paths" connecting
document `A` to document `B` in an `L`-layer model is the block sum:

```
paths(A, B; L) = sum_{a in A, b in B} (M ** L)[a, b]
```

**Worked example (matches the request).** One layer, docs A and B of 20 tokens, full
attention → `paths(A, B; 1) = 20 * 20 = 400`. Two layers over a corpus of `N` such docs →
each of the 400 endpoint pairs routes through up to `20 * N` intermediate tokens, giving up
to `400 * 20 * N` walks. With causality applied the code realizes the causal fraction of
that bound (e.g. 2 docs × 20 tok, L=2 → 8400 of the 16000 upper bound).

## Layout

```
analysis/
  attention_graphs/
    corpus.py   # Corpus: documents -> flat token sequence, doc<->token maps, landmarks
    masks.py    # idealized (T,T) boolean masks per attention type + a registry
    paths.py    # walk_count_matrix (M**L), doc-pair aggregation, sweeps, reachability
    viz.py      # mask heatmap, doc-path heatmap, sweep line plots
  run_analysis.py   # end-to-end demo -> writes figures to visualizations/attention_graphs/
```

## Attention types (idealized)

| name | rule (causal throughout) |
|------|--------------------------|
| `dense` | attend all earlier tokens |
| `sliding_window` | attend the last `window` positions |
| `doc_chunked` | attend only within own document — **no cross-doc edges** |
| `doc_chunked_landmark` | within-doc + attend earlier docs' **landmark** (last) tokens only (models `SparseLandmarkAttention`) |
| `global_tokens` | within-doc + a few per-doc **global/sink** tokens everyone reads |
| `hierarchical_dilated` | within-doc + attend earlier docs at exponentially dilated offsets (1,2,4,…) |
| `random_doc` | within-doc + each earlier document kept with prob `keep_prob` (BigBird-style, doc-granular) |

These capture *connectivity structure*, not the softmax weights. Note the repo's plain
grouped-softmax `landmark` reaches every earlier token (gated through the landmark), so its
connectivity is identical to `dense`; the graph-distinct variant is the sparse one above.
Add your own with the `@register("name")` decorator in `masks.py`.

## Quick start

```python
import sys; sys.path.insert(0, "analysis")
from attention_graphs import Corpus, build_mask, doc_pair_paths, average_cross_doc_paths

corpus = Corpus.uniform(num_docs=8, doc_len=20)          # or Corpus.random_lengths(...)
M = build_mask("doc_chunked_landmark", corpus, landmarks_per_doc=1)

D = doc_pair_paths(M, corpus, num_layers=2)              # (8,8) matrix, D[p,q] = paths p->q
avg = average_cross_doc_paths(M, corpus, num_layers=2)   # scalar: mean over doc pairs p!=q
```

Sweeps:

```python
from attention_graphs import sweep_layers, sweep_corpus_size
types = {"dense": {"name": "dense"}, "doc_chunked": {"name": "doc_chunked"},
         "landmark": {"name": "doc_chunked_landmark"}}
by_L = sweep_layers(corpus, types, layers=[1,2,3,4], metric="paths")      # or metric="reachable"
by_N = sweep_corpus_size(doc_len=16, corpus_sizes=[2,4,8,16], attn_types=types, num_layers=3)
```

Run the whole demo (writes ~17 PNGs to `visualizations/attention_graphs/`):

```bash
python analysis/run_analysis.py
```

## Metrics

- **`doc_pair_paths` / `average_cross_doc_paths`** — raw walk counts. Grow exponentially
  with depth (use the `symlog`/`log` plots). `reduce="mean"` gives per-token-pair average.
- **`doc_pair_reachable`** — fraction of token pairs connected by *any* walk of length ≤ L.
  Bounded in `[0,1]`; good for "at what depth do docs become reachable at all".

## Notes / caveats

- Walk counts use `float64` (they overflow int64 past a few layers). For very large `T`
  the `M ** L` matmul is `O(T^3)`; keep demo corpora to a few hundred–thousand tokens, or
  use `doc_pair_reachable` (boolean) for bigger graphs.
- Walks may revisit tokens (standard adjacency-power counting) — this matches the
  "20·(20N)·20" counting in the request.
- Direction: `D[p, q]` counts flow from a token in doc `p` to a token in doc `q`; with
  causal attention it is non-zero only when `p` is positioned at/after `q`.
