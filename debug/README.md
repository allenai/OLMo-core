# debug/ — one-off diagnostics (not part of the library)

Scripts written to root-cause a specific bug. Kept because each one is the *evidence* behind a record
in `records/`. Not imported by anything; safe to ignore unless you are re-checking that finding.

| script | what it proves |
|---|---|
| `flex_parity.py` | FlexAttention (`seq_len >= 8192`) is element-equivalent to the dense chunked mask: `mask_mod == build_chunked_allowed_mask` for chunked/standard/random_doc, and outputs agree to bf16 noise. Rules flex out as a cause of training collapse. |
| `flex_blockmask_cache_bug.py` | Probes the single-slot BlockMask cache in `document_chunked.py`, whose key is `(id(cids), data_ptr, _version, ...)`. `id()` **is** recycled across forwards; `data_ptr` currently is not, so the key does not collide **today** — but the key is identity-based and therefore fragile. |
| `build_free_varied.py` | Builds `contra_n100_v2_free60v`: the FREE-token budget of `free_pad_repeat` rebuilt from *distinct* sentences. Separates "more FREE positions" from "a block of exactly-repeated text". |
| `build_chunkpad.py` / `build_chunkpad2.py` | The within-chunk twin of the above (`contra_n100_v2_chunkpad{,2}`): the same varied filler placed *inside* each chunk. `chunkpad2` drops the claim index from the filler (`chunkpad` restated each claim's own number, which is a redundant index label, not neutral filler). |
| `collect_n100fix.py` | Collects the n100fix eval JSONs into one table (f1 + binomial SE + eval_size). |

See `records/n100-chunked-marker-position-bug.md` and `records/free-pad-probe-is-confounded.md`.
