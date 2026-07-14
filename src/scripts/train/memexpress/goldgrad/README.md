# goldgrad/ — gold-gradient KV-detach probe (local)
Backward through only the gold docs' KV: train (sbatch+sh), eval, speed bench, reap helper.
Finding: grad through 8-16/100 docs ≈ full f1, killer variable is gold over-representation;
measured speedup 1.00x → it's a probe, not an O(1) backward. See goldgrad memory + results-hub.
