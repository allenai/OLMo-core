# Commands to launch

```shell
python src/scripts/train/linear-rnns/1b/control.py launch control ai2/jupiter

# Runs of different DeltaNet variants.
python src/scripts/train/linear-rnns/1b/gated_deltanet.py launch gated-deltanet ai2/jupiter
python src/scripts/train/linear-rnns/1b/gated_deltanet++.py launch gated-deltanet-neg ai2/jupiter
python src/scripts/train/linear-rnns/1b/deltanet.py launch deltanet ai2/jupiter
python src/scripts/train/linear-rnns/1b/deltanet++.py launch deltanet-neg ai2/jupiter

# Hybrid model alternating layers.
python src/scripts/train/linear-rnns/1b/hybrid_gated_deltanet++.py launch hybrid-gated-deltanet-neg ai2/jupiter
```