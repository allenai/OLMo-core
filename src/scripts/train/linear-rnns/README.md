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

To launch the big OLMo 3.1 7B run:

```shell
python src/scripts/train/OLMo3/OLMo3-7B.py launch OLMo3-7B ai2/jupiter --launch.num_nodes=2

python src/scripts/train/linear-rnns/OLMo3-7B-hybrid.py launch OLMo3-7B-hybrid ai2/augusta --launch.num_nodes=2

while true; do
    python src/scripts/train/linear-rnns/OLMo3.1-7B-hybrid.py launch OLMo3.1-7B-6T-30h ai2/augusta \
        --launch.use_hostname_constraints=true \
        --launch.priority="urgent" \
        --launch.num_nodes=64 \
        --trainer.callbacks.profiler.enabled=true --trainer.callbacks.profiler.wait=10 # profiler
done
```