# Training Linear RNNs and Hybrid Models

## 1B and 7B Experiments

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

Some examples of how to launch 7B hybrid models:

```shell
python src/scripts/train/OLMo3/OLMo3-7B.py launch OLMo3-7B ai2/jupiter --launch.num_nodes=2
python src/scripts/train/linear-rnns/OLMo3-7B-hybrid.py launch OLMo3-7B-hybrid ai2/augusta --launch.num_nodes=2
```

## 6T Token Run

First you will need to authenticate with Google because this config enforces hostname constraints for performance:

```shell
gcloud auth application-default login
gcloud config set project h100-cluster-owner
```

This may fail if you don't have permissions.
It could also be the case that there are no hosts satisfying these constraints, resulting in an `BeakerInsufficientResourcesError`.
In either case, you can add `--launch.use_hostname_constraints=false` to disable hostname constraints (which means you won't need to authenticate).

At this point, you can launch the run inside a while loop as follows so it immediately restarts when it fails:

```shell
while true; do
    python src/scripts/train/linear-rnns/OLMo3.1-7B-hybrid.py launch OLMo3.1-7B-6T-30h ai2/augusta \
        --launch.priority="urgent" \
        --launch.num_nodes=64 \
        --train_module.dp_config.name=hsdp \
        --train_module.dp_config.shard_degree=64 \
        --launch.beaker_image=tylerr/olmo-core-tch270cu128-2025-09-24 \
        --model.block.attention.backend=flash_3
done
```

### Optimizing Throughput

To enable the profiler, you can add the following flags:

```shell
--trainer.callbacks.profiler.enabled=true --trainer.callbacks.profiler.wait=10
```

Pete recommended always setting `num_execution_units=1`.
Beyond that, it's worth experimenting a bit with FSDP vs. HSDP (and different shard degrees), e.g.:

```shell
--train_module.dp_config.name=hsdp \
--train_module.dp_config.shard_degree=64
```

To run with Flash Attention 3 instead of 2, add:

```shell
--launch.beaker_image=tylerr/olmo-core-tch270cu128-2025-09-24 \
--model.block.attention.backend=flash_3
```