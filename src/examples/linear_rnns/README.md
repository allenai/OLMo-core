# Running benchmarks

Example invocation:

```bash
for prefill_length in 32768 16384 8192 4096 2048 1024; do
    echo "Running benchmark with prefill_length=$prefill_length"
    python3 src/examples/linear_rnns/benchmark.py \
        --run_name=hybrid-gated-deltanet_prefill${prefill_length} \
        --path=/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/hybrid-gated-deltanet-neg/step23842/ \
        --output-dir=benchmark_outputs \
        --generate-length=1024 \
        --prefill-length=$prefill_length \
        --batch-size=8 \
        --n-batches=6

    python3 src/examples/linear_rnns/benchmark.py \
        --run_name=control_prefill${prefill_length} \
        --path=/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/control-test/step23842/ \
        --output-dir=benchmark_outputs \
        --generate-length=1024 \
        --prefill-length=$prefill_length \
        --batch-size=8 \
        --n-batches=6

    python3 src/examples/linear_rnns/benchmark.py \
        --run_name=gated-deltanet-neg_prefill${prefill_length} \
        --path=/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/gated-deltanet-neg/step23842/ \
        --output-dir=benchmark_outputs \
        --generate-length=1024 \
        --prefill-length=$prefill_length \
        --batch-size=8 \
        --n-batches=6
done
```