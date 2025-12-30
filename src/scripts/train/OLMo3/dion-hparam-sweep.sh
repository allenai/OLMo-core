#!/bin/bash

# Learning rates to sweep
LRS=(2.5e-3 1.94e-2)  # too small: 1.0e-3 1.94e-3  5.0e-4

# Batch sizes to sweep (in tokens)
BATCH_SIZES=(1048576 2097152)  # too small: 262144 524288

for lr in "${LRS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        # Format batch size for run name (e.g., 524288 -> 524k)
        if [ $bs -ge 1048576 ]; then
            bs_name="$((bs / 1048576))M"
        else
            bs_name="$((bs / 1024))k"
        fi

        # Format lr for run name
        lr_name=$(echo $lr | sed 's/e-/e-/')

        run_name="dion-190M-lr${lr_name}-bs${bs_name}"

        echo "Launching: $run_name"
        uv run src/scripts/train/OLMo3/OLMo3-190M.py launch \
            "$run_name" \
            ai2/saturn \
            --train_module.optim.lr=$lr \
            --data_loader.global_batch_size=$bs
    done
done