import matplotlib.pyplot as plt
import torch
import wandb

wandb.init(
    project="olmoe-dev",
    name="routing_table_logging",
)

# Example: 32 layers, 64 experts
num_layers, num_experts = 32, 64

# After each logging step
for step in range(10):  # Simulating 10 logging steps
    token_counts = torch.randint(0, 200, (num_layers, num_experts))  # fake data

    for layer_id in range(token_counts.shape[0]):
        repeat = token_counts[layer_id]
        idx = torch.repeat_interleave(
            torch.arange(repeat.size(0)), repeat  # [0, 1, 2]  # repeat counts
        )

        wandb.log(
            {f"routing_hist/layer_{layer_id}": wandb.Histogram(idx.tolist())},
            step=step,
            commit=False,  # accumulate all layer histograms in a single dashboard point
        )
    wandb.log({}, step=step)  # flush commit
