## Logging in

SSH into one of the login nodes.
Pick a unique username to identify your environment (e.g., your Ai2/Beaker/GitHub username).
Then start or attach your own tmux session:

```bash
USERNAME=petew
tmux new-session -A -s $USERNAME
```

Export the following environment variables:

```bash
export USERNAME=petew
export WANDB_API_KEY=XXXX
export GITHUB_TOKEN=XXXX
export GOOGLE_APPLICATION_CREDENTIALS=/data/$USERNAME/google-credentials.json
```

And copy your Google credentials JSON file to that location (`/data/$USERNAME/google-credentials.json`).

Optionally, export a `SLACK_WEBHOOK_URL` for alerting.

```bash
export SLACK_WEBHOOK_URL=XXXX
```

## Initial setup

We all log in as the same user, so we need to be careful to use unique repo directories and virtual environment, which we'll identify by our Ai2/Beaker/GitHub username (set to the `USERNAME` env var).

```bash
# Use GitHub CLI as credential helper for git to clone our private fork.
gh auth setup-git

# Clone repo to unique directory.
cd "/data/$USERNAME"
gh repo clone allenai/RI-OLMo-core
cd RI-OLMo-core

# Create unique virtual environment.
uv venv --python=3.12 /data/$USERNAME/uv/OLMo-core
source "/data/$USERNAME/uv/OLMo-core/bin/activate"
uv pip install --torch-backend=cu129 numpy torch==2.9.1 torchvision torchaudio torchao==0.15.0
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
uv pip install https://github.com/windreamer/flash-attention3-wheels/releases/download/2025.12.29-c947cd1/flash_attn_3-3.0.0b1%2B20251229.cu128torch291cxx11abitrue.58fe37-cp39-abi3-linux_x86_64.whl
uv pip install -e '.[all]'
```

You may also need these version constraints in order for evaluation to work:

```bash
# fix pyarrow issue in evalutor:
uv pip install -U "datasets>=2.20.0"
# fox torchmetrics issue in evaluator:
uv pip install -U "huggingface-hub>=0.34.0,<1.0"
```

## Run a test job

Submit a job through SLURM with:

```bash
./src/scripts/prime-intellect/launch.sh ./src/scripts/prime-intellect/slurm-test-job.sbatch test-001 1
```

The first argument to the `launch.sh` is the sbatch script to run.
The second is a name to assign to the run.
The third is how many nodes to use.
This will print out the job ID, wait for it to start, and then stream the logs.

## Running your own script

Copy the test sbatch script `slurm-test-job.sbatch` and modifying to your needs.
Then launch it with the `./src/scripts/prime-intellect/launch.sh` script in the same way as we did in the previous section.

## Running a job with retries

For long training jobs that should be automatically resubmitted after failures, use the `launch_with_retries.sh` script
instead of `launch.sh`. It takes the same arguments but will continually resubmit the job until it completes successfully,
and send Slack notifications on failures if the `SLACK_WEBHOOK_URL` environment variable is set.

## How to restart nodes

If you check the status of the Slurm cluster with `sinfo` and any nodes show something other than `idle` or the usual healthy states, follow the steps below.

**If a node is in `down*` status:**

The node is most likely not reachable and needs a full restart. Ping the Prime Intellect team in this case.

**If a node is in `down` status:**

The node is reachable and just needs to be re-added to the Slurm cluster.
Run the following, replacing "hostname" with the correct node name:

```jsx
sudo -i scontrol update NodeName=hostname State=RESUME
```
