## Logging in

SSH into one of the login nodes:

```bash
ssh ubuntu@192.222.61.240
# OR: ssh ubuntu@192.222.60.249
```

Start or attach your own tmux session:

```bash
username=petew
tmux new-session -A -s $username
```

Check that `/data/ai2/bin/` is in your `PATH`, and if not add it:

```bash
export PATH="/data/ai2/bin:$PATH"
```

## Initial setup

We all login as the same user, so we need to be careful to use unique repo directories and virtual environment.

```bash
# Set your unique username (ideally use your Ai2/Beaker username).
username=petew

# Clone repo to unique directory.
cd "/data/ai2/$username"
git clone https://github.com/allenai/OLMo-core.git
cd OLMo-core

# Create unique virtual environment.
uv venv --python=3.12 /data/ai2/uv/OLMo-core-$username
source "/data/ai2/uv/OLMo-core-${username}/bin/activate"
uv pip install --torch-backend=cu129 numpy torch==2.8.0 torchvision torchaudio torchao==0.15.0
uv pip install "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@0abbe028dfb5f033b35eb6da6fc6924accb0dc7a"
uv pip install -e '.[all]'
```

## Run a test job

Submit a job through SLURM with:

```bash
./src/scripts/lambda/launch.sh ./src/scripts/lambda/slurm-test-job.sbatch
```

The first argument to the `launch.sh` is the sbatch script to run.
This will print out the job ID, wait for it to start, and then stream the logs.
