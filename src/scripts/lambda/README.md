## Initial setup

We all login as the same user, so we need to be careful to use unique repo directories and virtual environment.

```bash
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
