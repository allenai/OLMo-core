# Base image comes with PyTorch, numpy, flash-attn
FROM ghcr.io/allenai/pytorch:2.4.0-cuda12.1-python3.11

# Install other dependencies, but not the source code.
COPY pyproject.toml
COPY src/olmo_core/__init__.py
COPY src/olmo_core/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y ai2-olmo-core && \
    rm -rf *

WORKDIR /app/olmo-core
