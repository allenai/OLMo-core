# Base image comes with PyTorch, numpy, flash-attn
ARG BASE
FROM ${BASE} as stable

# Install torchao
ARG TORCHAO_VERSION
RUN pip install --no-cache-dir \
    torchao==${TORCHAO_VERSION} \
    git+https://git@github.com/databricks/megablocks.git@f4c08bc89e73b0b343e0815c680c4c9c9875302b

# Install other dependencies, but not the source code.
COPY pyproject.toml .
COPY src/olmo_core/__init__.py src/olmo_core/__init__.py
COPY src/olmo_core/version.py src/olmo_core/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y ai2-olmo-core && \
    rm -rf *

WORKDIR /app/olmo-core

FROM stable as nightly

ARG NIGHTLY_VERSION
RUN pip install --no-cache-dir --pre torch==${NIGHTLY_VERSION}
