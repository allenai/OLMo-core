#########################################################################
# Stable image
#########################################################################

# Base image comes with PyTorch, numpy, flash-attn
ARG BASE
FROM ${BASE} as stable

# Install torchao.
ARG TORCHAO_VERSION
RUN pip install --no-cache-dir ${TORCHAO_VERSION}

# Install direct dependencies, but not source code.
COPY pyproject.toml .
COPY src/olmo_core/__init__.py src/olmo_core/__init__.py
COPY src/olmo_core/version.py src/olmo_core/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y ai2-olmo-core && \
    rm -rf *

WORKDIR /app/olmo-core

#########################################################################
# Nightly image
#########################################################################

FROM stable as nightly

ARG NIGHTLY_VERSION
RUN pip install --no-cache-dir --pre torch==${NIGHTLY_VERSION}

#########################################################################
# Dev image
#########################################################################

FROM nightly as dev

# Install core dev dependencies.
ENV TORCH_CUDA_ARCH_LIST=8.0
ENV GROUPED_GEMM_CUTLASS=1
ARG MEGABLOCKS_VERSION
RUN pip install --no-cache-dir "${MEGABLOCKS_VERSION}"
