# Base image comes with PyTorch, numpy, flash-attn
ARG BASE

#########################################################################
# Build image
#########################################################################

FROM ${BASE} as build

WORKDIR /app/build

# Install CUDA toolkit.
ARG CUDA_TOOLKIT_VERSION
RUN conda install -y -c nvidia cuda-toolkit==${CUDA_TOOLKIT_VERSION}

# Build megablocks and grouped-gemm.
ENV TORCH_CUDA_ARCH_LIST="8.0 9.0"
ENV GROUPED_GEMM_CUTLASS=1
ARG MEGABLOCKS_VERSION
RUN pip wheel --no-build-isolation --no-cache-dir "${MEGABLOCKS_VERSION}" \
    && rm -rf torch-*.whl numpy-*.whl triton-*.whl

#########################################################################
# Stable image
#########################################################################

FROM ${BASE} as stable

# Install torchao.
ARG TORCHAO_VERSION
RUN pip install --no-cache-dir ${TORCHAO_VERSION}

# Copy and install wheels from build image.
COPY --from=build /app/build /app/build
RUN pip install --no-cache-dir /app/build/*

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
