ARG BASE_BUILD
ARG BASE_RUNTIME

#########################################################################
# Build image
#########################################################################

FROM ${BASE_BUILD} as build

WORKDIR /app/build

# Install system dependencies.
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        libxml2-dev \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install/upgrade Python build dependencies.
RUN pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja

# Build flash-attn.
ARG FLASH_ATTN_VERSION
RUN pip wheel --no-build-isolation --no-cache-dir flash-attn==${FLASH_ATTN_VERSION}

# Build megablocks, grouped-gemm, stanford-stk
ENV TORCH_CUDA_ARCH_LIST="8.0 9.0"
ENV GROUPED_GEMM_CUTLASS=1
ARG MEGABLOCKS_VERSION
RUN pip wheel --no-build-isolation --no-cache-dir "${MEGABLOCKS_VERSION}"

# Only keep the target wheels and dependencies with CUDA extensions.
RUN echo "Built wheels:" \
    && ls -lh . \
    && ls -1 | grep -Ev 'megablocks|grouped_gemm|stanford_stk|flash_attn' | xargs rm \
    && echo "Final wheels:" \
    && ls -lh .

#########################################################################
# Stable image
#########################################################################

FROM ${BASE_RUNTIME} as stable

# Install system dependencies.
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        libxml2-dev \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install/upgrade Python build dependencies.
RUN pip install --upgrade --no-cache-dir pip wheel packaging

# Install torchao.
ARG TORCH_CUDA_VERSION
ARG TORCHAO_VERSION
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION} \
    torchao==${TORCHAO_VERSION}

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

ARG TORCH_CUDA_VERSION
ARG NIGHTLY_VERSION
RUN pip install --no-cache-dir --pre \
    --index-url https://download.pytorch.org/whl/nightly/cu${TORCH_CUDA_VERSION} \
    torch==${NIGHTLY_VERSION}
