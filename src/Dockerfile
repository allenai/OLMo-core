#########################################################################
# Build image
#########################################################################

ARG BASE_BUILD
FROM ${BASE_BUILD} as build

WORKDIR /app/build

RUN pip install --upgrade --no-cache-dir pip wheel packing "setuptools<70.0.0" ninja

# Build megablocks, grouped-gemm, stanford-stk
ENV TORCH_CUDA_ARCH_LIST="8.0 9.0"
ENV GROUPED_GEMM_CUTLASS=1
ARG MEGABLOCKS_VERSION
RUN pip wheel --no-build-isolation --no-cache-dir "${MEGABLOCKS_VERSION}"

# Build flash-attn.
RUN pip wheel --no-build-isolation --no-cache-dir "flash-attn==2.6.3"

# Flash-attn from pre-built wheel (can't get this to work at the moment)
#RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# Only keep the target wheels and dependencies with CUDA extensions.
RUN echo "Built wheels:" \
    && ls -lh . \
    && ls -1 | grep -Ev 'megablocks|grouped_gemm|stanford_stk|flash_attn' | xargs rm \
    && echo "Final wheels:" \
    && ls -lh .

#########################################################################
# Stable image
#########################################################################

ARG BASE_RUNTIME
FROM ${BASE_RUNTIME} as stable

# Install torchao.
ARG TORCH_CUDA_VERSION
ARG TORCHAO_VERSION
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION} \
    ${TORCHAO_VERSION}

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
