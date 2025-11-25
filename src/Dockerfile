ARG UBUNTU_VERSION=22.04
ARG TARGET_PLATFORM=x86_64
ARG CUDA_VERSION=12.8.1
ARG CUDA_VERSION_PATH=cu128
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

#########################################################################
# Build image
#########################################################################

FROM ${DEVEL_BASE_IMAGE} AS build

WORKDIR /app/build

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        wget \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        gcc \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda, Python, and Python build dependencies.
ARG TARGET_PLATFORM
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:$PATH
RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${TARGET_PLATFORM}.sh"
# NOTE: Manually invoke bash on miniconda script per https://github.com/conda/conda/issues/10431
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja && \
    /opt/conda/bin/conda clean -ya

# Install PyTorch core ecosystem.
ARG CUDA_VERSION_PATH
ARG TORCH_VERSION=2.9.1
ARG INSTALL_CHANNEL=whl
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} torchao torchvision torchaudio

ENV TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0"

# Install grouped-gemm.
# NOTE: right now we need to build with CUTLASS so we can pass batch sizes on GPU.
# See https://github.com/tgale96/grouped_gemm/pull/21
ENV GROUPED_GEMM_CUTLASS="1"
ARG GROUPED_GEMM_VERSION="grouped_gemm @ git+https://git@github.com/tgale96/grouped_gemm.git@main"
RUN pip install --no-build-isolation --no-cache-dir "${GROUPED_GEMM_VERSION}"

# Install flash-attn 2
ARG FLASH_ATTN_VERSION=2.8.2
RUN pip install --no-build-isolation --no-cache-dir flash-attn==${FLASH_ATTN_VERSION}

# Install flash-attn 3
ARG FLASH_ATTN_3_SHA="92ca9da8d66f7b34ff50dc080ec0fef9661260d6"
ARG FA3_MAX_JOBS=16
RUN git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git fetch --depth 1 origin ${FLASH_ATTN_3_SHA} && \
    git checkout ${FLASH_ATTN_3_SHA} && \
    git submodule update --init --depth 1 && \
    cd hopper && \
    FLASH_ATTENTION_DISABLE_FP16=TRUE MAX_JOBS=${FA3_MAX_JOBS} python setup.py install && \
    cd /app/build && \
    rm -rf flash-attention

# Install transformer-engine.
ARG TE_VERSION=2.9
RUN pip install --no-build-isolation --no-cache-dir transformer-engine[pytorch]==${TE_VERSION}

# Install ring-flash-attn.
ARG RING_FLASH_ATTN_VERSION=0.1.8
RUN pip install --no-build-isolation --no-cache-dir ring-flash-attn==${RING_FLASH_ATTN_VERSION}

# Install liger-kernel.
ARG LIGER_KERNEL_VERSION=0.6.4
RUN pip install --no-build-isolation --no-cache-dir liger-kernel==${LIGER_KERNEL_VERSION}

# Install direct dependencies, but not source code.
COPY pyproject.toml .
COPY src/olmo_core/__init__.py src/olmo_core/__init__.py
COPY src/olmo_core/version.py src/olmo_core/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y ai2-olmo-core && \
    rm -rf *

#########################################################################
# Release image
#########################################################################

FROM ${BASE_IMAGE} AS release

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        wget \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        gcc \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install MLNX OFED user-space drivers
# See https://docs.nvidia.com/networking/pages/releaseview.action?pageId=15049785#Howto:DeployRDMAacceleratedDockercontaineroverInfiniBandfabric.-Dockerfile
ARG UBUNTU_VERSION
ARG TARGET_PLATFORM
ENV MOFED_VER="24.01-0.3.3.1"
RUN wget --quiet https://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}/mlnxofedinstall --basic --user-space-only --without-fw-update -q && \
    rm -rf MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM} && \
    rm MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}.tgz

# Copy conda environment.
COPY --from=build /opt/conda /opt/conda

ENV PATH=/opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

LABEL org.opencontainers.image.source=https://github.com/allenai/OLMo-core
WORKDIR /app/olmo-core
