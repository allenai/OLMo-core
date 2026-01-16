ARG UBUNTU_VERSION=22.04
ARG TARGET_PLATFORM=x86_64
ARG CUDA_VERSION=12.8.1
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

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set Python version for uv
ARG PYTHON_VERSION
ENV UV_PYTHON=${PYTHON_VERSION}

# Set CUDA arch for torch extensions
ENV TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0"

# Sync all dependencies (uv handles Python installation automatically)
COPY pyproject.toml uv.lock ./
COPY src/olmo_core/__init__.py src/olmo_core/__init__.py
COPY src/olmo_core/version.py src/olmo_core/version.py
RUN uv sync --frozen --extra all && \
    uv pip uninstall ai2-olmo-core && \
    rm -rf pyproject.toml uv.lock src

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

# Copy uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy the virtual environment from build stage
COPY --from=build /app/build/.venv /app/.venv

ENV PATH=/app/.venv/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

LABEL org.opencontainers.image.source=https://github.com/allenai/OLMo-core
WORKDIR /app/olmo-core
