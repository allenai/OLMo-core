ARG UBUNTU_VERSION=24.04
ARG TARGET_PLATFORM=x86_64
ARG CUDA_VERSION=12.8.1
ARG CUDA_VERSION_PATH=cu128
ARG PYTHON_VERSION=3.12
# CUDA 12.8 nvcc does not support sm_103. For CUDA 13/B300-native builds,
# override with: --build-arg TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0 10.3"
ARG TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0"
ARG NVIDIA_NCCL_CU13_VERSION=2.30.7
ARG NVIDIA_NVSHMEM_CU13_VERSION=3.7.0
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
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
# Optional cuBLAS override for package mismatches. For example, TE cu13 wheels
# reference cublasLtGroupedMatrixLayoutInit_internal, which is not in cuBLAS 13.1.
ARG NVIDIA_CUBLAS_VERSION=""
# DeepEP V2's Gin backend needs NCCL >= 2.30.4 headers/runtime. Keep these
# as package-level overrides only; the DeepEP source itself is not baked in.
ARG NVIDIA_NCCL_CU13_VERSION
ARG NVIDIA_NVSHMEM_CU13_VERSION
ARG INSTALL_CHANNEL=whl
RUN set -eux; \
    torch_packages="torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}"; \
    if [ -n "${TORCHAUDIO_VERSION}" ]; then \
        torch_packages="${torch_packages} torchaudio==${TORCHAUDIO_VERSION}"; \
    fi; \
    pip install --no-cache-dir --index-url "https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/" ${torch_packages}; \
    if [ -n "${NVIDIA_CUBLAS_VERSION}" ]; then \
        pip install --no-cache-dir --no-deps --upgrade "nvidia-cublas==${NVIDIA_CUBLAS_VERSION}"; \
    fi; \
    if [ -n "${NVIDIA_NCCL_CU13_VERSION}" ]; then \
        pip install --no-cache-dir --no-deps --upgrade "nvidia-nccl-cu13==${NVIDIA_NCCL_CU13_VERSION}"; \
    fi; \
    if [ -n "${NVIDIA_NVSHMEM_CU13_VERSION}" ]; then \
        pip install --no-cache-dir --no-deps --upgrade "nvidia-nvshmem-cu13==${NVIDIA_NVSHMEM_CU13_VERSION}"; \
    fi

ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Install grouped-gemm.
# NOTE: right now we need to build with CUTLASS so we can pass batch sizes on GPU.
# See https://github.com/tgale96/grouped_gemm/pull/21
# ENV GROUPED_GEMM_CUTLASS="1"
ARG GROUPED_GEMM_SHA="f1429a3c44c98f7912aa4b00125144cdf4e7fdb2"
RUN unset GROUPED_GEMM_CUTLASS && pip install --no-build-isolation --no-cache-dir "grouped_gemm @ git+https://git@github.com/tgale96/grouped_gemm.git@${GROUPED_GEMM_SHA}"

# Install flash-attn 2
ARG FLASH_ATTN_VERSION=2.8.2
ARG FLASH_ATTN_CUDA_ARCHS="90;100"
RUN FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" pip install --no-build-isolation --no-cache-dir flash-attn==${FLASH_ATTN_VERSION}

# Install flash-attn 3.
ARG FLASH_ATTN_3_SHA="92ca9da8d66f7b34ff50dc080ec0fef9661260d6"
ARG FA3_MAX_JOBS=16
RUN git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git fetch --depth 1 origin ${FLASH_ATTN_3_SHA} && \
    git checkout ${FLASH_ATTN_3_SHA} && \
    git submodule update --init --depth 1 && \
    cd hopper && \
    FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=${FA3_MAX_JOBS} python setup.py install && \
    cd /app/build && \
    rm -rf flash-attention

# flash-attn 3 will be installed to '/opt/conda/lib/python3.12/site-packages/flash_attn_3-*.egg/'
# and will only be importable as 'flash_attn_interface'. However, TE expects to import this
# as 'flash_attn_3.flash_attn_interface'. So we modify the package directory structure to adhere
# to that.
RUN cd /opt/conda/lib/python${PYTHON_VERSION}/site-packages/ && \
    mkdir -p flash_attn_3 && \
    mv flash_attn_3-*.egg/flash_attn_3/* flash_attn_3/ && \
    mv flash_attn_3-*.egg/flash_attn_interface.py flash_attn_3/ && \
    touch flash_attn_3/__init__.py && \
    rm -rf flash_attn_3-*

# Install flash-attn 4.
ARG FLASH_ATTN_4_VERSION=4.0.0b12
ARG FLASH_ATTN_4_EXTRAS=""
RUN python -c "import torch; print('torch=='+torch.__version__)" > /tmp/torch-constraint.txt && \
    python -m pip install --no-cache-dir -c /tmp/torch-constraint.txt "flash-attn-4${FLASH_ATTN_4_EXTRAS}==${FLASH_ATTN_4_VERSION}" && \
    rm -f /tmp/torch-constraint.txt

# Install transformer-engine.
ARG TE_VERSION=2.14
RUN pip install --no-build-isolation --no-cache-dir transformer-engine[pytorch]==${TE_VERSION}

# Install ring-flash-attn.
ARG RING_FLASH_ATTN_VERSION=0.1.8
RUN pip install --no-build-isolation --no-cache-dir ring-flash-attn==${RING_FLASH_ATTN_VERSION}

# Install liger-kernel.
ARG LIGER_KERNEL_VERSION=0.6.4
RUN pip install --no-build-isolation --no-cache-dir liger-kernel==${LIGER_KERNEL_VERSION}

# Optionally install Quack kernels.
ARG QUACK_VERSION=""
RUN if [ -n "${QUACK_VERSION}" ]; then \
    pip install --no-build-isolation --no-cache-dir quack-kernels==${QUACK_VERSION}; fi

# Install direct dependencies, but not source code.
COPY pyproject.toml .
COPY src/olmo_core/__init__.py src/olmo_core/__init__.py
COPY src/olmo_core/version.py src/olmo_core/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y ai2-olmo-core && \
    rm -rf *

# Install post-training data/tokenization dependencies used by the OLMo-core SFT
# scripts and the native MoE DPO path in Open Instruct. Keep RLVR-only packages
# such as ray, vLLM, DeepSpeed, peft, and docker out of this image layer.
RUN python -c "import torch; print('torch=='+torch.__version__)" > /tmp/torch-constraint.txt && \
    pip install --no-cache-dir -c /tmp/torch-constraint.txt \
        "datasets==4.5.0" \
        "hf-transfer==0.1.9" \
        "huggingface-hub==1.8.0" \
        "multiprocess==0.70.17" \
        "pyarrow==23.0.0" \
        "sentencepiece==0.2.1" \
        "tokenizers==0.22.2" \
        "tqdm==4.67.1" \
        "transformers==5.4.0" \
        "wandb==0.23.1" && \
    rm -f /tmp/torch-constraint.txt

#########################################################################
# Release image
#########################################################################

# FROM ${BASE_IMAGE} AS release
FROM ${DEVEL_BASE_IMAGE} AS release
ARG NVIDIA_CUBLAS_VERSION=""

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        wget \
        unzip \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        rdma-core \
        libibverbs1 \
        libibverbs-dev \
        ibverbs-providers \
        ibverbs-utils \
        librdmacm1 \
        librdmacm-dev \
        perftest \
        gnupg \
        jq \
        gcc \
        net-tools \
        openssh-server \
        git-lfs \
        bc \
        sudo \
        htop \
        tmux \
        vmtouch \
        bwm-ng \
        git && \
    rm -rf /var/lib/apt/lists/* \
    # AWS CLI \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm awscliv2.zip \
    # Google Cloud SDK \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
        | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update \
    && apt-get install -y google-cloud-sdk \
    && rm -rf /var/lib/apt/lists/* \
    # GitHub CLI \
    && curl -sS https://webi.sh/gh | sh \
    # uv \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Optional Podman-in-Docker support for Beaker subcontainers. This expects the
# Beaker job to be launched with BEAKER_ALLOW_SUBCONTAINERS=1 and
# BEAKER_SKIP_DOCKER_SOCKET=1. The containers.conf profile mirrors Ai2's
# cuda custom-podman image and avoids nested networking/cgroup/proc mount
# failures inside Beaker.
ARG INSTALL_PODMAN=0
ARG PODMAN_VERSION=5.6.2
ARG CRUN_VERSION=1.14.3
COPY src/podman/ /tmp/olmo-core-podman/
RUN set -eux; \
    if [ "${INSTALL_PODMAN}" = "1" ]; then \
        apt-get update; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
          autoconf \
          automake \
          conmon \
          gcc \
          go-md2man \
          golang-go \
          iptables \
          libassuan-dev \
          libbtrfs-dev \
          libc6-dev \
          libcap-dev \
          libdevmapper-dev \
          libglib2.0-dev \
          libgpg-error-dev \
          libgpgme-dev \
          libprotobuf-c-dev \
          libprotobuf-dev \
          libseccomp-dev \
          libselinux1-dev \
          libsystemd-dev \
          libtool \
          libyajl-dev \
          netavark \
          passt \
          pkg-config \
          python3-sphinx \
          systemd \
          uidmap; \
        mkdir -p /etc/containers/registries.conf.d; \
        cp /tmp/olmo-core-podman/containers.conf /etc/containers/containers.conf; \
        cp /tmp/olmo-core-podman/policy.json /etc/containers/policy.json; \
        cp /tmp/olmo-core-podman/10-unqualified-search-registries.conf /etc/containers/registries.conf.d/10-unqualified-search-registries.conf; \
        curl -fsSL "https://github.com/containers/podman/archive/refs/tags/v${PODMAN_VERSION}.tar.gz" \
          | tar xz -C /tmp; \
        make -C "/tmp/podman-${PODMAN_VERSION}" BUILDTAGS="selinux seccomp" PREFIX=/usr; \
        make -C "/tmp/podman-${PODMAN_VERSION}" install PREFIX=/usr; \
        git clone --depth 1 -b "${CRUN_VERSION}" https://github.com/containers/crun.git /tmp/crun; \
        cd /tmp/crun; \
        ./autogen.sh; \
        ./configure --prefix=/usr --sysconfdir=/etc; \
        make; \
        make install; \
        cd /; \
        cp /tmp/olmo-core-podman/setup_dockerio_mirror /usr/local/bin/setup_dockerio_mirror; \
        chmod +x /usr/local/bin/setup_dockerio_mirror; \
        ln -sf "$(command -v podman)" /usr/local/bin/docker; \
        grep -q '^root:10000:11165536$' /etc/subuid || echo 'root:10000:11165536' >> /etc/subuid; \
        grep -q '^root:10000:11165536$' /etc/subgid || echo 'root:10000:11165536' >> /etc/subgid; \
        mkdir -p /run/user/0/libpod/tmp /run/podman; \
        chmod 700 /run/user/0; \
        rm -rf /var/lib/apt/lists/* /tmp/olmo-core-podman "/tmp/podman-${PODMAN_VERSION}" /tmp/crun; \
    else \
        rm -rf /tmp/olmo-core-podman; \
    fi

# Minimal DOCA RDMA user-space for NCCL IB/GDRDMA.
# Keep this narrower than doca-ofed/doca-ofed-userspace: kernel drivers and DKMS
# belong to the host, while the container only needs verbs libraries and providers.
ARG UBUNTU_VERSION
ARG INSTALL_DOCA_RDMA=0
ARG DOCA_VERSION=3.3.0
ARG DOCA_RDMA_CORE_VERSION=2601.0.7-1
ARG DOCA_PERFTEST_VERSION=26.01.5-1
RUN set -eux; \
    if [ "${INSTALL_DOCA_RDMA}" = "1" ]; then \
        curl -fsSL https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub \
          | gpg --batch --yes --dearmor -o /usr/share/keyrings/doca-mellanox.gpg; \
        echo "deb [signed-by=/usr/share/keyrings/doca-mellanox.gpg] https://linux.mellanox.com/public/repo/doca/${DOCA_VERSION}/ubuntu${UBUNTU_VERSION}/x86_64 /" \
          > /etc/apt/sources.list.d/doca.list; \
        apt-get update; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
          rdma-core=${DOCA_RDMA_CORE_VERSION} \
          libibverbs1=${DOCA_RDMA_CORE_VERSION} \
          libibverbs-dev=${DOCA_RDMA_CORE_VERSION} \
          ibverbs-providers=${DOCA_RDMA_CORE_VERSION} \
          librdmacm1=${DOCA_RDMA_CORE_VERSION} \
          librdmacm-dev=${DOCA_RDMA_CORE_VERSION} \
          ibverbs-utils=${DOCA_RDMA_CORE_VERSION} \
          perftest=${DOCA_PERFTEST_VERSION}; \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Copy conda environment.
COPY --from=build /opt/conda /opt/conda

# Keep these for non-interactive processes (docker run / docker exec / VS Code server)
ARG PYTHON_VERSION
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Prefer CUDA libraries bundled with PyTorch wheels over the base image's
# /usr/local/cuda libraries so extension wheels resolve against matching libs.
ENV PYTORCH_NVIDIA_LIB=/opt/conda/lib/python${PYTHON_VERSION}/site-packages/nvidia
ENV EP_NCCL_ROOT_DIR=${PYTORCH_NVIDIA_LIB}/nccl
ENV EP_NVSHMEM_ROOT_DIR=${PYTORCH_NVIDIA_LIB}/nvshmem
ENV LD_LIBRARY_PATH=${PYTORCH_NVIDIA_LIB}/nccl/lib:${PYTORCH_NVIDIA_LIB}/nvshmem/lib:${PYTORCH_NVIDIA_LIB}/cu13/lib:${PYTORCH_NVIDIA_LIB}/cublas/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/root/.local/bin:/root/.cargo/bin:/opt/conda/bin:$PATH

# One canonical env script (login shells) + source it for interactive non-login bash shells (VS Code terminal)
RUN set -eux; \
  cat > /etc/profile.d/99-container-env.sh <<EOF
# Container-wide env for login shells (SSH, su -) and (via bash.bashrc below) interactive shells.
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/root/.local/bin:/root/.cargo/bin:/opt/conda/bin:\${PATH}
export PYTORCH_NVIDIA_LIB=/opt/conda/lib/python${PYTHON_VERSION}/site-packages/nvidia
export EP_NCCL_ROOT_DIR=\${PYTORCH_NVIDIA_LIB}/nccl
export EP_NVSHMEM_ROOT_DIR=\${PYTORCH_NVIDIA_LIB}/nvshmem
export LD_LIBRARY_PATH=\${PYTORCH_NVIDIA_LIB}/nccl/lib:\${PYTORCH_NVIDIA_LIB}/nvshmem/lib:\${PYTORCH_NVIDIA_LIB}/cu13/lib:\${PYTORCH_NVIDIA_LIB}/cublas/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\${LD_LIBRARY_PATH}
export UV_SYSTEM_PYTHON=1
export UV_BREAK_SYSTEM_PACKAGES=1
EOF
RUN chmod 0644 /etc/profile.d/99-container-env.sh && \
  grep -q '99-container-env.sh' /etc/bash.bashrc || \
  printf '\n# Load container env for interactive shells\n[ -r /etc/profile.d/99-container-env.sh ] && . /etc/profile.d/99-container-env.sh\n' >> /etc/bash.bashrc

LABEL org.opencontainers.image.source=https://github.com/allenai/OLMo-core
WORKDIR /app/olmo-core


# additional tools
RUN pip install nvitop nvtx matplotlib pybind11 tensorboard

# nsys
# the latest version has problems with nvtx; pinning to 2025.5.1 
ARG UBUNTU_VERSION
RUN set -eux; \
    . /etc/os-release; \
    UBUNTU_SHORT="$(echo "$UBUNTU_VERSION" | tr -d .)"; \
    ARCH="$(dpkg --print-architecture)"; \
    echo "deb [trusted=yes] https://developer.download.nvidia.com/devtools/repos/ubuntu${UBUNTU_SHORT}/${ARCH}/ /" \
      > /etc/apt/sources.list.d/nvidia-devtools.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends nsight-systems-cli-2025.5.1; \
    rm -rf /var/lib/apt/lists/*

# install beaker
RUN wget https://beaker.org/install -O /tmp/install-beaker.sh && \
    bash /tmp/install-beaker.sh && \
    rm /tmp/install-beaker.sh

# Keep this after all release-stage Python installs. PyTorch's cu13 wheel
# currently constrains some NVIDIA packages below what DeepEP V2 / TE need.
ARG NVIDIA_NCCL_CU13_VERSION
ARG NVIDIA_NVSHMEM_CU13_VERSION
RUN set -eux; \
    if [ -n "${NVIDIA_CUBLAS_VERSION}" ]; then \
        pip install --no-cache-dir --no-deps --upgrade "nvidia-cublas==${NVIDIA_CUBLAS_VERSION}"; \
    fi; \
    if [ -n "${NVIDIA_NCCL_CU13_VERSION}" ]; then \
        pip install --no-cache-dir --no-deps --upgrade "nvidia-nccl-cu13==${NVIDIA_NCCL_CU13_VERSION}"; \
    fi; \
    if [ -n "${NVIDIA_NVSHMEM_CU13_VERSION}" ]; then \
        pip install --no-cache-dir --no-deps --upgrade "nvidia-nvshmem-cu13==${NVIDIA_NVSHMEM_CU13_VERSION}"; \
    fi
