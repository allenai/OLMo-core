.PHONY : checks
checks : style-check lint-check type-check

.PHONY : style-check
style-check :
	@echo "======== running isort... ========"
	@isort --check .
	@echo "======== running black... ========"
	@black --check .

.PHONY : lint-check
lint-check :
	@echo "======== running ruff... ========="
	@ruff check .

.PHONY : type-check
type-check :
	@echo "======== running mypy... ========="
	@mypy src/

.PHONY : style
style:
	@echo "======== formatting with isort... ========"
	@isort .
	@echo "======== formatting with black... ========"
	@black .

.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch src/olmo_core/ --watch README.md docs/source/ docs/build/

.PHONY : build
build :
	rm -rf *.egg-info/ dist/
	python -m build
	python src/scripts/release/check_dist_metadata.py dist/*

####################################################################################################
# Docker build
####################################################################################################

#-----------------#
# Build variables #
#-----------------#

# NOTE: When upgrading dependency versions (like for torch) make sure:
#  * The corresponding versions specified in 'pyproject.toml' include the new version.
#  * The versions installed in '.github/actions/setup-python-env/action.yml' match if necessary.
# NOTE: See https://hub.docker.com/r/nvidia/cuda/tags?name=devel-ubuntu22.04 for available CUDA versions.
CUDA_VERSION = 12.8.1
CUDA_VERSION_PATH=cu$(shell echo $(CUDA_VERSION) | cut -d"." -f1-2 | tr -d .)
CUDA_NVCC_VERSION = 12.8.93
PYTHON_VERSION = 3.12
TORCH_VERSION = 2.10.0
TORCH_VERSION_SHORT = $(shell echo $(TORCH_VERSION) | tr -d .)
INSTALL_CHANNEL = whl
DION_SHA = "7452a5823cf9655b93c3f1d8020b4ebb2535239b"
GROUPED_GEMM_SHA = "f1429a3c44c98f7912aa4b00125144cdf4e7fdb2"
# Compute capabilities the from-source CUDA extensions (grouped-gemm, transformer-engine, ...) are
# built for. The CUDA-13 targets (beaker-image-cu130*) extend these with sm_103 (B300 / Blackwell Ultra).
TORCH_CUDA_ARCH_LIST = 9.0 10.0
FLASH_ATTN_VERSION = 2.8.2
# Archs flash-attn 2 is compiled for (it reads FLASH_ATTN_CUDA_ARCHS, not TORCH_CUDA_ARCH_LIST).
FLASH_ATTN_CUDA_ARCHS = 90;100
FLASH_ATTN_3_SHA = "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
FA3_MAX_JOBS = 64
# flash-attn 4 provides the `flash_attn.cute` backend (AttentionBackendName.flash_4). Optional:
# leave FLASH_ATTN_4_VERSION empty to skip it. When set, it's installed (as a prebuilt wheel) and
# the image tag gets a '-fa4' suffix (see FA4_TAG). Use FLASH_ATTN_4_EXTRAS='[cu13]' for CUDA 13.
# FA4 is CUDA-13 only; the beaker-image-cu130-fa4* targets enable it (see FA4_ARGS).
FLASH_ATTN_4_VERSION =
FLASH_ATTN_4_EXTRAS =
# Pin for nvidia-cutlass-dsl (FA4's DSL dep). FA4 betas pin it loosely, so pip can float to a
# version whose API dropped symbols the wheel needs (4.6.x removed ThrMma, breaking 4.0.0b16).
FLASH_ATTN_4_CUTLASS_DSL_VERSION =
TE_VERSION = 2.9
RING_FLASH_ATTN_VERSION = 0.1.8
LIGER_KERNEL_VERSION = 0.6.4
# NOTE: Quack currently requires CUDA 12.9 or higher and PyTorch 2.9.1
# QUACK_VERSION = 0.2.4
QUACK_VERSION = ""
# B300 (sm_103) image switches. Empty for the default image; set by the CUDA-13 targets (via
# CUDA13_ARGS) to bake in the B300-only fixes (see CUDA13_ARGS and the Dockerfile release stage).
B300 =
TRITON_PTXAS_PATH =
# symm-mem / RMA image switches. Empty for the default image; set by the '-rma' targets so the
# symm_mem_vdev2d (rowwise EP) and nccl_rma_p2p (custom PP) extensions can JIT-build at runtime.
UBUNTU_VERSION = 22.04
# Release-stage base. Default plain Ubuntu; the RMA targets override with a CUDA 'devel' base so the
# runtime ships nvcc + CUDA headers those extensions need to compile.
BASE_IMAGE = ubuntu:$(UBUNTU_VERSION)
# NCCL exposing the RMA one-sided window signal API (ncclPutSignal / ncclWaitSignal) for nccl_rma_p2p.
NCCL_RMA_VERSION = 2.29.7
# NCCL / NVSHMEM install specs passed to the Dockerfile. Empty = torch's bundled NCCL, no NVSHMEM.
NCCL_PIP_SPEC =
NVSHMEM_PIP_SPEC =

#--------------#
# Build naming #
#--------------#

VERSION = $(shell python src/olmo_core/version.py)
VERSION_SHORT = $(shell python src/olmo_core/version.py short)
IMAGE_SUFFIX = $(shell date "+%Y-%m-%d")
# Build-variant marker ('-sm80', ...) set by the specialized image targets; combined with the FA4/RMA
# matrix suffixes below.
IMAGE_VARIANT =
# '-fa4' marker in the tag when flash-attn 4 is baked in (empty otherwise; see FLASH_ATTN_4_VERSION).
FA4_TAG = $(if $(FLASH_ATTN_4_VERSION),-fa4,)
# '-rma' marker when the symm-mem/RMA stack (NVSHMEM + RMA-capable NCCL on a devel base) is baked in.
RMA_TAG = $(if $(NVSHMEM_PIP_SPEC),-rma,)
# Image tag omits the GPU generation: a CUDA-13 image is built for sm_90/100/103, so one image
# serves H100 + B200 + B300 (naming stays consistent with the older CUDA-12 images).
IMAGE_TAG = tch$(TORCH_VERSION_SHORT)$(CUDA_VERSION_PATH)$(IMAGE_VARIANT)$(FA4_TAG)$(RMA_TAG)-$(IMAGE_SUFFIX)

# Imports the built image is smoke-tested against. The CUDA-13 targets override this to drop
# flash_attn_3, which isn't built on CUDA 13 (see CUDA13_ARGS).
DOCKER_VALIDATE_IMPORTS = import torch; import transformer_engine.pytorch; import flash_attn; import flash_attn_3.flash_attn_interface

.PHONY : docker-image
docker-image :
	docker build -f src/Dockerfile \
		--platform=linux/amd64 \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg CUDA_VERSION_PATH=$(CUDA_VERSION_PATH) \
		--build-arg CUDA_NVCC_VERSION=$(CUDA_NVCC_VERSION) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		--build-arg UBUNTU_VERSION=$(UBUNTU_VERSION) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg NCCL_PIP_SPEC="$(NCCL_PIP_SPEC)" \
		--build-arg NVSHMEM_PIP_SPEC="$(NVSHMEM_PIP_SPEC)" \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		--build-arg TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
		--build-arg INSTALL_CHANNEL=$(INSTALL_CHANNEL) \
		--build-arg DION_SHA=$(DION_SHA) \
		--build-arg GROUPED_GEMM_SHA=$(GROUPED_GEMM_SHA) \
		--build-arg FLASH_ATTN_VERSION=$(FLASH_ATTN_VERSION) \
		--build-arg FLASH_ATTN_CUDA_ARCHS="$(FLASH_ATTN_CUDA_ARCHS)" \
		--build-arg FLASH_ATTN_3_SHA=$(FLASH_ATTN_3_SHA) \
		--build-arg FA3_MAX_JOBS=$(FA3_MAX_JOBS) \
		--build-arg FLASH_ATTN_4_VERSION=$(FLASH_ATTN_4_VERSION) \
		--build-arg FLASH_ATTN_4_EXTRAS="$(FLASH_ATTN_4_EXTRAS)" \
		--build-arg FLASH_ATTN_4_CUTLASS_DSL_VERSION=$(FLASH_ATTN_4_CUTLASS_DSL_VERSION) \
		--build-arg TE_VERSION=$(TE_VERSION) \
		--build-arg RING_FLASH_ATTN_VERSION=$(RING_FLASH_ATTN_VERSION) \
		--build-arg LIGER_KERNEL_VERSION=$(LIGER_KERNEL_VERSION) \
		--build-arg QUACK_VERSION=$(QUACK_VERSION) \
		--build-arg B300="$(B300)" \
		--build-arg TRITON_PTXAS_PATH="$(TRITON_PTXAS_PATH)" \
		--target release \
		-t olmo-core:$(IMAGE_TAG) .
	@docker run --rm olmo-core:$(IMAGE_TAG) python -c '$(DOCKER_VALIDATE_IMPORTS)'
	@echo "✓ Image validated. Python environment:"
	@echo ""
	@docker run --rm olmo-core:$(IMAGE_TAG) pip list
	@echo ""
	@echo "✓ Build complete: olmo-core:$(IMAGE_TAG) (size=$$(docker inspect -f '{{ .Size }}' olmo-core:$(IMAGE_TAG) | numfmt --to=si))"
	@echo ""

.PHONY : docker-image-sm80
docker-image-sm80 :
	$(MAKE) docker-image \
		TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0" \
		FLASH_ATTN_CUDA_ARCHS="80;90;100" \
		IMAGE_VARIANT=-sm80

.PHONY : ghcr-image
ghcr-image : docker-image
	docker tag olmo-core:$(IMAGE_TAG) ghcr.io/allenai/olmo-core:$(IMAGE_TAG)
	docker push ghcr.io/allenai/olmo-core:$(IMAGE_TAG)
	docker tag olmo-core:$(IMAGE_TAG) ghcr.io/allenai/olmo-core:latest
	docker push ghcr.io/allenai/olmo-core:latest

BEAKER_WORKSPACE = ai2/OLMo-core
BEAKER_USER = $(shell beaker account whoami --format=json | jq -r '.[0].name')

.PHONY : beaker-image
beaker-image : docker-image
	@./src/scripts/beaker/create_beaker_image.sh olmo-core:$(IMAGE_TAG) olmo-core-$(IMAGE_TAG) $(BEAKER_WORKSPACE)
	@echo "✓ Done"

####################################################################################################
# Image build matrix
#
# Two CUDA families, each optionally layered with flash-attn 4 (FA4) and/or the symm-mem/RMA stack:
#   * CUDA 12.8 (torch 2.10) — built for sm_90/100 (H100, B200).
#   * CUDA 13.0 (torch 2.11) — built for sm_90/100/103, so ONE image serves H100 + B200 + B300.
#     The tag carries no GPU generation (naming stays consistent with the older CUDA-12 images).
#
# FA4 (`flash_attn.cute`) is CUDA-13 only: the `flash-attn-4` package ships a `cu13` extra and no
# `cu12`, so it is offered on the CUDA-13 targets only. RMA works on both families (the NVSHMEM /
# RMA-NCCL wheels exist for cu12 and cu13).
#
# Build args are layered onto the parametrized `beaker-image` target; last assignment of a variable
# wins, so an FA4 layer's DOCKER_VALIDATE_IMPORTS overrides the base one.
####################################################################################################

# CUDA-13 base (torch 2.11, validated on B300). Adds sm_103 to the arch lists, registers torch's
# bundled nvrtc so transformer-engine imports on CUDA 13, points Triton at a CUDA-13 ptxas, and skips
# the flash-attn 3 build (FA3 has no sm_103 kernels and doesn't build on CUDA 13 — dropped from the
# smoke test too). If flash-attn / transformer-engine / grouped-gemm fail to build, bump them to
# CUDA-13-compatible releases.
CUDA13_ARGS = \
	TORCH_VERSION=2.11.0 \
	CUDA_VERSION=13.0.1 \
	TORCH_CUDA_ARCH_LIST="9.0 10.0 10.3" \
	FLASH_ATTN_CUDA_ARCHS="90;100;103" \
	B300=1 \
	TRITON_PTXAS_PATH=/usr/local/bin/triton-ptxas \
	DOCKER_VALIDATE_IMPORTS="import torch; import transformer_engine.pytorch; import flash_attn"

# FA4 layer (CUDA-13 only): installs the flash_attn.cute wheel (AttentionBackendName.flash_4) and
# appends it to the smoke test; adds the '-fa4' tag suffix (see FA4_TAG). The cutlass-dsl pin avoids
# 4.6.x, which dropped symbols 4.0.0b16 needs (ThrMma).
FA4_ARGS = \
	FLASH_ATTN_4_VERSION=4.0.0b16 \
	FLASH_ATTN_4_EXTRAS="[cu13]" \
	FLASH_ATTN_4_CUTLASS_DSL_VERSION=4.5.3 \
	DOCKER_VALIDATE_IMPORTS="import torch; import transformer_engine.pytorch; import flash_attn; import flash_attn.cute"

# symm-mem / RMA layer: a CUDA 'devel' release base (ships nvcc + headers) plus NVSHMEM and an
# RMA-capable NCCL, so the symm_mem_vdev2d (rowwise EP) and nccl_rma_p2p (custom PP) extensions can
# JIT-build at runtime. Required for EP>1 / rowwise runs; EP=1 runs don't need it. Adds the '-rma'
# tag suffix (see RMA_TAG). One bundle per CUDA family — the wheels are CUDA-major-specific.
RMA_CU12_ARGS = \
	BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu$(UBUNTU_VERSION) \
	NVSHMEM_PIP_SPEC=nvidia-nvshmem-cu12 \
	NCCL_PIP_SPEC=nvidia-nccl-cu12==$(NCCL_RMA_VERSION)
RMA_CU13_ARGS = \
	BASE_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu$(UBUNTU_VERSION) \
	NVSHMEM_PIP_SPEC=nvidia-nvshmem-cu13 \
	NCCL_PIP_SPEC=nvidia-nccl-cu13==$(NCCL_RMA_VERSION)

# ---- CUDA 12.8 family (H100, B200) — torch 2.10 -------------------------------------------------
# olmo-core-tch2100cu128-<date>
.PHONY : beaker-image-cu128
beaker-image-cu128 :
	$(MAKE) beaker-image

# olmo-core-tch2100cu128-rma-<date>
.PHONY : beaker-image-cu128-rma
beaker-image-cu128-rma :
	$(MAKE) beaker-image $(RMA_CU12_ARGS)

# ---- CUDA 13.0 family (H100, B200, B300) — torch 2.11 ------------------------------------------
# olmo-core-tch2110cu130-<date>
.PHONY : beaker-image-cu130
beaker-image-cu130 :
	$(MAKE) beaker-image $(CUDA13_ARGS)

# olmo-core-tch2110cu130-fa4-<date>
.PHONY : beaker-image-cu130-fa4
beaker-image-cu130-fa4 :
	$(MAKE) beaker-image $(CUDA13_ARGS) $(FA4_ARGS)

# olmo-core-tch2110cu130-rma-<date>
.PHONY : beaker-image-cu130-rma
beaker-image-cu130-rma :
	$(MAKE) beaker-image $(CUDA13_ARGS) $(RMA_CU13_ARGS)

# olmo-core-tch2110cu130-fa4-rma-<date>  (flash_4 attention + symm-mem/RMA rowwise EP)
.PHONY : beaker-image-cu130-fa4-rma
beaker-image-cu130-fa4-rma :
	$(MAKE) beaker-image $(CUDA13_ARGS) $(FA4_ARGS) $(RMA_CU13_ARGS)

# ---- sm_80 (A100) variant ----------------------------------------------------------------------
.PHONY : beaker-image-sm80
beaker-image-sm80 :
	$(MAKE) beaker-image \
		TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0" \
		FLASH_ATTN_CUDA_ARCHS="80;90;100" \
		IMAGE_VARIANT=-sm80

.PHONY : get-beaker-workspace
get-beaker-workspace :
	@echo $(BEAKER_WORKSPACE)

.PHONY : get-full-beaker-image-name
get-full-beaker-image-name :
	@./src/scripts/beaker/get_full_image_name.sh $(IMAGE_TAG) $(BEAKER_WORKSPACE)
