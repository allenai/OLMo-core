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
	rm -rf *.egg-info/
	python -m build

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
PYTHON_VERSION = 3.12
TORCH_VERSION = 2.10.0
TORCH_VERSION_SHORT = $(shell echo $(TORCH_VERSION) | tr -d .)
INSTALL_CHANNEL = whl
# Compute capabilities the from-source CUDA extensions (grouped-gemm, transformer-engine, ...) are
# built for. The 'beaker-image-b300' target extends these with sm_103 (B300 / Blackwell Ultra).
TORCH_CUDA_ARCH_LIST = 9.0 10.0
GROUPED_GEMM_SHA = "f1429a3c44c98f7912aa4b00125144cdf4e7fdb2"
FLASH_ATTN_VERSION = 2.8.2
# Archs flash-attn 2 is compiled for (it reads FLASH_ATTN_CUDA_ARCHS, not TORCH_CUDA_ARCH_LIST).
FLASH_ATTN_CUDA_ARCHS = 90;100
FLASH_ATTN_3_SHA = "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
FA3_MAX_JOBS = 64
# flash-attn 4 provides the `flash_attn.cute` backend (AttentionBackendName.flash_4). Optional:
# leave FLASH_ATTN_4_VERSION empty to skip it. When set, it's installed (as a prebuilt wheel) and
# the image tag gets a '-fa4' suffix (see FA4_TAG). Use FLASH_ATTN_4_EXTRAS='[cu13]' for CUDA 13.
# The 'beaker-image-b300-fa4' target enables it with the CUDA-13 wheel.
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
# B300 (sm_103) image switches. Empty for the default image; set by the 'beaker-image-b300' target
# to bake in the B300-only fixes (see that target and the Dockerfile release stage).
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
# '-fa4' marker in the tag when flash-attn 4 is baked in (empty otherwise; see FLASH_ATTN_4_VERSION).
FA4_TAG = $(if $(FLASH_ATTN_4_VERSION),-fa4,)
IMAGE_TAG = tch$(TORCH_VERSION_SHORT)$(CUDA_VERSION_PATH)$(FA4_TAG)-$(IMAGE_SUFFIX)

# Imports the built image is smoke-tested against. The B300 targets override this to drop
# flash_attn_3, which isn't built for B300 (see the 'beaker-image-b300' target).
DOCKER_VALIDATE_IMPORTS = import torch; import transformer_engine.pytorch; import flash_attn; import flash_attn_3.flash_attn_interface

.PHONY : docker-image
docker-image :
	docker build -f src/Dockerfile \
		--platform=linux/amd64 \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg CUDA_VERSION_PATH=$(CUDA_VERSION_PATH) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		--build-arg UBUNTU_VERSION=$(UBUNTU_VERSION) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg NCCL_PIP_SPEC="$(NCCL_PIP_SPEC)" \
		--build-arg NVSHMEM_PIP_SPEC="$(NVSHMEM_PIP_SPEC)" \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		--build-arg TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
		--build-arg INSTALL_CHANNEL=$(INSTALL_CHANNEL) \
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

# Build + register a B300 (Blackwell Ultra, sm_103) image, on CUDA 13.0. Both torch 2.10 and 2.11
# pin Triton 3.6 (which supports sm_103); the difference is that 2.11 is the config validated on B300
# hardware while 2.10 tracks the default image's torch. Build whichever you need separately.
# Shared B300 build args:
#  - "10.3"/"103" add sm_103 to the arch lists so the from-source extensions target B300.
#  - B300=1 makes the Dockerfile register torch's bundled nvrtc with ldconfig (so transformer-engine
#    imports on CUDA 13), symlink a CUDA-13 ptxas to /usr/local/bin/triton-ptxas, and skip the
#    flash-attn 3 build (FA3 has no sm_103 kernels and its bundled CUTLASS doesn't build on CUDA 13).
#  - TRITON_PTXAS_PATH points Triton at that symlink (Triton's bundled ptxas doesn't know sm_103a).
#  - DOCKER_VALIDATE_IMPORTS drops flash_attn_3 from the smoke test since it isn't built for B300.
# NOTE: the from-source extensions compile against CUDA 13; if flash-attn / transformer-engine /
# grouped-gemm fail to build, bump their versions to CUDA-13-compatible releases.
B300_BUILD_ARGS = \
	CUDA_VERSION=13.0.1 \
	TORCH_CUDA_ARCH_LIST="9.0 10.0 10.3" \
	FLASH_ATTN_CUDA_ARCHS="90;100;103" \
	B300=1 \
	TRITON_PTXAS_PATH=/usr/local/bin/triton-ptxas \
	DOCKER_VALIDATE_IMPORTS="import torch; import transformer_engine.pytorch; import flash_attn"

# Optional flash-attn 4 layer for the B300 build: installs the CUDA-13 flash_attn.cute wheel
# (AttentionBackendName.flash_4) and appends flash_attn.cute to the smoke test. Layered after
# B300_BUILD_ARGS on the '-fa4' targets, so its DOCKER_VALIDATE_IMPORTS wins (last assignment).
# Setting FLASH_ATTN_4_VERSION also adds the '-fa4' tag suffix (see FA4_TAG).
B300_FA4_ARGS = \
	FLASH_ATTN_4_VERSION=4.0.0b16 \
	FLASH_ATTN_4_EXTRAS="[cu13]" \
	FLASH_ATTN_4_CUTLASS_DSL_VERSION=4.5.3 \
	DOCKER_VALIDATE_IMPORTS="import torch; import transformer_engine.pytorch; import flash_attn; import flash_attn.cute"

# torch 2.11.0 (validated on B300 hardware). Produces 'olmo-core-tch2110cu130-<date>'.
.PHONY : beaker-image-b300
beaker-image-b300 :
	$(MAKE) beaker-image TORCH_VERSION=2.11.0 $(B300_BUILD_ARGS)

# torch 2.10.0 (matches the default image's torch). Produces 'olmo-core-tch2100cu130-<date>'.
.PHONY : beaker-image-b300-torch210
beaker-image-b300-torch210 :
	$(MAKE) beaker-image TORCH_VERSION=2.10.0 $(B300_BUILD_ARGS)

# Push a B300 image to GHCR instead of Beaker. Same build args as the beaker-image-b300 targets.
# torch 2.11.0 (validated on B300 hardware). Pushes 'ghcr.io/allenai/olmo-core:tch2110cu130-<date>'.
.PHONY : ghcr-image-b300
ghcr-image-b300 :
	$(MAKE) ghcr-image TORCH_VERSION=2.11.0 $(B300_BUILD_ARGS)

# B300 image with flash-attn 4 (flash_attn.cute) baked in. Same as beaker-image-b300 plus the
# optional FA4 layer; produces 'olmo-core-tch2110cu130-fa4-<date>'.
.PHONY : beaker-image-b300-fa4
beaker-image-b300-fa4 :
	$(MAKE) beaker-image TORCH_VERSION=2.11.0 $(B300_BUILD_ARGS) $(B300_FA4_ARGS)

# GHCR variant of the FA4 B300 image. Pushes 'ghcr.io/allenai/olmo-core:tch2110cu130-fa4-<date>'.
.PHONY : ghcr-image-b300-fa4
ghcr-image-b300-fa4 :
	$(MAKE) ghcr-image TORCH_VERSION=2.11.0 $(B300_BUILD_ARGS) $(B300_FA4_ARGS)

# symm-mem / RMA layer for the B300 build: a CUDA-13 'devel' release base (ships nvcc + headers) plus
# NVSHMEM and an RMA-capable NCCL, so the symm_mem_vdev2d (rowwise EP) and nccl_rma_p2p (custom PP)
# extensions can JIT-build at runtime. Required for EP>1 / rowwise runs (e.g. OLMoE3-dev-t002);
# t001 (EP=1) does not need it. IMAGE_SUFFIX gets an 'rma-' prefix to distinguish the tag.
B300_RMA_ARGS = \
	BASE_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu$(UBUNTU_VERSION) \
	NVSHMEM_PIP_SPEC=nvidia-nvshmem-cu13 \
	NCCL_PIP_SPEC=nvidia-nccl-cu13==$(NCCL_RMA_VERSION) \
	IMAGE_SUFFIX=rma-$(IMAGE_SUFFIX)

# B300 image with symm-mem/RMA support. Produces 'olmo-core-tch2110cu130-rma-<date>'.
.PHONY : beaker-image-b300-rma
beaker-image-b300-rma :
	$(MAKE) beaker-image TORCH_VERSION=2.11.0 $(B300_BUILD_ARGS) $(B300_RMA_ARGS)

# B300 image with BOTH flash-attn 4 and symm-mem/RMA (what OLMoE3-dev-t002 needs: flash_4 attention
# + rowwise EP). Produces 'olmo-core-tch2110cu130-fa4-rma-<date>'.
.PHONY : beaker-image-b300-fa4-rma
beaker-image-b300-fa4-rma :
	$(MAKE) beaker-image TORCH_VERSION=2.11.0 $(B300_BUILD_ARGS) $(B300_FA4_ARGS) $(B300_RMA_ARGS)

# torch 2.10.0 (matches the default image's torch). Pushes 'ghcr.io/allenai/olmo-core:tch2100cu130-<date>'.
.PHONY : ghcr-image-b300-torch210
ghcr-image-b300-torch210 :
	$(MAKE) ghcr-image TORCH_VERSION=2.10.0 $(B300_BUILD_ARGS)

.PHONY : get-beaker-workspace
get-beaker-workspace :
	@echo $(BEAKER_WORKSPACE)

.PHONY : get-full-beaker-image-name
get-full-beaker-image-name :
	@./src/scripts/beaker/get_full_image_name.sh $(IMAGE_TAG) $(BEAKER_WORKSPACE)
