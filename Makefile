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
UBUNTU_VERSION = 22.04
# Release-stage base image. Defaults to a plain Ubuntu image (matching the Dockerfile default). The
# RMA image targets override this with the CUDA 'devel' base so the release image ships nvcc + CUDA
# headers, which the nccl_rma_p2p LazyCudaExtension needs to JIT-compile at runtime.
BASE_IMAGE = ubuntu:$(UBUNTU_VERSION)
# NCCL override spec passed to the Dockerfile (see NCCL_PIP_SPEC there). Empty means "use torch's
# bundled NCCL". The RMA image targets set this to an RMA-capable NCCL wheel.
NCCL_PIP_SPEC =
# NCCL version exposing the RMA one-sided window signal API (ncclPutSignal / ncclWaitSignal) that
# nccl_rma_p2p requires. Used by the RMA image targets below.
NCCL_RMA_VERSION = 2.29.7
# NVSHMEM install spec passed to the Dockerfile (see NVSHMEM_PIP_SPEC there). Empty means "don't
# install NVSHMEM". The RMA image targets set this so the symm_mem_vdev2d ext can be built at runtime.
NVSHMEM_PIP_SPEC =
GROUPED_GEMM_SHA = "f1429a3c44c98f7912aa4b00125144cdf4e7fdb2"
FLASH_ATTN_VERSION = 2.8.2
FLASH_ATTN_3_SHA = "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
FA3_MAX_JOBS = 64
TE_VERSION = 2.9
RING_FLASH_ATTN_VERSION = 0.1.8
LIGER_KERNEL_VERSION = 0.6.4
# NOTE: Quack currently requires CUDA 12.9 or higher and PyTorch 2.9.1
# QUACK_VERSION = 0.2.4
QUACK_VERSION = ""

#--------------#
# Build naming #
#--------------#

VERSION = $(shell python src/olmo_core/version.py)
VERSION_SHORT = $(shell python src/olmo_core/version.py short)
IMAGE_SUFFIX = $(shell date "+%Y-%m-%d")
IMAGE_TAG = tch$(TORCH_VERSION_SHORT)$(CUDA_VERSION_PATH)-$(IMAGE_SUFFIX)

.PHONY : docker-image
docker-image :
	docker build -f src/Dockerfile \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg CUDA_VERSION_PATH=$(CUDA_VERSION_PATH) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		--build-arg INSTALL_CHANNEL=$(INSTALL_CHANNEL) \
		--build-arg NCCL_PIP_SPEC=$(NCCL_PIP_SPEC) \
		--build-arg NVSHMEM_PIP_SPEC=$(NVSHMEM_PIP_SPEC) \
		--build-arg GROUPED_GEMM_SHA=$(GROUPED_GEMM_SHA) \
		--build-arg FLASH_ATTN_VERSION=$(FLASH_ATTN_VERSION) \
		--build-arg FLASH_ATTN_3_SHA=$(FLASH_ATTN_3_SHA) \
		--build-arg FA3_MAX_JOBS=$(FA3_MAX_JOBS) \
		--build-arg TE_VERSION=$(TE_VERSION) \
		--build-arg RING_FLASH_ATTN_VERSION=$(RING_FLASH_ATTN_VERSION) \
		--build-arg LIGER_KERNEL_VERSION=$(LIGER_KERNEL_VERSION) \
		--build-arg QUACK_VERSION=$(QUACK_VERSION) \
		--target release \
		-t olmo-core:$(IMAGE_TAG) .
	@docker run --rm olmo-core:$(IMAGE_TAG) python -c \
		'import torch; import transformer_engine.pytorch; import flash_attn; import flash_attn_3.flash_attn_interface'
	@if [ -n "$(NVSHMEM_PIP_SPEC)" ]; then \
		echo ""; echo "── NVSHMEM diagnostic (symm_mem_vdev2d build prereqs) ──"; \
		docker run --rm olmo-core:$(IMAGE_TAG) bash -c '\
			hdr=$$(find /opt/conda -path "*nvidia/nvshmem/include/nvshmem.h" 2>/dev/null | head -1); \
			dev=$$(find /opt/conda -path "*nvidia/nvshmem*/libnvshmem_device.a" 2>/dev/null | head -1); \
			host=$$(find /opt/conda -path "*nvidia/nvshmem*/libnvshmem_host.so*" 2>/dev/null | head -1); \
			echo "nvshmem.h            : $${hdr:-MISSING}"; \
			echo "libnvshmem_device.a  : $${dev:-MISSING}"; \
			echo "libnvshmem_host.so.* : $${host:-MISSING}"; \
			if [ -n "$$hdr" ] && [ -n "$$dev" ] && [ -n "$$host" ]; then \
				echo "OK: NVSHMEM build prereqs present (pip wheel is sufficient for symm_mem_vdev2d)."; \
			else \
				echo "WARNING: NVSHMEM incomplete (see MISSING above). The symm_mem_vdev2d build will fail;"; \
				echo "         install the full NVSHMEM SDK and set NVSHMEM_HOME instead of the pip wheel."; \
			fi'; \
		echo ""; \
	fi
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

# Build + register beaker images with reliable support for the MoE-v2 GPU comm kernels — the custom
# pipeline-parallel transport (nccl_rma_p2p) and the rowwise-EP symmetric-memory backend
# (symm_mem_vdev2d). Relative to the default image these make three changes:
#   1. Release base = CUDA 'devel' image, so the runtime ships nvcc + CUDA headers. Both extensions
#      compile at runtime (nccl_rma_p2p JIT-builds on first use; symm_mem_vdev2d is built via
#      `python -m olmo_core.kernels.build_symm_mem_vdev2d_ext`), and the default plain-Ubuntu release
#      image has no nvcc, so those builds would fail.
#   2. NCCL is pinned to $(NCCL_RMA_VERSION), which exposes the one-sided window signal API the
#      transport compiles against. '_find_nccl_paths' then discovers it under conda site-packages.
#   3. NVSHMEM is installed so symm_mem_vdev2d can link it; '_find_nvshmem_paths' discovers it under
#      conda site-packages. (The package source is cloned at launch, so the ext itself can't be
#      prebuilt into the image — the image only carries the build prerequisites.)
# Both use torch 2.10 (the default TORCH_VERSION); they differ only in CUDA version. Images are
# tagged with an 'rma' suffix (e.g. olmo-core-tch2100cu128-rma-<date>) to distinguish them.

.PHONY : beaker-image-rma-cu128
beaker-image-rma-cu128 :
	$(MAKE) beaker-image \
		CUDA_VERSION=12.8.1 \
		BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu$(UBUNTU_VERSION) \
		NCCL_PIP_SPEC=nvidia-nccl-cu12==$(NCCL_RMA_VERSION) \
		NVSHMEM_PIP_SPEC=nvidia-nvshmem-cu12 \
		IMAGE_SUFFIX=rma-$(IMAGE_SUFFIX)

.PHONY : beaker-image-rma-cu13
beaker-image-rma-cu13 :
	$(MAKE) beaker-image \
		CUDA_VERSION=13.0.1 \
		BASE_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu$(UBUNTU_VERSION) \
		NCCL_PIP_SPEC=nvidia-nccl-cu13==$(NCCL_RMA_VERSION) \
		NVSHMEM_PIP_SPEC=nvidia-nvshmem-cu13 \
		IMAGE_SUFFIX=rma-$(IMAGE_SUFFIX)

.PHONY : get-beaker-workspace
get-beaker-workspace :
	@echo $(BEAKER_WORKSPACE)

.PHONY : get-full-beaker-image-name
get-full-beaker-image-name :
	@./src/scripts/beaker/get_full_image_name.sh $(IMAGE_TAG) $(BEAKER_WORKSPACE)
