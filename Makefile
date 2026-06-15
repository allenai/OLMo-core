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
TORCH_VERSION = 2.11.0
TORCH_VERSION_SHORT = $(shell echo $(TORCH_VERSION) | tr -d .)
TORCHVISION_VERSION = 0.26.0
TORCHAUDIO_VERSION = 2.11.0
TORCH_CUDA_ARCH_LIST = 8.0 9.0 10.0
NVIDIA_CUBLAS_VERSION =
INSTALL_CHANNEL = whl
GROUPED_GEMM_SHA = "f1429a3c44c98f7912aa4b00125144cdf4e7fdb2"
FLASH_ATTN_VERSION = 2.8.2
FLASH_ATTN_CUDA_ARCHS = 90;100
FLASH_ATTN_3_SHA = "92ca9da8d66f7b34ff50dc080ec0fef9661260d6"
FA3_MAX_JOBS = 64
TE_VERSION = 2.14
RING_FLASH_ATTN_VERSION = 0.1.8
LIGER_KERNEL_VERSION = 0.6.4
FLASH_ATTN_4_VERSION = 4.0.0b12
FLASH_ATTN_4_EXTRAS =
# NOTE: Quack currently requires CUDA 12.9 or higher and PyTorch 2.9.1
# QUACK_VERSION = 0.2.4
QUACK_VERSION = ""
INSTALL_DOCA_RDMA = 0
DOCA_VERSION = 3.3.0
DOCA_RDMA_CORE_VERSION = 2601.0.7-1
DOCA_PERFTEST_VERSION = 26.01.5-1

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
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		--build-arg TORCHVISION_VERSION="$(TORCHVISION_VERSION)" \
		--build-arg TORCHAUDIO_VERSION="$(TORCHAUDIO_VERSION)" \
		--build-arg TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
		--build-arg NVIDIA_CUBLAS_VERSION="$(NVIDIA_CUBLAS_VERSION)" \
		--build-arg INSTALL_CHANNEL=$(INSTALL_CHANNEL) \
		--build-arg GROUPED_GEMM_SHA=$(GROUPED_GEMM_SHA) \
		--build-arg FLASH_ATTN_VERSION=$(FLASH_ATTN_VERSION) \
		--build-arg FLASH_ATTN_CUDA_ARCHS="$(FLASH_ATTN_CUDA_ARCHS)" \
		--build-arg FLASH_ATTN_3_SHA=$(FLASH_ATTN_3_SHA) \
		--build-arg FA3_MAX_JOBS=$(FA3_MAX_JOBS) \
		--build-arg FLASH_ATTN_4_VERSION=$(FLASH_ATTN_4_VERSION) \
		--build-arg FLASH_ATTN_4_EXTRAS="$(FLASH_ATTN_4_EXTRAS)" \
		--build-arg TE_VERSION=$(TE_VERSION) \
		--build-arg RING_FLASH_ATTN_VERSION=$(RING_FLASH_ATTN_VERSION) \
		--build-arg LIGER_KERNEL_VERSION=$(LIGER_KERNEL_VERSION) \
		--build-arg QUACK_VERSION=$(QUACK_VERSION) \
		--build-arg INSTALL_DOCA_RDMA=$(INSTALL_DOCA_RDMA) \
		--build-arg DOCA_VERSION=$(DOCA_VERSION) \
		--build-arg DOCA_RDMA_CORE_VERSION=$(DOCA_RDMA_CORE_VERSION) \
		--build-arg DOCA_PERFTEST_VERSION=$(DOCA_PERFTEST_VERSION) \
		--target release \
		-t olmo-core:$(IMAGE_TAG) .
	@docker run --rm olmo-core:$(IMAGE_TAG) python -c \
		'import torch; import transformer_engine.pytorch; import flash_attn; import flash_attn.cute; import flash_attn_3.flash_attn_interface'
	@echo "✓ Image validated. Python environment:"
	@echo ""
	@docker run --rm olmo-core:$(IMAGE_TAG) pip list
	@echo ""
	@echo "✓ Build complete: olmo-core:$(IMAGE_TAG) (size=$$(docker inspect -f '{{ .Size }}' olmo-core:$(IMAGE_TAG) | numfmt --to=si))"
	@echo ""

.PHONY : b300-image
b300-image :
	$(MAKE) docker-image \
		CUDA_VERSION=13.0.2 \
		CUDA_VERSION_PATH=cu130 \
		TORCH_VERSION=2.12.0 \
		TORCHVISION_VERSION=0.27.0 \
		TORCHAUDIO_VERSION= \
		TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0 10.3+PTX" \
		NVIDIA_CUBLAS_VERSION=13.2.0.9 \
		FLASH_ATTN_CUDA_ARCHS="90;100;103" \
		FLASH_ATTN_4_VERSION=4.0.0b16 \
		FLASH_ATTN_4_EXTRAS="[cu13]" \
		TE_VERSION=2.16.0 \
		LIGER_KERNEL_VERSION=0.8.0 \
		INSTALL_DOCA_RDMA=1

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

.PHONY : get-beaker-workspace
get-beaker-workspace :
	@echo $(BEAKER_WORKSPACE)

.PHONY : get-full-beaker-image-name
get-full-beaker-image-name :
	@./src/scripts/beaker/get_full_image_name.sh $(IMAGE_TAG) $(BEAKER_WORKSPACE)
