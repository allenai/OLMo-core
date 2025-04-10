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
#  * The versions installed in '.github/actions/setup-venv/action.yml' match if necessary.
CUDA_VERSION = 12.6.3
CUDA_PATH=cu$(shell echo $(CUDA_VERSION) | cut -d"." -f1-2 | tr -d .)
PYTHON_VERSION = 3.11
TORCH_VERSION = 2.6.0
TORCH_VERSION_SHORT = $(shell echo $(TORCH_VERSION) | tr -d .)
INSTALL_CHANNEL = whl/test
GROUPED_GEMM_VERSION = "grouped_gemm @ git+https://git@github.com/tgale96/grouped_gemm.git@main"
FLASH_ATTN_VERSION = 2.7.4.post1
RING_FLASH_ATTN_VERSION = 0.1.4
LIGER_KERNEL_VERSION = 0.5.4

#--------------#
# Build naming #
#--------------#

VERSION = $(shell python src/olmo_core/version.py)
VERSION_SHORT = $(shell python src/olmo_core/version.py short)
IMAGE_SUFFIX = ""
IMAGE_TAG = tch$(TORCH_VERSION_SHORT)$(CUDA_PATH)$(IMAGE_SUFFIX)

.PHONY : docker-image
docker-image :
	docker build -f src/Dockerfile \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg CUDA_PATH=$(CUDA_PATH) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		--build-arg INSTALL_CHANNEL=$(INSTALL_CHANNEL) \
		--build-arg GROUPED_GEMM_VERSION=$(GROUPED_GEMM_VERSION) \
		--build-arg FLASH_ATTN_VERSION=$(FLASH_ATTN_VERSION) \
		--build-arg RING_FLASH_ATTN_VERSION=$(RING_FLASH_ATTN_VERSION) \
		--build-arg LIGER_KERNEL_VERSION=$(LIGER_KERNEL_VERSION) \
		--target release \
		--progress plain \
		-t olmo-core:$(IMAGE_TAG) .
	echo "Built image 'olmo-core:$(IMAGE_TAG)', size: $$(docker inspect -f '{{ .Size }}' olmo-core:$(IMAGE_TAG) | numfmt --to=si)"

.PHONY : ghcr-image
ghcr-image : docker-image
	docker tag olmo-core:$(IMAGE_TAG) ghcr.io/allenai/olmo-core:$(IMAGE_TAG)
	docker push ghcr.io/allenai/olmo-core:$(IMAGE_TAG)
	docker tag olmo-core:$(IMAGE_TAG) ghcr.io/allenai/olmo-core:$(IMAGE_TAG)-v$(VERSION_SHORT)
	docker push ghcr.io/allenai/olmo-core:$(IMAGE_TAG)-v$(VERSION_SHORT)
	docker tag olmo-core:$(IMAGE_TAG) ghcr.io/allenai/olmo-core:$(IMAGE_TAG)-v$(VERSION)
	docker push ghcr.io/allenai/olmo-core:$(IMAGE_TAG)-v$(VERSION)
	docker tag olmo-core:$(IMAGE_TAG) ghcr.io/allenai/olmo-core:latest
	docker push ghcr.io/allenai/olmo-core:latest

BEAKER_WORKSPACE = ai2/OLMo-core
BEAKER_USER = $(shell beaker account whoami --format=json | jq -r '.[0].name')

.PHONY : beaker-image
beaker-image : docker-image
	./src/scripts/beaker/create_beaker_image.sh olmo-core:$(IMAGE_TAG) olmo-core-$(IMAGE_TAG) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh olmo-core:$(IMAGE_TAG) olmo-core-$(IMAGE_TAG)-v$(VERSION_SHORT) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh olmo-core:$(IMAGE_TAG) olmo-core-$(IMAGE_TAG)-v$(VERSION) $(BEAKER_WORKSPACE)

.PHONY : get-beaker-workspace
get-beaker-workspace :
	@echo $(BEAKER_WORKSPACE)

.PHONY : get-full-beaker-image-name
get-full-beaker-image-name :
	@./src/scripts/beaker/get_full_image_name.sh $(IMAGE_NAME) $(BEAKER_WORKSPACE)
