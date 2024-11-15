CUDA_VERSION = "12.1"
TORCH_CUDA_VERSION = $(shell echo $(CUDA_VERSION) | tr -d .)
TORCH_VERSION = "2.5.1"
# NOTE: when upgrading the nightly version you also need to upgrade the torch version specification
# in 'pyproject.toml' to include that nightly version.
TORCH_NIGHTLY_VERSION = "2.6.0.dev20241009"
TORCHAO_VERSION = "0.5.0"
MEGABLOCKS_VERSION = "megablocks[gg] @ git+https://git@github.com/epwalsh/megablocks.git@epwalsh/deps"
FLASH_ATTN_WHEEL = https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

VERSION = $(shell python src/olmo_core/version.py)
VERSION_SHORT = $(shell python src/olmo_core/version.py short)
IMAGE_BASENAME = olmo-core
BEAKER_WORKSPACE = ai2/OLMo-core
BEAKER_USER = $(shell beaker account whoami --format=json | jq -r '.[0].name')

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

.PHONY : stable-image
stable-image :
	docker build -f src/Dockerfile \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg TORCH_CUDA_VERSION=$(TORCH_CUDA_VERSION) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		--build-arg FLASH_ATTN_WHEEL=$(FLASH_ATTN_WHEEL) \
		--build-arg MEGABLOCKS_VERSION=$(MEGABLOCKS_VERSION) \
		--build-arg TORCHAO_VERSION=$(TORCHAO_VERSION) \
		--target stable \
		--progress plain \
		-t $(IMAGE_BASENAME) .
	echo "Built image '$(IMAGE_BASENAME)', size: $$(docker inspect -f '{{ .Size }}' $(IMAGE_BASENAME) | numfmt --to=si)"

.PHONY : nightly-image
nightly-image :
	docker build -f src/Dockerfile \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg TORCH_CUDA_VERSION=$(TORCH_CUDA_VERSION) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		--build-arg FLASH_ATTN_WHEEL=$(FLASH_ATTN_WHEEL) \
		--build-arg MEGABLOCKS_VERSION=$(MEGABLOCKS_VERSION) \
		--build-arg TORCHAO_VERSION=$(TORCHAO_VERSION) \
		--build-arg TORCH_NIGHTLY_VERSION=$(TORCH_NIGHTLY_VERSION) \
		--target nightly \
		--progress plain \
		-t $(IMAGE_BASENAME)-nightly .
	echo "Built image '$(IMAGE_BASENAME)-nightly', size: $$(docker inspect -f '{{ .Size }}' $(IMAGE_BASENAME)-nightly | numfmt --to=si)"

.PHONY : beaker-image-stable
beaker-image-stable : stable-image
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME)-v$(VERSION_SHORT) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME)-v$(VERSION) $(BEAKER_WORKSPACE)

.PHONY : beaker-image-nightly
beaker-image-nightly : nightly-image
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME)-nightly $(IMAGE_BASENAME)-nightly $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME)-nightly $(IMAGE_BASENAME)-v$(VERSION_SHORT)-nightly $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME)-nightly $(IMAGE_BASENAME)-v$(VERSION)-nightly $(BEAKER_WORKSPACE)

.PHONY : get-beaker-workspace
get-beaker-workspace :
	@echo $(BEAKER_WORKSPACE)

.PHONY : get-full-beaker-image-name
get-full-beaker-image-name :
	@./src/scripts/beaker/get_full_image_name.sh $(IMAGE_NAME) $(BEAKER_WORKSPACE)
