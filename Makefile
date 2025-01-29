CUDA_VERSION = "12.4"
TORCH_CUDA_VERSION = $(shell echo $(CUDA_VERSION) | tr -d .)
TORCH_VERSION = "2.5.1"
TORCH_VERSION_SHORT = $(shell echo $(TORCH_VERSION) | tr -d .)
# NOTE: when upgrading the nightly version you also need to upgrade the torch version specification
# in 'pyproject.toml' to include that nightly version.
TORCH_NIGHTLY_VERSION = "2.6.0.dev20241209"
TORCH_NIGHTLY_VERSION_SHORT = $(shell echo $(TORCH_NIGHTLY_VERSION) | tr -d .)
TORCHAO_VERSION = "0.6.1"
MEGABLOCKS_VERSION = "megablocks[gg] @ git+https://git@github.com/epwalsh/megablocks.git@epwalsh/deps"
FLASH_ATTN_WHEEL = https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

VERSION = $(shell python src/olmo_core/version.py)
VERSION_SHORT = $(shell python src/olmo_core/version.py short)
STABLE_IMAGE = tch$(TORCH_VERSION_SHORT)cu$(TORCH_CUDA_VERSION)
NIGHTLY_IMAGE = tch$(TORCH_NIGHTLY_VERSION_SHORT)cu$(TORCH_CUDA_VERSION)
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
		-t olmo-core:$(STABLE_IMAGE) .
	echo "Built image 'olmo-core:$(STABLE_IMAGE)', size: $$(docker inspect -f '{{ .Size }}' olmo-core:$(STABLE_IMAGE) | numfmt --to=si)"

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
		-t olmo-core:$(NIGHTLY_IMAGE) .
	echo "Built image 'olmo-core:$(NIGHTLY_IMAGE)', size: $$(docker inspect -f '{{ .Size }}' olmo-core:$(NIGHTLY_IMAGE) | numfmt --to=si)"

.PHONY : ghcr-image-stable
ghcr-image-stable : stable-image
	docker tag olmo-core:$(STABLE_IMAGE) ghcr.io/allenai/olmo-core:$(STABLE_IMAGE)
	docker push ghcr.io/allenai/olmo-core:$(STABLE_IMAGE)
	docker tag olmo-core:$(STABLE_IMAGE) ghcr.io/allenai/olmo-core:$(STABLE_IMAGE)-v$(VERSION_SHORT)
	docker push ghcr.io/allenai/olmo-core:$(STABLE_IMAGE)-v$(VERSION_SHORT)
	docker tag olmo-core:$(STABLE_IMAGE) ghcr.io/allenai/olmo-core:$(STABLE_IMAGE)-v$(VERSION)
	docker push ghcr.io/allenai/olmo-core:$(STABLE_IMAGE)-v$(VERSION)
	docker tag olmo-core:$(STABLE_IMAGE) ghcr.io/allenai/olmo-core:latest
	docker push ghcr.io/allenai/olmo-core:latest

.PHONY : beaker-image-stable
beaker-image-stable : stable-image
	./src/scripts/beaker/create_beaker_image.sh olmo-core:$(STABLE_IMAGE) olmo-core-$(STABLE_IMAGE) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh olmo-core:$(STABLE_IMAGE) olmo-core-$(STABLE_IMAGE)-v$(VERSION_SHORT) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh olmo-core:$(STABLE_IMAGE) olmo-core-$(STABLE_IMAGE)-v$(VERSION) $(BEAKER_WORKSPACE)

.PHONY : ghcr-image-nightly
ghcr-image-nightly : nightly-image
	docker tag olmo-core:$(NIGHTLY_IMAGE) ghcr.io/allenai/olmo-core:$(NIGHTLY_IMAGE)
	docker push ghcr.io/allenai/olmo-core:$(NIGHTLY_IMAGE)
	docker tag olmo-core:$(NIGHTLY_IMAGE) ghcr.io/allenai/olmo-core:$(NIGHTLY_IMAGE)-v$(VERSION_SHORT)
	docker push ghcr.io/allenai/olmo-core:$(NIGHTLY_IMAGE)-v$(VERSION_SHORT)
	docker tag olmo-core:$(NIGHTLY_IMAGE) ghcr.io/allenai/olmo-core:$(NIGHTLY_IMAGE)-v$(VERSION)
	docker push ghcr.io/allenai/olmo-core:$(NIGHTLY_IMAGE)-v$(VERSION)

.PHONY : beaker-image-nightly
beaker-image-nightly : nightly-image
	./src/scripts/beaker/create_beaker_image.sh olmo-core:$(NIGHTLY_IMAGE) olmo-core-$(NIGHTLY_IMAGE) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh olmo-core:$(NIGHTLY_IMAGE) olmo-core-$(NIGHTLY_IMAGE)-v$(VERSION_SHORT) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh olmo-core:$(NIGHTLY_IMAGE) olmo-core-$(NIGHTLY_IMAGE)-v$(VERSION) $(BEAKER_WORKSPACE)

.PHONY : get-beaker-workspace
get-beaker-workspace :
	@echo $(BEAKER_WORKSPACE)

.PHONY : get-full-beaker-image-name
get-full-beaker-image-name :
	@./src/scripts/beaker/get_full_image_name.sh $(IMAGE_NAME) $(BEAKER_WORKSPACE)
