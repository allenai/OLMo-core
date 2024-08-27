BASE_IMAGE = ghcr.io/allenai/pytorch:2.4.0-cuda12.1-python3.11
# NOTE: when upgrading the nightly version you also need to upgrade the torch version specification
# in 'pyproject.toml' to include that nightly version.
NIGHTLY_BASE_IMAGE = ghcr.io/allenai/pytorch:2.5.0.dev20240826-cuda12.1-python3.11

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

.PHONY : beaker-image
beaker-image :
	docker build -f src/Dockerfile --build-arg BASE=$(BASE_IMAGE) -t $(IMAGE_BASENAME) .
	beaker image create $(IMAGE_BASENAME) --name $(IMAGE_BASENAME)-tmp --workspace $(BEAKER_WORKSPACE)
	beaker image delete $(BEAKER_USER)/$(IMAGE_BASENAME) || true
	beaker image rename $(BEAKER_USER)/$(IMAGE_BASENAME)-tmp $(IMAGE_BASENAME)

.PHONY : beaker-image-nightly
beaker-image-nightly :
	docker build -f src/Dockerfile --build-arg BASE=$(NIGHTLY_BASE_IMAGE) -t $(IMAGE_BASENAME)-nightly .
	beaker image create $(IMAGE_BASENAME)-nightly --name $(IMAGE_BASENAME)-nightly-tmp --workspace $(BEAKER_WORKSPACE)
	beaker image delete $(BEAKER_USER)/$(IMAGE_BASENAME)-nightly || true
	beaker image rename $(BEAKER_USER)/$(IMAGE_BASENAME)-nightly-tmp $(IMAGE_BASENAME)-nightly
