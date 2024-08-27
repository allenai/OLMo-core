BASE_IMAGE = ghcr.io/allenai/pytorch:2.4.0-cuda12.1-python3.11
# NOTE: when upgrading the nightly version you also need to upgrade the torch version specification
# in 'pyproject.toml' to include that nightly version.
NIGHTLY_BASE_IMAGE = ghcr.io/allenai/pytorch:2.5.0.dev20240826-cuda12.1-python3.11

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
	docker build -f src/Dockerfile --build-arg BASE=$(BASE_IMAGE) -t $(IMAGE_BASENAME) .

.PHONY : beaker-image-stable
beaker-image-stable : stable-image
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME)-v$(VERSION_SHORT) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME)-v$(VERSION) $(BEAKER_WORKSPACE)

.PHONY : nightly-image
nightly-image :
	docker build -f src/Dockerfile --build-arg BASE=$(NIGHTLY_BASE_IMAGE) -t $(IMAGE_BASENAME)-nightly .

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
	@./src/scripts/beaker/get_full_image_name.sh $(IMAGE_BASENAME) $(BEAKER_WORKSPACE)
