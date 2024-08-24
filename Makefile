IMAGE_BASENAME = olmo-core
BEAKER_WORKSPACE = ai2/OLMo-core
BEAKER_USER = $(shell beaker account whoami --format=json | jq -r '.[0].name')

.PHONY : checks
checks : style-check lint-check type-check

.PHONY : style-check
style-check :
	isort --check .
	black --check .

.PHONY : lint-check
lint-check :
	ruff check .

.PHONY : type-check
type-check :
	mypy src/

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
	docker build -f src/Dockerfile -t $(IMAGE_BASENAME) .
	beaker image create $(IMAGE_BASENAME) --name $(IMAGE_BASENAME)-tmp --workspace $(BEAKER_WORKSPACE)
	beaker image delete $(BEAKER_USER)/$(IMAGE_BASENAME) || true
	beaker image rename $(BEAKER_USER)/$(IMAGE_BASENAME)-tmp $(IMAGE_BASENAME)
