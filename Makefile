.PHONY : checks
checks : style-check lint-check type-check

.PHONY : style-check
style-check :
	isort --check .
	black --check .

.PHONY : lint-check
format-check :
	ruff check .

.PHONY : type-check
type-check :
	mypy .

.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch src/olmo_core/ --watch README.md docs/source/ docs/build/

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build
