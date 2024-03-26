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
