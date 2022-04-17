.PHONY: check format test

# Check formatting issues

check:
	black --check --line-length 88 --target-version py39 joeynmt scripts/*.py
	isort --check-only --profile black joeynmt test/unit scripts/*.py
	flake8 --max-line-length 88 joeynmt test/unit scripts/*.py
	pylint --rcfile=.pylintrc --enable=useless-suppression joeynmt test/unit scripts/*.py

# Format source code automatically

format:
	black --line-length 88 --target-version py39 joeynmt scripts/*.py
	isort --profile black joeynmt test/unit scripts/*.py

# Run tests for the package

test:
	python -m pytest
