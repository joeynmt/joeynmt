.PHONY: check format test

# Check formatting issues

check:
	yapf -dr joeynmt test/unit scripts/*.py
	pylint --rcfile=.pylintrc joeynmt test/unit scripts/*.py
	flake8 --max-line-length 88 joeynmt test/unit scripts/*.py

# Format source code automatically

format:
	isort --profile black joeynmt test/unit scripts/*.py
	yapf -ir joeynmt test/unit scripts/*.py

# Run tests for the package

test:
	python -m pytest
