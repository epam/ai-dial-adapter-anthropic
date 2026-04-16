ARGS ?=
POETRY ?= poetry
POETRY_PYTHON ?= python
VENV_DIR ?= .venv

-include .env.dev
export

all: build

init_env:
	$(POETRY) env use $(POETRY_PYTHON)

install: init_env
	$(POETRY) install --all-extras

build: install
	$(POETRY) build

clean:
	rm -rf $$($(POETRY) env info --path)
	rm -rf .nox
	rm -rf .pytest_cache
	rm -rf dist
	find . -type d -name __pycache__ | xargs rm -r

publish: build
	$(POETRY) publish -u __token__ -p $(PYPI_TOKEN) --skip-existing

install_git_hooks: install
	$(VENV_DIR)/bin/pre-commit install

lint: install
	$(POETRY) run nox -s lint

format: install
	$(POETRY) run nox -s format

test: install
	$(POETRY) run -- nox -s test $(if $(PYTHON),--python=$(PYTHON),) -- $(ARGS)

help:
	@echo '===================='
	@echo 'build                        - build the library'
	@echo 'clean                        - clean virtual env and build artifacts'
	@echo 'publish                      - publish the library to Pypi'
	@echo 'install_git_hooks            - install the git hooks'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo '-- TESTS --'
	@echo 'test                         - run unit tests'
	@echo 'test PYTHON=<python_version> - run unit tests with the specific python version'
