ARGS ?=
POETRY ?= poetry
POETRY_PYTHON ?= python

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

lint: install
	$(POETRY) run nox -s lint

format: install
	$(POETRY) run nox -s format

test: install
	$(POETRY) run -- nox -s test $(if $(PYTHON),--python=$(PYTHON),) -- $(ARGS)

test_fast: install
	$(POETRY) run -- nox -s test $(if $(PYTHON),--python=$(PYTHON),) -- -m 'not slow' $(ARGS)

benchmark: install
	python -m benchmark.benchmark_merge_chunks

help:
	@echo '===================='
	@echo 'build                        - build the library'
	@echo 'clean                        - clean virtual env and build artifacts'
	@echo 'publish                      - publish the library to Pypi'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo '-- TESTS --'
	@echo 'test                         - run unit tests'
	@echo 'test_fast                    - run unit tests without slow tests'
	@echo 'test PYTHON=<python_version> - run unit tests with the specific python version'
