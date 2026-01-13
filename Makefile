PORT ?= 5001
IMAGE_NAME ?= ai-dial-adapter-anthropic
PLATFORM ?= linux/amd64
DEV_PYTHON ?= 3.11
DOCKER ?= docker
VENV_DIR ?= .venv
POETRY ?= $(VENV_DIR)/bin/poetry
POETRY_VERSION ?= 2.1.1
ARGS ?=

.PHONY: all init_env install build clean lint format test

all: build

init_env:
	python -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install poetry==$(POETRY_VERSION) --quiet

install: init_env
	$(POETRY) env use python$(DEV_PYTHON)
	$(POETRY) install

build: install
	$(POETRY) build

clean:
	$(POETRY) run python -m scripts.clean
	$(POETRY) env remove --all

lint: install
	$(POETRY) run nox -s lint

format: install
	$(POETRY) run nox -s format

test: install
	$(POETRY) run -- nox -s test -- $(ARGS)