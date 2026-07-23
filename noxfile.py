import os

import nox

nox.options.reuse_existing_virtualenvs = True
if os.environ.get("CI"):
    nox.options.default_venv_backend = "none"

SRC = ["aidial_adapter_anthropic", "tests", "noxfile.py"]


def format_with_args(session: nox.Session, *args):
    session.run("autoflake", *args)
    session.run("isort", *args)
    session.run("black", *args)


@nox.session
def lint(session: nox.Session):
    """Runs linters and fixers"""
    try:
        session.run("poetry", "install", "--with", "lint", external=True)
        session.run("poetry", "check", "--lock", "--strict", external=True)
        session.run("ruff", "check", *SRC)
        session.run("ruff", "format", "--check", *SRC)
        session.run("pyright", *SRC)
    except Exception:
        session.error(
            "linting has failed. Run 'make format' to fix formatting and fix other errors manually"
        )


@nox.session
def format(session: nox.Session):
    """Runs linters and fixers"""
    session.run("poetry", "install", "--only", "lint", external=True)
    session.run("ruff", "check", "--fix", *SRC)
    session.run("ruff", "format", *SRC)


@nox.session(python=["3.11", "3.12", "3.13"])
@nox.parametrize("pydantic", ["2.8.2", "2.12.5"])
def test(session: nox.Session, pydantic: str) -> None:
    """Runs tests"""
    session.run("poetry", "install", external=True)
    session.install(f"pydantic=={pydantic}")
    session.run("pytest", *session.posargs, env={"PYDANTIC_V2": "1"})
