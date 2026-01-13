import pytest


@pytest.fixture(autouse=True)
def configure_unit_tests(monkeypatch, request):
    """
    Set up fake environment variables for unit tests.
    """
    if "tests/unit_tests" in request.node.nodeid:
        monkeypatch.setenv("AWS_DEFAULT_REGION", "test-region")
