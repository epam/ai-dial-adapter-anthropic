import pytest

from aidial_adapter_anthropic.passthrough import create_anthropic_api_app
from tests.unit_tests.anthropic_mocks import (
    AnthropicMocker,
    asgi_client,
    get_mocker_types,
)


@pytest.fixture(
    params=get_mocker_types(),
    ids=[cls.id for cls in get_mocker_types()],
)
def mocker(request):
    ret = request.param.create()
    with ret.router:
        yield ret


@pytest.fixture
async def http_client(mocker: AnthropicMocker):
    app = create_anthropic_api_app(mocker.make_client())
    async with asgi_client(app) as client:
        yield client
