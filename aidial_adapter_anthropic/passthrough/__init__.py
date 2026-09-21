"""Transparent passthrough for the Anthropic Messages API.

Exposes a subset of the Anthropic API (messages, batches, and token counting)
as a mountable sub-application that forwards requests to an
upstream Anthropic client chosen per request.

Typical usage::

    from aidial_sdk import DIALApp
    from aidial_adapter_anthropic.passthrough import mount_anthropic_api

    app = DIALApp(...)

    async def get_client(request):
        return AsyncAnthropic(api_key=...)

    mount_anthropic_api(app, get_client)

Adapting a request to what its upstream cloud actually implements (dropping
beta flags and server tools the backend has no support for) needs no
configuration: it follows from the client the factory returns.
"""

from starlette.applications import Starlette

from aidial_adapter_anthropic.passthrough._middleware import AnthropicClient
from aidial_adapter_anthropic.passthrough._proxy import (
    ClientFactory,
    ClientT,
    create_anthropic_api_app,
)

__all__ = [
    "AnthropicClient",
    "ClientFactory",
    "create_anthropic_api_app",
    "mount_anthropic_api",
]


def mount_anthropic_api(
    app: Starlette,
    get_client: ClientFactory[ClientT],
    *,
    path: str = "/anthropic",
    name: str = "Claude API passthrough",
) -> None:
    passthrough_app = create_anthropic_api_app(get_client)
    app.mount(path=path, app=passthrough_app, name=name)
