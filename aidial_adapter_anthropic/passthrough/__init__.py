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
"""

from starlette.applications import Starlette

from aidial_adapter_anthropic.passthrough._proxy import (
    AnthropicClient,
    ClientFactory,
    ClientT,
    OnAnthropicBetaHeader,
    create_anthropic_api_app,
)

__all__ = [
    "AnthropicClient",
    "ClientFactory",
    "OnAnthropicBetaHeader",
    "create_anthropic_api_app",
    "mount_anthropic_api",
]


def mount_anthropic_api(
    app: Starlette,
    get_client: ClientFactory[ClientT],
    *,
    path: str = "/anthropic",
    name: str = "Claude API passthrough",
    on_anthropic_beta_header: OnAnthropicBetaHeader[ClientT] | None = None,
) -> None:
    passthrough_app = create_anthropic_api_app(
        get_client, on_anthropic_beta_header=on_anthropic_beta_header
    )
    app.mount(path=path, app=passthrough_app, name=name)
