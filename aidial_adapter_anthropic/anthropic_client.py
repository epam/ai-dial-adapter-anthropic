from datetime import datetime
from functools import cache
from typing import Mapping, Tuple

import anthropic
import httpx
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

from aidial_adapter_anthropic._utils.cache import ttl_cache
from aidial_adapter_anthropic._utils.env import get_env_int
from aidial_adapter_anthropic.upstream_config import (
    ApiKeyUpstreamConfig,
    UpstreamConfig,
)

Body = dict
Headers = Mapping[str, str]

ANTHROPIC_MAX_CONNECTIONS = get_env_int("ANTHROPIC_MAX_CONNECTIONS", 1000)
ANTHROPIC_MAX_KEEPALIVE_CONNECTIONS = get_env_int(
    "ANTHROPIC_MAX_KEEPALIVE_CONNECTIONS", 100
)
BOTOCORE_CLIENT_MAX_POOL_CONNECTIONS = get_env_int(
    "BOTOCORE_CLIENT_MAX_POOL_CONNECTIONS", 1000
)
ANTHROPIC_MAX_RETRY_ATTEMPTS = get_env_int("ANTHROPIC_MAX_RETRY_ATTEMPTS", 0)


@cache
def get_default_anthropic_timeout() -> httpx.Timeout:
    # Providing a timeout marginally different from the default Anthropic timeout
    # in order to disable the check that throws an error when
    # stream=False & max_tokens>=128K/6:
    # https://github.com/anthropics/anthropic-sdk-python/blob/f5bdf5137cc3da4d3663aedb8c63d54652981c3b/src/anthropic/resources/beta/messages/messages.py#L2175-L2176

    timeout = anthropic._constants.DEFAULT_TIMEOUT.as_dict()
    timeout["connect"] *= 1.0001  # type: ignore
    return httpx.Timeout(**timeout)


@ttl_cache
async def create_anthropic_client(
    upstream_config: UpstreamConfig,
) -> Tuple[datetime | None, AsyncAnthropicBedrock | AsyncAnthropic]:
    http_client = httpx.AsyncClient(
        timeout=get_default_anthropic_timeout(),
        follow_redirects=True,
        limits=httpx.Limits(
            # Max number of concurrent requests to the same upstream.
            # It limits number of concurrent requests.
            # `max_connections+1`-th request will be *blocked* until some other request has finished.
            max_connections=ANTHROPIC_MAX_CONNECTIONS,
            # Max number of idle connection to keep in a connection pool.
            max_keepalive_connections=ANTHROPIC_MAX_KEEPALIVE_CONNECTIONS,
        ),
    )

    if isinstance(upstream_config, ApiKeyUpstreamConfig):
        anthropic_client = AsyncAnthropic(
            api_key=upstream_config.api_key,
            http_client=http_client,
            max_retries=ANTHROPIC_MAX_RETRY_ATTEMPTS,
        )
        return (None, anthropic_client)
    else:
        (expiration, creds) = await upstream_config.get_credentials()
        anthropic_client = AsyncAnthropicBedrock(
            aws_region=upstream_config.region,
            aws_access_key=creds.aws_access_key_id,
            aws_secret_key=creds.aws_secret_access_key,
            aws_session_token=creds.aws_session_token,
            http_client=http_client,
            max_retries=ANTHROPIC_MAX_RETRY_ATTEMPTS,
        )
        return expiration, anthropic_client
