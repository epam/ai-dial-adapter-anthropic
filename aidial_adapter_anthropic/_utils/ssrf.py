import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import aiohttp

from aidial_adapter_anthropic.adapter._errors import ValidationError

# Only regular web schemes are allowed. Everything else (e.g. ``file``,
# ``ftp``, ``gopher``) could be abused to reach local files or internal
# services.
_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Redirects are followed manually so that every hop can be validated against
# SSRF, otherwise a public URL could redirect to an internal address.
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def _is_public_ip(ip: str) -> bool:
    address = ipaddress.ip_address(ip)
    # IPv4-mapped IPv6 addresses (e.g. ``::ffff:169.254.169.254``) must be
    # judged by the semantics of the underlying IPv4 address. On Python < 3.13
    # ``is_global`` does not unwrap them automatically.
    if isinstance(address, ipaddress.IPv6Address):
        mapped = address.ipv4_mapped
        if mapped is not None:
            address = mapped
    return address.is_global


async def _resolve_host(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValidationError(
            f"Can't resolve the host of the file URL: {host}"
        ) from e
    return [info[4][0] for info in infos]


async def validate_public_url(url: str) -> None:
    """Guard against SSRF by rejecting file URLs that don't point to a
    publicly routable address.

    An attacker can supply an arbitrary attachment URL (e.g. the cloud
    metadata endpoint ``http://169.254.169.254``) and force the adapter to
    fetch it. We only allow ``http``/``https`` URLs whose host resolves
    exclusively to globally routable IP addresses.
    """
    parsed = urlsplit(url)

    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValidationError(
            f"Downloading files over the {scheme or 'empty'!r} URL scheme "
            "is not allowed"
        )

    host = parsed.hostname
    if not host:
        raise ValidationError("The file URL has no host")

    for ip in await _resolve_host(host):
        if not _is_public_ip(ip):
            raise ValidationError(
                "Downloading files from a non-public address "
                f"({ip}) is not allowed"
            )


async def download_public_file(url: str) -> bytes:
    """Download a file from an untrusted URL with SSRF protection.

    Redirects are followed manually so that the target of every hop is
    validated to be a public address before it is requested.
    """
    async with aiohttp.ClientSession() as session:
        for _ in range(_MAX_REDIRECTS + 1):
            await validate_public_url(url)

            async with session.get(url, allow_redirects=False) as response:
                if response.status in _REDIRECT_STATUSES and (
                    location := response.headers.get("Location")
                ):
                    url = urljoin(url, location)
                    continue

                response.raise_for_status()
                return await response.read()

    raise ValidationError("The file URL has too many redirects")
