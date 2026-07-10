from urllib.parse import urlsplit

_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    return (
        scheme,
        (parsed.hostname or "").lower(),
        (parsed.port or _DEFAULT_PORTS.get(scheme)),
    )


def has_same_origin(a: str, b: str) -> bool:
    """Whether two URLs share the same origin (scheme, host and port).

    Comparing origins is safer than a string-prefix check: a URL like
    ``http://<host>@evil.example`` shares a prefix with ``http://<host>`` but
    has a different origin.
    """
    return _origin(a) == _origin(b)
