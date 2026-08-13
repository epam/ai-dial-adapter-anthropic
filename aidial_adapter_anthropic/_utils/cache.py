import contextlib

# 5min is a default TTL for Clade cache breakpoints
# https://platform.claude.com/docs/en/build-with-claude/prompt-caching#ttl-support
DEFAULT_TTL_SEC = 5 * 60

_UNITS = {"h": 3600, "m": 60, "s": 1}


def parse_ttl_sec(ttl: object) -> int:
    """The TTL of a cache breakpoint ("5m", "1h") in seconds.

    An absent or malformed value falls back to the Claude default: the TTL only
    tells DIAL Core how long to keep the upstream affinity, so a wrong guess
    costs cache hits, never correctness.
    """
    if isinstance(ttl, str):
        with contextlib.suppress(ValueError):
            for unit, secs in _UNITS.items():
                if ttl.endswith(unit):
                    return secs * int(ttl[:-1])
    return DEFAULT_TTL_SEC
