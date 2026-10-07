import contextlib
from typing import Self

from aidial_sdk.chat_completion import CacheBreakpoint as DialCacheBreakpoint
from aidial_sdk.chat_completion.request import PromptCacheOptions
from pydantic import BaseModel

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


class CacheBreakpoint(BaseModel):
    """A cache breakpoint, unifying the DIAL and the native OpenAI dialects."""

    ttl: str | None = None
    """How long to keep the cache entry, or None for the upstream default.

    Left untyped on purpose: each dialect spells the duration in its own
    vocabulary ("5m" and "1h" for DIAL, "30m" for the native API), so it is up
    to the upstream to accept or reject the one it is given.
    """

    @property
    def ttl_sec(self) -> int:
        return parse_ttl_sec(self.ttl)

    @classmethod
    def from_native(cls, options: PromptCacheOptions) -> Self | None:
        if options.mode == "explicit":
            return None
        return cls(ttl=options.ttl)

    @classmethod
    def from_dial(cls, breakpoint: DialCacheBreakpoint | None) -> Self | None:
        if breakpoint is None:
            return None

        # The DIAL breakpoint declares no `ttl` field, it arrives as an extra.
        # Its declared `expire_at` is a DIAL Core concern, not an upstream one.
        ttl = (breakpoint.model_extra or {}).get("ttl")
        return cls(ttl=ttl if isinstance(ttl, str) else None)
