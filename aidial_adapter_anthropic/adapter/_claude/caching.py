import time

from aidial_sdk.chat_completion import CacheBreakpoint
from aidial_sdk.chat_completion import Message as DialMessage

_DIAL_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_DIAL_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"

# 5min is a default TTL for Clade cache breakpoints
# https://platform.claude.com/docs/en/build-with-claude/prompt-caching#ttl-support
_DEFAULT_TTL_SEC = 5 * 60


def _parse_ttl(ttl: str) -> int | None:
    try:
        for unit, secs in {"h": 3600, "m": 60}.items():
            if ttl[-1] == unit:
                return secs * int(ttl[:-1])
    except Exception:
        return None


def _ttl_from_breakpoint(breakpoint: CacheBreakpoint) -> int:
    s = (breakpoint.model_extra or {}).get("ttl")
    if s and isinstance(s, str) and (ttl := _parse_ttl(s)):
        return ttl
    return _DEFAULT_TTL_SEC


def get_response_headers_for_caching(
    automatic_cache_breakpoint: CacheBreakpoint | None,
    messages: list[DialMessage],
) -> dict | None:
    if automatic_cache_breakpoint is not None:
        ttl = _ttl_from_breakpoint(automatic_cache_breakpoint)
        idx = len(messages) - 1
    else:
        ttl = 0
        idx = None
        for i, message in enumerate(messages):
            if (
                (cf := message.custom_fields)
                and (breakpoint := cf.cache_breakpoint)
                and (msg_ttl := _ttl_from_breakpoint(breakpoint))
            ):
                ttl = max(ttl, msg_ttl)
                idx = i

        if idx is None:
            return None

    return {
        _DIAL_CACHE_BREAKPOINT_PATH: f"prefix.body.messages[{idx}]",
        _DIAL_CACHE_EXPIRE_AT: str(int(time.time()) + ttl),
    }
