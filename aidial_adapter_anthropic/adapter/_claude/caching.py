import time
from dataclasses import dataclass

from aidial_sdk.chat_completion import CacheBreakpoint, CacheBreakpointPath
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Tool as DialTool

# 5min is a default TTL for Clade cache breakpoints
# https://platform.claude.com/docs/en/build-with-claude/prompt-caching#ttl-support
_DEFAULT_TTL_SEC = 5 * 60


def _parse_ttl(ttl: str) -> int | None:
    try:
        for unit, secs in {"h": 3600, "m": 60, "s": 1}.items():
            if ttl[-1] == unit:
                return secs * int(ttl[:-1])
    except Exception:
        return None


def _ttl_from_breakpoint(breakpoint: CacheBreakpoint) -> int:
    s = (breakpoint.model_extra or {}).get("ttl")
    if s and isinstance(s, str) and (ttl := _parse_ttl(s)):
        return ttl
    return _DEFAULT_TTL_SEC


@dataclass
class CachingInfo:
    breakpoint_path: CacheBreakpointPath
    expired_at: str


def get_caching_info(
    automatic_cache_breakpoint: CacheBreakpoint | None,
    messages: list[DialMessage],
    tools: list[DialTool],
) -> CachingInfo | None:
    ttl = 0
    automatic_path = None
    message_path = None
    tool_path = None

    if automatic_cache_breakpoint is not None:
        ttl = _ttl_from_breakpoint(automatic_cache_breakpoint)
        automatic_path = CacheBreakpointPath.messages(len(messages) - 1)

    for i, message in enumerate(messages):
        if (
            (cf := message.custom_fields)
            and (breakpoint := cf.cache_breakpoint)
            and (msg_ttl := _ttl_from_breakpoint(breakpoint))
        ):
            ttl = max(ttl, msg_ttl)
            message_path = CacheBreakpointPath.messages(i)

    for i, tool in enumerate(tools):
        if (
            (cf := tool.custom_fields)
            and (breakpoint := cf.cache_breakpoint)
            and (tool_ttl := _ttl_from_breakpoint(breakpoint))
        ):
            ttl = max(ttl, tool_ttl)
            tool_path = CacheBreakpointPath.tools(i)

    path = automatic_path or message_path or tool_path

    if path is None:
        return None

    return CachingInfo(path, str(int(time.time()) + ttl))
