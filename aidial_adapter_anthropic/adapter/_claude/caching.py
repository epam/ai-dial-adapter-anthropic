import time
from dataclasses import dataclass

from aidial_sdk.chat_completion import CacheBreakpoint, CacheBreakpointPath
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Tool as DialTool

from aidial_adapter_anthropic._utils.cache import parse_ttl_sec


def _ttl_from_breakpoint(breakpoint: CacheBreakpoint) -> int:
    return parse_ttl_sec((breakpoint.model_extra or {}).get("ttl"))


@dataclass
class CacheInfo:
    breakpoint_path: CacheBreakpointPath
    expired_at: str


def get_cache_info(
    automatic_cache_breakpoint: CacheBreakpoint | None,
    messages: list[DialMessage],
    tools: list[DialTool],
) -> CacheInfo | None:
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

    return CacheInfo(path, str(int(time.time()) + ttl))
