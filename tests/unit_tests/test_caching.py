import pytest
from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
)
from anthropic import Omit
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam as CacheControlEphemeralParam,
)

from aidial_adapter_anthropic._utils.cache import CacheBreakpoint
from aidial_adapter_anthropic.adapter._claude.adapter import (
    Adapter,
    ClaudeRequest,
)
from aidial_adapter_anthropic.adapter._claude.blocks import (
    to_claude_cache_control,
)
from aidial_adapter_anthropic.dial.request import AdapterRequest

_EPHEMERAL = CacheControlEphemeralParam(type="ephemeral")


async def _to_clade_request(adapter: Adapter, request: dict) -> ClaudeRequest:
    req = ChatCompletionRequest.model_validate(request)
    return await adapter._prepare_claude_request(AdapterRequest.create(req))


def _user(content: str, *, cache_breakpoint: dict | None = None) -> dict:
    msg: dict = {"role": "user", "content": content}
    if cache_breakpoint is not None:
        msg["custom_fields"] = {"cache_breakpoint": cache_breakpoint}
    return msg


def _sys(content: str, *, cache_breakpoint: dict | None = None) -> dict:
    msg: dict = {"role": "system", "content": content}
    if cache_breakpoint is not None:
        msg["custom_fields"] = {"cache_breakpoint": cache_breakpoint}
    return msg


_BREAKPOINT = {"mode": "explicit"}
_PNG_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _text_part(text: str, *, breakpoint: bool = False) -> dict:
    part: dict = {"type": "text", "text": text}
    if breakpoint:
        part["prompt_cache_breakpoint"] = _BREAKPOINT
    return part


def _cache_controls(blocks: object) -> list:
    assert isinstance(blocks, list)
    result = []
    for block in blocks:
        assert isinstance(block, dict)
        result.append(block.get("cache_control"))
    return result


def _create_request_with_tool(add_breakpoint: bool) -> dict:
    tool = {"type": "function", "function": {"name": "get_weather"}}
    if add_breakpoint:
        tool["custom_fields"] = {"cache_breakpoint": {}}
    return {"messages": [_user("hi")], "tools": [tool]}


def test_to_claude_cache_control_returns_ephemeral():
    result = to_claude_cache_control(CacheBreakpoint())
    assert result == _EPHEMERAL


async def test_expire_at_is_not_forwarded_to_claude(adapter: Adapter):
    request = await _to_clade_request(
        adapter,
        {
            "messages": [_user("hi")],
            "custom_fields": {"cache_breakpoint": {"expire_at": "2099-01-01"}},
        },
    )
    assert request.params["cache_control"] == _EPHEMERAL


async def test_top_level_cache_breakpoint(adapter: Adapter):
    request = await _to_clade_request(
        adapter,
        {"messages": [_user("hi")], "custom_fields": {"cache_breakpoint": {}}},
    )
    assert request.params["cache_control"] == _EPHEMERAL


async def test_no_top_level_cache_breakpoint(adapter: Adapter):
    request = await _to_clade_request(adapter, {"messages": [_user("hi")]})
    assert isinstance(request.params["cache_control"], Omit)


async def test_user_message_cache_control(adapter: Adapter):
    request = await _to_clade_request(
        adapter, {"messages": [_user("hello", cache_breakpoint={})]}
    )
    content = request.claude_messages[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    last = content[0]
    assert isinstance(last, dict)
    assert last.get("cache_control") == _EPHEMERAL


async def test_user_message_no_cache_control(adapter: Adapter):
    request = await _to_clade_request(adapter, {"messages": [_user("hello")]})
    content = request.claude_messages[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], dict)
    assert "cache_control" not in content[0]


async def test_system_message_cache_control(adapter: Adapter):
    request = await _to_clade_request(
        adapter,
        {"messages": [_sys("be helpful", cache_breakpoint={}), _user("hi")]},
    )
    system = request.params["system"]
    assert isinstance(system, list)
    assert len(system) == 1
    assert system[0].get("cache_control") == _EPHEMERAL


async def test_tool_with_cache_control(adapter: Adapter):
    request = await _to_clade_request(adapter, _create_request_with_tool(True))
    tools = request.params["tools"]

    assert isinstance(tools, list)
    assert tools[0].get("cache_control") == _EPHEMERAL


async def test_tool_with_no_cache_control(adapter: Adapter):
    request = await _to_clade_request(adapter, _create_request_with_tool(False))

    tools = request.params["tools"]
    assert isinstance(tools, list)
    assert "cache_control" not in tools[0]


@pytest.mark.parametrize("ttl", ["5m", "1h", "foobar"])
async def test_cache_control_ttl(adapter: Adapter, ttl: str):
    request = await _to_clade_request(
        adapter,
        {
            "messages": [_user("hi")],
            "custom_fields": {"cache_breakpoint": {"ttl": ttl}},
        },
    )

    assert request.params["cache_control"] == {"type": "ephemeral", "ttl": ttl}


@pytest.mark.parametrize("ttl", [None, 300])
async def test_cache_control_without_usable_ttl(adapter: Adapter, ttl: object):
    """A missing or non-string TTL leaves the Claude default in place."""
    request = await _to_clade_request(
        adapter,
        {
            "messages": [_user("hi")],
            "custom_fields": {"cache_breakpoint": {"ttl": ttl}},
        },
    )

    assert request.params["cache_control"] == _EPHEMERAL


async def test_native_prompt_cache_options_enable_automatic_caching(
    adapter: Adapter,
):
    request = await _to_clade_request(
        adapter,
        {
            "messages": [_user("hi")],
            "prompt_cache_options": {"mode": "implicit"},
        },
    )
    assert request.params["cache_control"] == _EPHEMERAL


async def test_native_prompt_cache_options_default_to_implicit_mode(
    adapter: Adapter,
):
    request = await _to_clade_request(
        adapter, {"messages": [_user("hi")], "prompt_cache_options": {}}
    )
    assert request.params["cache_control"] == _EPHEMERAL


async def test_native_explicit_mode_disables_dial_automatic_caching(
    adapter: Adapter,
):
    request = await _to_clade_request(
        adapter,
        {
            "messages": [_user("hi")],
            "prompt_cache_options": {"mode": "explicit"},
            "custom_fields": {"cache_breakpoint": {}},
        },
    )
    assert isinstance(request.params["cache_control"], Omit)


async def test_native_prompt_cache_options_override_dial_ttl(
    adapter: Adapter,
):
    request = await _to_clade_request(
        adapter,
        {
            "messages": [_user("hi")],
            "prompt_cache_options": {"ttl": "1h"},
            "custom_fields": {"cache_breakpoint": {"ttl": "5m"}},
        },
    )
    assert request.params["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


@pytest.mark.parametrize("ttl", ["5m", "1h", "30m"])
async def test_native_prompt_cache_options_forward_ttl(
    adapter: Adapter, ttl: str
):
    """The TTL is forwarded verbatim, exactly as the DIAL breakpoint one is.

    Claude rejects anything but "5m" and "1h", so an unsupported TTL surfaces
    as an upstream error rather than being silently replaced.
    """
    request = await _to_clade_request(
        adapter,
        {"messages": [_user("hi")], "prompt_cache_options": {"ttl": ttl}},
    )
    assert request.params["cache_control"] == {
        "type": "ephemeral",
        "ttl": ttl,
    }


async def test_native_breakpoint_marks_its_own_content_part(adapter: Adapter):
    request = await _to_clade_request(
        adapter,
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _text_part("doc", breakpoint=True),
                        _text_part("question"),
                    ],
                }
            ]
        },
    )
    assert _cache_controls(request.claude_messages[0]["content"]) == [
        _EPHEMERAL,
        None,
    ]


async def test_native_breakpoint_is_anchored_to_the_part_index(
    adapter: Adapter,
):
    """A non-text part occupies a part index just like a text one does."""
    request = await _to_clade_request(
        adapter,
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _PNG_URL}},
                        _text_part("doc", breakpoint=True),
                        _text_part("question"),
                    ],
                }
            ]
        },
    )
    assert _cache_controls(request.claude_messages[0]["content"]) == [
        None,
        _EPHEMERAL,
        None,
    ]


async def test_native_breakpoint_marks_a_non_text_part(adapter: Adapter):
    request = await _to_clade_request(
        adapter,
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _PNG_URL},
                            "prompt_cache_breakpoint": _BREAKPOINT,
                        },
                        _text_part("question"),
                    ],
                }
            ]
        },
    )
    assert _cache_controls(request.claude_messages[0]["content"]) == [
        _EPHEMERAL,
        None,
    ]


async def test_native_breakpoint_wins_over_dial_message_breakpoint(
    adapter: Adapter,
):
    request = await _to_clade_request(
        adapter,
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _text_part("doc", breakpoint=True),
                        _text_part("question"),
                    ],
                    "custom_fields": {"cache_breakpoint": {"ttl": "1h"}},
                }
            ]
        },
    )
    assert _cache_controls(request.claude_messages[0]["content"]) == [
        _EPHEMERAL,
        None,
    ]


async def test_native_breakpoint_on_system_content_part(adapter: Adapter):
    request = await _to_clade_request(
        adapter,
        {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        _text_part("be helpful", breakpoint=True),
                        _text_part("be brief"),
                    ],
                },
                _user("hi"),
            ]
        },
    )
    assert _cache_controls(request.params["system"]) == [_EPHEMERAL, None]
