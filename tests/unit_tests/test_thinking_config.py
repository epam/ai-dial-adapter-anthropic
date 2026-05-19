from anthropic import Omit, omit

from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.dial.request import ModelParameters
from tests.utils.openai import user


async def test_adaptive_thinking_is_mapped_with_default_display(
    adapter: Adapter,
):
    request = await adapter._prepare_claude_request(
        ModelParameters(
            configuration={"thinking": {"type": "adaptive"}},
            temperature=1.0,
        ),
        [user("hello")],
    )

    assert request.params["thinking"] == {
        "type": "adaptive",
        "display": "omitted",
    }
    assert request.params["temperature"] == omit


async def test_adaptive_thinking_allows_custom_display(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(
            configuration={
                "thinking": {
                    "type": "adaptive",
                    "display": "summarized",
                }
            }
        ),
        [user("hello")],
    )

    assert request.params["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }


async def test_enabled_thinking_still_omits_temperature(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(
            configuration={
                "thinking": {"type": "enabled", "budget_tokens": 1024}
            },
            temperature=1.0,
        ),
        [user("hello")],
    )

    assert request.params["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }
    assert isinstance(request.params["temperature"], Omit)


async def test_thinking_configuration_free_format(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(
            configuration={
                "thinking": {"my": "very", "secret": "configuration"}
            },
        ),
        [user("hello")],
    )

    assert request.params["thinking"] == {
        "my": "very",
        "secret": "configuration",
    }
