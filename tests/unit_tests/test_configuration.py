from anthropic import Omit

from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.dial.request import ModelParameters
from tests.utils.openai import user


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


async def test_configuration_schema_top_level_properties(adapter: Adapter):
    conf_cls = await adapter.configuration()
    conf_schema = conf_cls.model_json_schema()
    props = set(conf_schema["properties"])
    assert props == {"betas", "enable_citations", "thinking"}
