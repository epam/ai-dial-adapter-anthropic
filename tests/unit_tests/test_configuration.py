import pytest
from aidial_sdk.chat_completion.request import (
    ReasoningEffort,
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaObject,
)
from anthropic import Omit

from aidial_adapter_anthropic.adapter import ValidationError
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


@pytest.mark.parametrize("effort", ["xhigh", "max"])
async def test_thinking_effort_from_config(adapter: Adapter, effort: str):
    config = {"thinking": {"type": "adaptive"}, "effort": effort}
    request = await adapter._prepare_claude_request(
        ModelParameters(configuration=config),
        [user("hello")],
    )
    assert request.params["thinking"] == {"type": "adaptive"}
    assert request.params["output_config"] == {"effort": effort}


async def test_thinking_effort_from_model_params(adapter: Adapter):
    config = {"thinking": {"type": "adaptive"}}
    request = await adapter._prepare_claude_request(
        ModelParameters(
            configuration=config, reasoning_effort=ReasoningEffort.MEDIUM
        ),
        [user("hello")],
    )
    assert request.params["thinking"] == {"type": "adaptive"}
    assert request.params["output_config"] == {"effort": "medium"}


async def test_thinking_effort_both_provided(adapter: Adapter):
    config = {"thinking": {"type": "adaptive"}, "effort": "high"}
    request = adapter._prepare_claude_request(
        ModelParameters(
            configuration=config, reasoning_effort=ReasoningEffort.MEDIUM
        ),
        [user("hello")],
    )
    msg = (
        'Conflicting reasoning effort values: "reasoning_effort"=medium '
        'and "custom_fields.configuration.effort"=high. '
        "Only one may be specified."
    )
    with pytest.raises(ValidationError, match=msg):
        await request


async def test_thinking_effort_preserved_when_response_format(adapter: Adapter):
    config = {"thinking": {"type": "adaptive"}, "effort": "medium"}
    json_schema = ResponseFormatJsonSchemaObject(
        name="MinimalSchema",
        schema={"type": "object", "properties": {}},
    )
    response_format = ResponseFormatJsonSchema(
        type="json_schema",
        json_schema=json_schema,
    )
    request = await adapter._prepare_claude_request(
        ModelParameters(
            configuration=config,
            response_format=response_format,
        ),
        [user("hello")],
    )

    output_config = request.params["output_config"]
    assert not isinstance(output_config, Omit)
    assert output_config.get("effort") == "medium"


async def test_configuration_schema_top_level_properties(adapter: Adapter):
    conf_cls = await adapter.configuration()
    conf_schema = conf_cls.model_json_schema()
    props = set(conf_schema["properties"])
    assert props == {
        "betas",
        "enable_citations",
        "thinking",
        "effort",
    }
