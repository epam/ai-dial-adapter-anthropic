import anthropic
import pytest
from aidial_sdk.chat_completion.request import (
    ResponseFormatJsonObject,
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaObject,
    ResponseFormatText,
)

from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.dial.request import ModelParameters


@pytest.fixture
def adapter():
    return Adapter(
        deployment="test-deployment",
        storage=None,
        client=anthropic.AsyncAnthropic(),
        tokenizer=None,  # type: ignore
        default_max_tokens=1024,
        supports_thinking=True,
        supports_documents=True,
    )


class TestResponseFormatConversion:
    def test_no_response_format(self, adapter):
        params = ModelParameters(response_format=None)
        output_config = adapter._convert_response_format(params)

        assert isinstance(output_config, anthropic.Omit)

    def test_text_response_format(self, adapter):
        params = ModelParameters(
            response_format=ResponseFormatText(type="text")
        )
        output_config = adapter._convert_response_format(params)

        assert isinstance(output_config, anthropic.Omit)

    def test_json_object_response_format(self, adapter):
        params = ModelParameters(
            response_format=ResponseFormatJsonObject(type="json_object")
        )
        output_config = adapter._convert_response_format(params)

        assert not isinstance(output_config, anthropic.Omit)
        assert output_config["format"]["type"] == "json_schema"
        assert output_config["format"]["schema"]["type"] == "object"
        assert (
            output_config["format"]["schema"]["additionalProperties"] is False
        )

    def test_json_schema_response_format(self, adapter):
        test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
            "additionalProperties": False,
        }

        params = ModelParameters(
            response_format=ResponseFormatJsonSchema(
                type="json_schema",
                json_schema=ResponseFormatJsonSchemaObject(
                    name="PersonSchema",
                    schema=test_schema,
                ),
            )
        )
        output_config = adapter._convert_response_format(params)

        assert not isinstance(output_config, anthropic.Omit)
        assert output_config["format"]["type"] == "json_schema"
        assert output_config["format"]["schema"] == test_schema
