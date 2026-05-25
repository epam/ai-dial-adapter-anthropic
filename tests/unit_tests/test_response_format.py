from aidial_sdk.chat_completion.request import (
    ResponseFormatJsonObject,
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaObject,
    ResponseFormatText,
)

from aidial_adapter_anthropic.adapter._claude.converters import (
    to_claude_output_config,
)


class TestResponseFormatConversion:
    _EMPTY_EFFORT = None

    def test_no_response_format(self):
        output_config = to_claude_output_config(None, None)

        assert output_config is None

    def test_text_response_format(self):
        output_config = to_claude_output_config(
            ResponseFormatText(type="text"), self._EMPTY_EFFORT
        )

        assert output_config is None

    def test_json_object_response_format(self):
        output_config = to_claude_output_config(
            ResponseFormatJsonObject(type="json_object"), self._EMPTY_EFFORT
        )

        assert output_config is None

    def test_json_schema_response_format(self):
        test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
            "additionalProperties": False,
        }

        output_config = to_claude_output_config(
            ResponseFormatJsonSchema(
                type="json_schema",
                json_schema=ResponseFormatJsonSchemaObject(
                    name="PersonSchema",
                    schema=test_schema,
                ),
            ),
            self._EMPTY_EFFORT,
        )

        assert output_config == {
            "format": {"type": "json_schema", "schema": test_schema}
        }

    def test_json_schema_nested_objects(self):
        test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
        }

        output_config = to_claude_output_config(
            ResponseFormatJsonSchema(
                type="json_schema",
                json_schema=ResponseFormatJsonSchemaObject(
                    name="PersonSchema",
                    schema=test_schema,
                ),
            ),
            self._EMPTY_EFFORT,
        )

        assert output_config == {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "additionalProperties": False,
                },
            }
        }

    def test_json_schema_additional_properties_true(self):
        test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "additionalProperties": True,
        }

        output_config = to_claude_output_config(
            ResponseFormatJsonSchema(
                type="json_schema",
                json_schema=ResponseFormatJsonSchemaObject(
                    name="PersonSchema",
                    schema=test_schema,
                ),
            ),
            self._EMPTY_EFFORT,
        )

        assert output_config == {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "additionalProperties": False,
                },
            }
        }

    def test_json_schema_nested_additional_properties_true(self):
        test_schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}},
                    "additionalProperties": True,
                },
            },
        }

        output_config = to_claude_output_config(
            ResponseFormatJsonSchema(
                type="json_schema",
                json_schema=ResponseFormatJsonSchemaObject(
                    name="PersonSchema",
                    schema=test_schema,
                ),
            ),
            self._EMPTY_EFFORT,
        )

        assert output_config == {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "address": {
                            "type": "object",
                            "properties": {"street": {"type": "string"}},
                            "additionalProperties": False,
                        }
                    },
                    "additionalProperties": False,
                },
            }
        }
