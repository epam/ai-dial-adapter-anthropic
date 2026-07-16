from typing import Literal

import pydantic
from aidial_sdk.chat_completion.request import StaticFunction
from anthropic.types.beta import (
    BetaWebSearchTool20250305Param,
    BetaWebSearchTool20260209Param,
)

from aidial_adapter_anthropic.adapter._errors import ValidationError

WebSearchToolParam = (
    BetaWebSearchTool20250305Param | BetaWebSearchTool20260209Param
)


class WebSearchTool(pydantic.BaseModel):
    name: Literal["web_search"]
    configuration: WebSearchToolParam


def parse_static_function(idx: str, func: StaticFunction) -> WebSearchToolParam:
    """
    Convert the ``web_search`` static function into the Anthropic web search
    server-tool definition. The ``name`` is defaulted from the static function
    so clients don't have to repeat it inside the configuration.

    See https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool
    """
    config = dict(func.configuration or {})
    name = config["name"] = config.get("name") or func.name
    obj = {"name": name, "configuration": config}

    try:
        tool = WebSearchTool.model_validate(obj)
        return tool.configuration
    except pydantic.ValidationError as e:
        error = e.errors()[0]
        path = ".".join(map(str, error["loc"]))
        raise ValidationError(
            "Invalid static tool definition at "
            f"'tools[{idx}].static_function.{path}': {error['msg']}"
        ) from None
