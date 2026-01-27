"""
An attempt to approximate the tokenizer for Claude models.

This tokenizer doesn't provide the precise token count,
because Anthropic doesn't provide the exact tokenization algorithm.

This tokenizer provides an *overestimation* of the request token count.
We need to be conservative, since the tokenizer is used in the prompt
truncation algorithm. So we are choosing to be unable to pack the request with tokens
as tightly as possible over making an additional chat completion request,
which is going to fail with a token overflow error.

1. For the text parts of request we count every byte in their UTF-8 encoding.
Note that the official Claude 2 tokenizer couldn't be used
for anything more than a very rough estimate:
https://github.com/anthropics/anthropic-sdk-python/blob/246a2978694b584429d4bbd5b44245ff8eac2ac2/src/anthropic/_client.py#L270-L283

2. For the image parts we use the official approximation:
> tokens = (width px * height px)/750
https://docs.anthropic.com/en/docs/build-with-claude/vision#calculate-image-costs

3. For the tool usage we use the official approximation:
https://docs.anthropic.com/en/docs/build-with-claude/tool-use#pricing
    a. tool-related components of the request are serialized to strings and tokenized as such,
    b. the hidden tool-enabling system prompt is accounted as per the documentation.

TODO: use the official tokenizer:
    https://docs.anthropic.com/en/docs/build-with-claude/token-counting
    once it's supported in Bedrock:
    https://github.com/anthropics/anthropic-sdk-python/blob/599f2b9a9501b8c98fb3132043c3ec71e3026f84/src/anthropic/lib/bedrock/_client.py#L61-L62
"""

import base64
import io
import json
import logging
import math
from typing import List, Literal, Tuple, assert_never

from anthropic._types import Base64FileInput
from anthropic.types.beta import (
    BetaBashCodeExecutionToolResultBlock as BashCodeExecutionToolResultBlock,
)
from anthropic.types.beta import (
    BetaCodeExecutionToolResultBlock as CodeExecutionToolResultBlock,
)
from anthropic.types.beta import (
    BetaContainerUploadBlock as ContainerUploadBlock,
)
from anthropic.types.beta import BetaContentBlockParam as ContentBlockParam
from anthropic.types.beta import BetaMCPToolResultBlock as MCPToolResultBlock
from anthropic.types.beta import BetaMCPToolUseBlock as MCPToolUseBlock
from anthropic.types.beta import BetaMessageParam as ClaudeMessage
from anthropic.types.beta import (
    BetaRedactedThinkingBlock as RedactedThinkingBlock,
)
from anthropic.types.beta import BetaServerToolUseBlock as ServerToolUseBlock
from anthropic.types.beta import BetaTextBlock as TextBlock
from anthropic.types.beta import (
    BetaTextEditorCodeExecutionToolResultBlock as TextEditorCodeExecutionToolResultBlock,
)
from anthropic.types.beta import BetaThinkingBlock as ThinkingBlock
from anthropic.types.beta import BetaToolParam as ToolParam
from anthropic.types.beta import (
    BetaToolResultBlockParam as ToolResultBlockParam,
)
from anthropic.types.beta import (
    BetaToolSearchToolResultBlock as ToolSearchToolResultBlock,
)
from anthropic.types.beta import BetaToolUseBlock as ToolUseBlock
from anthropic.types.beta import (
    BetaWebFetchToolResultBlock as WebFetchToolResultBlock,
)
from anthropic.types.beta import (
    BetaWebSearchToolResultBlock as WebSearchToolResultBlock,
)
from anthropic.types.beta.beta_image_block_param import Source
from anthropic.types.beta.beta_tool_result_block_param import (
    Content as ToolResultBlockParamContent,
)
from PIL import Image

from aidial_adapter_anthropic.adapter._claude.params import ClaudeParameters
from aidial_adapter_anthropic.adapter._tokenize import default_tokenize_string

_log = logging.getLogger(__name__)


class ApproximateTokenizer:
    def tokenize_text(self, text: str) -> int:
        return default_tokenize_string(text)

    def _get_image_size(
        self, image_data: str | Base64FileInput
    ) -> Tuple[int, int]:
        try:
            if not isinstance(image_data, str):
                raise ValueError("Images as files aren't yet supported.")

            image_bytes = base64.b64decode(image_data)
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.size
        except Exception:
            _log.exception("Cannot compute image size, assuming 1000x1000")
            return 1000, 1000

    def _tokenize_image(self, source: Source) -> int:
        match source["type"]:
            case "url" | "file":
                return 0
            case "base64":
                width, height = self._get_image_size(source["data"])
                return math.ceil((width * height) / 750.0)
            case _:
                assert_never(source)

    def _tokenize_tool_use(self, id: str, input: object, name: str) -> int:
        return self.tokenize_text(f"{id} {name} {json.dumps(input)}")

    def _tokenize_tool_result(self, message: ToolResultBlockParam) -> int:
        tokens: int = self.tokenize_text(message["tool_use_id"])
        if (content := message.get("content")) is not None:
            if isinstance(content, str):
                tokens += self.tokenize_text(content)
            else:
                for sub_message in content:
                    tokens += self._tokenize_sub_message(sub_message)
        return tokens

    def _tokenize_sub_message(
        self,
        message: ContentBlockParam | ToolResultBlockParamContent,
    ) -> int:
        if isinstance(message, dict):
            match message["type"]:
                case "text":
                    return self.tokenize_text(message["text"])
                case "image":
                    return self._tokenize_image(message["source"])
                case "tool_use":
                    return self._tokenize_tool_use(
                        message["id"], message["input"], message["name"]
                    )
                case "tool_result":
                    return self._tokenize_tool_result(message)
                case "document":
                    return self.tokenize_text(json.dumps(message))
                case "thinking":
                    return self.tokenize_text(message["thinking"])
                case "redacted_thinking":
                    return self.tokenize_text(message["data"])
                case "server_tool_use":
                    return self.tokenize_text(json.dumps(message["input"]))
                case "web_search_tool_result":
                    return self.tokenize_text(json.dumps(message["content"]))
                case (
                    "search_result"
                    | "code_execution_tool_result"
                    | "mcp_tool_use"
                    | "mcp_tool_result"
                    | "container_upload"
                    | "bash_code_execution_tool_result"
                    | "text_editor_code_execution_tool_result"
                    | "web_fetch_tool_result"
                    | "tool_search_tool_result"
                    | "tool_reference"
                ):
                    return 0
                case _:
                    assert_never(message["type"])
        else:
            match message:
                case TextBlock():
                    return self.tokenize_text(message.text)
                case ToolUseBlock():
                    return self._tokenize_tool_use(
                        message.id, message.input, message.name
                    )
                case ThinkingBlock(thinking=thinking):
                    return self.tokenize_text(thinking)
                case RedactedThinkingBlock(data=data):
                    return self.tokenize_text(data)
                case ServerToolUseBlock(input=input):
                    return self.tokenize_text(json.dumps(input))
                case WebSearchToolResultBlock(content=content):
                    return self.tokenize_text(json.dumps(content))
                case (
                    CodeExecutionToolResultBlock()
                    | MCPToolUseBlock()
                    | MCPToolResultBlock()
                    | ContainerUploadBlock()
                    | BashCodeExecutionToolResultBlock()
                    | TextEditorCodeExecutionToolResultBlock()
                    | WebFetchToolResultBlock()
                    | ToolSearchToolResultBlock()
                ):
                    return 0
                case _:
                    assert_never(message)

    def _tokenize_message(self, message: ClaudeMessage) -> int:
        tokens: int = 0
        content = message["content"]

        match content:
            case str():
                tokens += self.tokenize_text(content)
            case _:
                for item in content:
                    tokens += self._tokenize_sub_message(item)

        return tokens

    def _tokenize_messages(self, messages: List[ClaudeMessage]) -> int:
        # A rough estimation
        per_message_tokens = 5

        tokens: int = 0
        for message in messages:
            tokens += self._tokenize_message(message) + per_message_tokens
        return tokens

    def _tokenize_tool_param(self, tool: ToolParam) -> int:
        return self.tokenize_text(json.dumps(tool))

    def tokenize_tool_system_message(
        self,
        tool_choice: Literal["none", "auto", "any", "tool"],
    ) -> int:
        # Different models has different pricing for the tool use:
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview#pricing
        # Here we provide a default for Claude Opus 3
        return 530 if tool_choice in ("auto", "none") else 281

    async def tokenize(
        self, params: ClaudeParameters, messages: List[ClaudeMessage]
    ) -> int:
        tokens: int = 0

        if system := params["system"]:
            if isinstance(system, str):
                tokens += self.tokenize_text(system)
            else:
                for item in system:
                    tokens += self._tokenize_sub_message(item)

        if tools := params["tools"]:
            if tool_choice := params["tool_choice"]:
                choice = tool_choice["type"]
            else:
                choice = "auto"

            tokens += self.tokenize_tool_system_message(choice)

            for tool in tools:
                tokens += self._tokenize_tool_param(tool)

        tokens += self._tokenize_messages(messages)

        return tokens
