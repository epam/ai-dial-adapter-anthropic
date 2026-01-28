import logging
from dataclasses import dataclass
from functools import cached_property
from logging import DEBUG
from typing import List, Optional, Tuple, Type, assert_never

from aidial_sdk.chat_completion import Message as DialMessage
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
    Omit,
    omit,
)
from anthropic._resource import AsyncAPIResource
from anthropic.lib.streaming import BetaInputJsonEvent as InputJsonEvent
from anthropic.lib.streaming import BetaTextEvent as TextEvent
from anthropic.lib.streaming import (
    ParsedBetaContentBlockStopEvent as ParsedContentBlockStopEvent,
)
from anthropic.lib.streaming._beta_types import (
    BetaCitationEvent as CitationEvent,
)
from anthropic.lib.streaming._beta_types import (
    BetaSignatureEvent as SignatureEvent,
)
from anthropic.lib.streaming._beta_types import (
    BetaThinkingEvent as ThinkingEvent,
)
from anthropic.lib.streaming._beta_types import (
    ParsedBetaMessageStopEvent as ParsedMessageStopEvent,
)
from anthropic.resources.beta import AsyncMessages as FirstPartyAsyncMessagesAPI
from anthropic.types.beta import (
    BetaBashCodeExecutionToolResultBlock as BashCodeExecutionToolResultBlock,
)
from anthropic.types.beta import (
    BetaCodeExecutionToolResultBlock as CodeExecutionToolResultBlock,
)
from anthropic.types.beta import (
    BetaContainerUploadBlock as ContainerUploadBlock,
)
from anthropic.types.beta import BetaMCPToolResultBlock as MCPToolResultBlock
from anthropic.types.beta import BetaMCPToolUseBlock as MCPToolUseBlock
from anthropic.types.beta import BetaMessage as ClaudeResponseMessage
from anthropic.types.beta import BetaMessageParam as ClaudeMessageParam
from anthropic.types.beta import (
    BetaRawContentBlockDeltaEvent as ContentBlockDeltaEvent,
)
from anthropic.types.beta import (
    BetaRawContentBlockStartEvent as ContentBlockStartEvent,
)
from anthropic.types.beta import BetaRawMessageDeltaEvent as MessageDeltaEvent
from anthropic.types.beta import BetaRawMessageStartEvent as MessageStartEvent
from anthropic.types.beta import (
    BetaRedactedThinkingBlock as RedactedThinkingBlock,
)
from anthropic.types.beta import BetaServerToolUseBlock as ServerToolUseBlock
from anthropic.types.beta import BetaTextBlock as TextBlock
from anthropic.types.beta import (
    BetaTextEditorCodeExecutionToolResultBlock as TextEditorCodeExecutionToolResultBlock,
)
from anthropic.types.beta import BetaThinkingBlock as ThinkingBlock
from anthropic.types.beta import BetaThinkingConfigParam as ThinkingConfigParam
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
from anthropic.types.beta.parsed_beta_message import (
    ParsedBetaTextBlock as ParsedTextBlock,
)

from aidial_adapter_anthropic._utils.json import json_dumps_short
from aidial_adapter_anthropic._utils.list import ListProjection
from aidial_adapter_anthropic.adapter._base import (
    ChatCompletionAdapter,
    default_preprocess_messages,
    keep_last,
    turn_based_partitioner,
)
from aidial_adapter_anthropic.adapter._claude.blocks import (
    IMAGE_ATTACHMENT_PROCESSOR,
    PDF_ATTACHMENT_PROCESSOR,
    PLAIN_TEXT_ATTACHMENT_PROCESSOR,
    create_text_block,
)
from aidial_adapter_anthropic.adapter._claude.citations import create_citations
from aidial_adapter_anthropic.adapter._claude.config import (
    ClaudeConfiguration,
    ClaudeConfigurationWithThinking,
)
from aidial_adapter_anthropic.adapter._claude.converters import (
    to_claude_messages,
    to_claude_tool_config,
    to_dial_finish_reason,
    to_dial_usage,
)
from aidial_adapter_anthropic.adapter._claude.params import ClaudeParameters
from aidial_adapter_anthropic.adapter._claude.state import MessageState
from aidial_adapter_anthropic.adapter._claude.tokenizer import (
    AnthropicTokenizer,
    ClaudeTokenizer,
    create_tokenizer,
)
from aidial_adapter_anthropic.adapter._claude.tools import (
    function_to_tool_messages,
    process_tools_block,
)
from aidial_adapter_anthropic.adapter._decorator.base import compose_decorators
from aidial_adapter_anthropic.adapter._decorator.preprocess import (
    preprocess_messages_decorator,
)
from aidial_adapter_anthropic.adapter._decorator.replicator import (
    replicator_decorator,
)
from aidial_adapter_anthropic.adapter._errors import ValidationError
from aidial_adapter_anthropic.adapter._truncate_prompt import (
    DiscardedMessages,
    truncate_prompt,
)
from aidial_adapter_anthropic.dial._attachments import (
    AttachmentProcessors,
    WithResources,
)
from aidial_adapter_anthropic.dial._message import parse_dial_message
from aidial_adapter_anthropic.dial.consumer import Consumer, ToolUseMessage
from aidial_adapter_anthropic.dial.request import (
    ModelParameters as DialParameters,
)
from aidial_adapter_anthropic.dial.resource import DialResource
from aidial_adapter_anthropic.dial.storage import FileStorage
from aidial_adapter_anthropic.dial.tools import ToolsMode

_log = logging.getLogger(__name__)


# Beta AsyncMessages doesn't provide the 'stream' method,
# so we enabled it via the adapter.
class _AsyncMessagesAdapter(AsyncAPIResource):
    create = FirstPartyAsyncMessagesAPI.create
    stream = FirstPartyAsyncMessagesAPI.stream

    def __init__(self, resource: AsyncAPIResource):
        super().__init__(resource._client)


# NOTE: it's not pydantic BaseModel, because
# anthropic.types.MessageParam.content is of Iterable type and
# pydantic automatically converts lists into
# list iterators following the type.
# See https://github.com/anthropics/anthropic-sdk-python/issues/656 for details.
@dataclass
class ClaudeRequest:
    params: ClaudeParameters
    messages: ListProjection[WithResources[ClaudeMessageParam]]

    @property
    def claude_messages(self) -> List[ClaudeMessageParam]:
        return [res.payload for res in self.messages.raw_list]

    @cached_property
    def resources(self) -> List[DialResource]:
        return [r for res in self.messages.raw_list for r in res.resources]

    def get_resource(self, index: int) -> DialResource | None:
        if 0 <= index < len(self.resources):
            return self.resources[index]
        return None


AnthropicClient = (
    AsyncAnthropic
    | AsyncAnthropicBedrock
    | AsyncAnthropicVertex
    | AsyncAnthropicFoundry
)


async def create_adapter(
    *,
    deployment: str,
    storage: FileStorage | None,
    client: AnthropicClient,
    default_max_tokens: int,
    supports_thinking: bool,
    supports_documents: bool,
    custom_tokenizer: ClaudeTokenizer | None = None,
) -> ChatCompletionAdapter:
    tokenizer = custom_tokenizer or AnthropicTokenizer(deployment, client)
    model = Adapter(
        deployment=deployment,
        storage=storage,
        client=client,
        tokenizer=tokenizer,
        default_max_tokens=default_max_tokens,
        supports_documents=supports_documents,
        supports_thinking=supports_thinking,
    )

    return compose_decorators(
        preprocess_messages_decorator(default_preprocess_messages),
        replicator_decorator(),
    )(model)


class Adapter(ChatCompletionAdapter):
    deployment: str
    storage: Optional[FileStorage]
    client: AnthropicClient

    tokenizer: ClaudeTokenizer
    default_max_tokens: int
    supports_thinking: bool
    supports_documents: bool

    async def configuration(self) -> Type[ClaudeConfiguration]:
        return ClaudeConfigurationWithThinking

    @property
    def attachment_processors(self) -> AttachmentProcessors:
        # Document support: https://docs.anthropic.com/en/docs/build-with-claude/pdf-support#supported-platforms-and-models
        document_processors = (
            [PDF_ATTACHMENT_PROCESSOR, PLAIN_TEXT_ATTACHMENT_PROCESSOR]
            if self.supports_documents
            else []
        )
        return AttachmentProcessors(
            text_handler=create_text_block,
            attachment_processors=(
                [IMAGE_ATTACHMENT_PROCESSOR] + document_processors
            ),
            file_storage=self.storage,
        )

    async def _prepare_claude_request(
        self, params: DialParameters, messages: List[DialMessage]
    ) -> ClaudeRequest:
        configuration = params.parse_configuration(await self.configuration())

        if len(messages) == 0:
            raise ValidationError("List of messages must not be empty")

        tools_config = to_claude_tool_config(params.tool_config)

        parsed_messages = [
            function_to_tool_messages(parse_dial_message(m)) for m in messages
        ]

        system_prompt, claude_messages = await to_claude_messages(
            self.attachment_processors, parsed_messages
        )

        thinking: ThinkingConfigParam | Omit = omit
        if (
            isinstance(configuration, ClaudeConfigurationWithThinking)
            and configuration.thinking is not None
        ):
            thinking = configuration.thinking.to_claude()

        temperature = omit
        if params.temperature is not None:
            # Mapping OpenAI temp [0,2] range to Anthropic temp [0,1] range
            temperature = params.temperature / 2

        if not isinstance(thinking, Omit) and thinking["type"] == "enabled":
            # Thinking isn’t compatible with temperature, top_p, or top_k
            # modifications as well as forced tool use.
            temperature = omit

        max_tokens = params.max_tokens or self.default_max_tokens

        claude_params = ClaudeParameters(
            max_tokens=max_tokens,
            stop_sequences=params.stop,
            system=system_prompt or omit,
            temperature=temperature,
            top_p=params.top_p or omit,
            tools=(tools_config and tools_config.tools) or omit,
            tool_choice=(tools_config and tools_config.tool_choice) or omit,
            thinking=thinking,
            betas=configuration.betas or omit,
        )

        return ClaudeRequest(params=claude_params, messages=claude_messages)

    async def _compute_discarded_messages(
        self, request: ClaudeRequest, max_prompt_tokens: int | None
    ) -> Tuple[DiscardedMessages | None, ClaudeRequest]:
        if max_prompt_tokens is None:
            return None, request

        discarded_messages, messages = await truncate_prompt(
            messages=request.messages.list,
            tokenizer=create_tokenizer(self.tokenizer, request.params),
            keep_message=keep_last,
            partitioner=turn_based_partitioner,
            model_limit=None,
            user_limit=max_prompt_tokens,
        )

        claude_messages = ListProjection(messages)

        discarded_messages = list(
            request.messages.to_original_indices(discarded_messages)
        )

        return discarded_messages, ClaudeRequest(
            params=request.params,
            messages=claude_messages,
        )

    async def chat(
        self,
        consumer: Consumer,
        params: DialParameters,
        messages: List[DialMessage],
    ):
        request = await self._prepare_claude_request(params, messages)

        discarded_messages, request = await self._compute_discarded_messages(
            request, params.max_prompt_tokens
        )

        if params.stream:
            await self.invoke_streaming(
                consumer,
                params.tools_mode,
                request,
                discarded_messages,
            )
        else:
            await self.invoke_non_streaming(
                consumer,
                params.tools_mode,
                request,
                discarded_messages,
            )

    async def count_prompt_tokens(
        self, params: DialParameters, messages: List[DialMessage]
    ) -> int:
        request = await self._prepare_claude_request(params, messages)
        tokenizer = create_tokenizer(self.tokenizer, request.params)
        return await tokenizer(request.messages.list)

    async def count_completion_tokens(self, string: str) -> int:
        return self.tokenizer.tokenize_text(string)

    async def compute_discarded_messages(
        self, params: DialParameters, messages: List[DialMessage]
    ) -> DiscardedMessages | None:
        request = await self._prepare_claude_request(params, messages)
        discarded_messages, _request = await self._compute_discarded_messages(
            request, params.max_prompt_tokens
        )
        return discarded_messages

    async def invoke_streaming(
        self,
        consumer: Consumer,
        tools_mode: ToolsMode | None,
        request: ClaudeRequest,
        discarded_messages: DiscardedMessages | None,
    ):
        if _log.isEnabledFor(DEBUG):
            msg = json_dumps_short(
                {"deployment": self.deployment, "request": request}
            )
            _log.debug(f"request: {msg}")

        async with (
            _AsyncMessagesAdapter(self.client.beta.messages).stream(
                messages=request.claude_messages,
                model=self.deployment,
                **request.params,
            ) as stream,
            consumer.create_stage("Thinking") as thinking_stage,
        ):
            stop_reason = None
            tool: ToolUseMessage | None = None

            async for event in stream:
                if _log.isEnabledFor(DEBUG):
                    _log.debug(f"response event: {json_dumps_short(event)}")

                match event:
                    case MessageStartEvent():
                        pass
                    case TextEvent(text=text):
                        consumer.append_content(text)

                    case ThinkingEvent(thinking=thinking):
                        thinking_stage.append_content(thinking)

                    case SignatureEvent() | MessageDeltaEvent():
                        pass

                    case ContentBlockStartEvent(content_block=content_block):
                        if isinstance(content_block, ToolUseBlock):
                            tool = process_tools_block(
                                consumer,
                                content_block,
                                tools_mode,
                                streaming=True,
                            )

                    case InputJsonEvent(partial_json=partial_json):
                        if tool:
                            tool.append_arguments(partial_json)
                        else:
                            _log.warning(
                                "The model generated tool input before start using it"
                            )

                    case ParsedContentBlockStopEvent(
                        content_block=content_block
                    ):
                        match content_block:
                            case TextBlock(citations=citations):
                                # The text content is already handled in TextEvent handler.
                                for citation in citations or []:
                                    create_citations(
                                        consumer, request.get_resource, citation
                                    )
                            case ToolUseBlock():
                                # Tool Use is processed in ContentBlockStartEvent and InputJsonEvent handlers
                                pass
                            case ThinkingBlock() | RedactedThinkingBlock():
                                # Thinking is processed in ThinkingEvent
                                pass
                            case (
                                ServerToolUseBlock()
                                | WebSearchToolResultBlock()
                                | CodeExecutionToolResultBlock()
                                | MCPToolUseBlock()
                                | MCPToolResultBlock()
                                | ContainerUploadBlock()
                                | BashCodeExecutionToolResultBlock()
                                | TextEditorCodeExecutionToolResultBlock()
                                | WebFetchToolResultBlock()
                                | ParsedTextBlock()
                                | BashCodeExecutionToolResultBlock()
                                | TextEditorCodeExecutionToolResultBlock()
                            ):
                                _log.error(
                                    f"Content block of type {content_block.type} isn't supported"
                                )
                            case _:
                                assert_never(content_block)

                    case ParsedMessageStopEvent(message=message):
                        consumer.add_usage(to_dial_usage(message.usage))
                        stop_reason = message.stop_reason
                        if self.supports_thinking:
                            consumer.choice.set_state(
                                MessageState(
                                    claude_message_content=message.content
                                ).to_dict()
                            )

                    case ContentBlockDeltaEvent() | CitationEvent():
                        pass

                    case _:
                        assert_never(event)

            consumer.close_content(
                to_dial_finish_reason(stop_reason, tools_mode)
            )

            consumer.set_discarded_messages(discarded_messages)

    async def invoke_non_streaming(
        self,
        consumer: Consumer,
        tools_mode: ToolsMode | None,
        request: ClaudeRequest,
        discarded_messages: DiscardedMessages | None,
    ):

        if _log.isEnabledFor(DEBUG):
            msg = json_dumps_short(
                {"deployment": self.deployment, "request": request}
            )
            _log.debug(f"request: {msg}")

        message: ClaudeResponseMessage = await self.client.beta.messages.create(
            messages=request.claude_messages,
            model=self.deployment,
            **request.params,
            stream=False,
        )

        if _log.isEnabledFor(DEBUG):
            _log.debug(f"response: {json_dumps_short(message)}")

        for content in message.content:
            match content:
                case TextBlock(text=text, citations=citations):
                    consumer.append_content(text)
                    for citation in citations or []:
                        create_citations(
                            consumer, request.get_resource, citation
                        )
                case ToolUseBlock():
                    process_tools_block(
                        consumer, content, tools_mode, streaming=False
                    )
                case ThinkingBlock(thinking=thinking):
                    with consumer.create_stage("Thinking") as stage:
                        stage.append_content(thinking)
                case RedactedThinkingBlock():
                    pass
                case (
                    ServerToolUseBlock()
                    | WebSearchToolResultBlock()
                    | CodeExecutionToolResultBlock()
                    | MCPToolUseBlock()
                    | MCPToolResultBlock()
                    | ContainerUploadBlock()
                    | BashCodeExecutionToolResultBlock()
                    | TextEditorCodeExecutionToolResultBlock()
                    | WebFetchToolResultBlock()
                    | ToolSearchToolResultBlock()
                ):
                    _log.error(
                        f"Content block of type {content.type} isn't supported"
                    )
                case _:
                    assert_never(content)

        if self.supports_thinking:
            consumer.choice.set_state(
                MessageState(claude_message_content=message.content).to_dict()
            )

        consumer.close_content(
            to_dial_finish_reason(message.stop_reason, tools_mode)
        )

        consumer.add_usage(to_dial_usage(message.usage))
        consumer.set_discarded_messages(discarded_messages)
