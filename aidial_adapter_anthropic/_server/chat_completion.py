import logging
from typing import List, assert_never

from aidial_sdk.chat_completion import (
    ChatCompletion,
    ConfigurationRequest,
    Request,
    Response,
)
from aidial_sdk.chat_completion.request import ChatCompletionRequest
from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from aidial_sdk.deployment.tokenize import (
    TokenizeError,
    TokenizeInputRequest,
    TokenizeInputString,
    TokenizeOutput,
    TokenizeRequest,
    TokenizeResponse,
    TokenizeSuccess,
)
from aidial_sdk.deployment.truncate_prompt import (
    TruncatePromptError,
    TruncatePromptRequest,
    TruncatePromptResponse,
    TruncatePromptResult,
    TruncatePromptSuccess,
)
from typing_extensions import override

from aidial_adapter_anthropic._server.exceptions import (
    dial_exception_decorator,
    not_implemented_handler,
)
from aidial_adapter_anthropic.adapter.base import ChatCompletionAdapter
from aidial_adapter_anthropic.adapter.errors import UserError, ValidationError
from aidial_adapter_anthropic.anthropic_client import create_anthropic_client
from aidial_adapter_anthropic.claude.adapter import create_adapter
from aidial_adapter_anthropic.dial.consumer import ChoiceConsumer
from aidial_adapter_anthropic.dial.request import ModelParameters
from aidial_adapter_anthropic.upstream_config import (
    UpstreamConfig,
    parse_upstream_config,
)

log = logging.getLogger(__name__)


class AnthropicChatCompletion(ChatCompletion):
    def _get_deployment(self, request: FromRequestDeploymentMixin) -> str:
        return request.original_request.path_params["deployment_id"]

    @staticmethod
    async def _get_adapter(
        *, deployment: str, api_key: str, upstream_config: UpstreamConfig
    ) -> ChatCompletionAdapter:
        client = await create_anthropic_client(upstream_config)

        return await create_adapter(
            deployment,
            api_key,
            client,
            1536,
            supports_thinking=True,
            supports_documents=True,
        )

    async def _get_model(
        self, request: FromRequestDeploymentMixin
    ) -> ChatCompletionAdapter:
        return await self._get_adapter(
            deployment=self._get_deployment(request),
            api_key=request.api_key,
            upstream_config=await parse_upstream_config(request),
        )

    @override
    @dial_exception_decorator
    @not_implemented_handler
    async def configuration(self, request: ConfigurationRequest):
        model = await self._get_model(request)
        cls = await model.configuration()
        return cls.schema()

    @dial_exception_decorator
    async def chat_completion(self, request: Request, response: Response):
        deployment = self._get_deployment(request)
        response.set_model(deployment)

        model = await self._get_model(request)
        params = ModelParameters.create(request)

        with ChoiceConsumer(response) as consumer:
            try:
                await model.chat(consumer, params, request.messages)
            except UserError as e:
                await e.report_usage(consumer.choice)
                await response.aflush()
                raise e

        log.debug(f"usage: {consumer.usage}")

    @override
    @dial_exception_decorator
    @not_implemented_handler
    async def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        model = await self._get_model(request)

        outputs: List[TokenizeOutput] = []
        for input in request.inputs:
            match input:
                case TokenizeInputRequest():
                    outputs.append(
                        await self._tokenize_request(model, input.value)
                    )
                case TokenizeInputString():
                    outputs.append(
                        await self._tokenize_string(model, input.value)
                    )
                case _:
                    assert_never(input.type)
        return TokenizeResponse(outputs=outputs)

    async def _tokenize_string(
        self, model: ChatCompletionAdapter, value: str
    ) -> TokenizeOutput:
        try:
            tokens = await model.count_completion_tokens(value)
            return TokenizeSuccess(token_count=tokens)
        except NotImplementedError:
            raise
        except Exception as e:
            log.exception("Error tokenizing string")
            return TokenizeError(error=str(e))

    async def _tokenize_request(
        self, model: ChatCompletionAdapter, request: ChatCompletionRequest
    ) -> TokenizeOutput:
        params = ModelParameters.create(request)

        try:
            token_count = await model.count_prompt_tokens(
                params, request.messages
            )
            return TokenizeSuccess(token_count=token_count)
        except NotImplementedError:
            raise
        except Exception as e:
            log.exception("Error tokenizing request")
            return TokenizeError(error=str(e))

    @override
    @dial_exception_decorator
    @not_implemented_handler
    async def truncate_prompt(
        self, request: TruncatePromptRequest
    ) -> TruncatePromptResponse:
        model = await self._get_model(request)

        outputs: List[TruncatePromptResult] = []
        for input in request.inputs:
            outputs.append(await self._truncate_prompt_request(model, input))
        return TruncatePromptResponse(outputs=outputs)

    async def _truncate_prompt_request(
        self, model: ChatCompletionAdapter, request: ChatCompletionRequest
    ) -> TruncatePromptResult:
        try:
            params = ModelParameters.create(request)

            if params.max_prompt_tokens is None:
                raise ValidationError("max_prompt_tokens is required")

            discarded_messages = await model.compute_discarded_messages(
                params, request.messages
            )

            return TruncatePromptSuccess(
                discarded_messages=discarded_messages or []
            )
        except NotImplementedError:
            raise
        except Exception as e:
            log.exception("Error truncating prompt")
            return TruncatePromptError(error=str(e))
