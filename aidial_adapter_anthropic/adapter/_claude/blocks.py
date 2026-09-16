import json
from typing import cast

from aidial_sdk.chat_completion import ToolCall
from anthropic.types.beta import (
    BetaBase64PDFSourceParam as Base64PDFSourceParam,
)
from anthropic.types.beta import (
    BetaCitationsConfigParam as CitationsConfigParam,
)
from anthropic.types.beta import BetaContentBlockParam as ContentBlockParam
from anthropic.types.beta import BetaImageBlockParam as ImageBlockParam
from anthropic.types.beta import (
    BetaPlainTextSourceParam as PlainTextSourceParam,
)
from anthropic.types.beta import (
    BetaRequestDocumentBlockParam as RequestDocumentBlockParam,
)
from anthropic.types.beta import BetaTextBlockParam as TextBlockParam
from anthropic.types.beta import (
    BetaToolResultBlockParam as ToolResultBlockParam,
)
from anthropic.types.beta import BetaToolUseBlockParam as ToolUseBlockParam
from anthropic.types.beta.beta_base64_image_source_param import (
    BetaBase64ImageSourceParam as Base64ImageSourceParam,
)
from anthropic.types.beta.beta_tool_result_block_param import (
    Content as ToolResultInnerContent,
)

from aidial_adapter_anthropic._utils.resource import Resource
from aidial_adapter_anthropic.adapter._claude.config import Configuration
from aidial_adapter_anthropic.dial._attachments import AttachmentProcessor


def _citations_config(config: Configuration | None) -> CitationsConfigParam:
    return CitationsConfigParam(
        enabled=config.enable_citations if config else False
    )


def create_text_block(text: str) -> TextBlockParam:
    return TextBlockParam(type="text", text=text)


def create_image_block(
    resource: Resource, config: Configuration | None
) -> ImageBlockParam:
    return ImageBlockParam(
        type="image",
        source=Base64ImageSourceParam(
            type="base64",
            media_type=resource.type,  # type: ignore
            data=resource.data_base64,
        ),
    )


def create_text_document_block(
    resource: Resource, config: Configuration | None
) -> RequestDocumentBlockParam:
    return RequestDocumentBlockParam(
        type="document",
        source=PlainTextSourceParam(
            type="text",
            media_type="text/plain",
            data=resource.data.decode("utf-8"),
        ),
        citations=_citations_config(config),
    )


def create_pdf_document_block(
    resource: Resource, config: Configuration | None
) -> RequestDocumentBlockParam:
    return RequestDocumentBlockParam(
        type="document",
        source=Base64PDFSourceParam(
            type="base64",
            media_type="application/pdf",
            data=resource.data_base64,
        ),
        citations=_citations_config(config),
    )


def create_tool_use_block(call: ToolCall) -> ContentBlockParam:
    return ToolUseBlockParam(
        type="tool_use",
        id=call.id,
        name=call.function.name,
        input=json.loads(call.function.arguments),
    )


def create_tool_result_block(
    tool_use_id: str, content: list[ContentBlockParam]
) -> ToolResultBlockParam:
    return ToolResultBlockParam(
        type="tool_result",
        tool_use_id=tool_use_id,
        content=cast(list[ToolResultInnerContent], content),
    )


IMAGE_ATTACHMENT_PROCESSOR = AttachmentProcessor(
    supported_types={
        "image/png": {"png"},
        "image/jpeg": {"jpeg", "jpg"},
        "image/gif": {"gif"},
        "image/webp": {"webp"},
    },
    handler=create_image_block,
)

PDF_ATTACHMENT_PROCESSOR = AttachmentProcessor(
    supported_types={"application/pdf": {"pdf"}},
    handler=create_pdf_document_block,
)

PLAIN_TEXT_ATTACHMENT_PROCESSOR = AttachmentProcessor(
    supported_types={
        "text/plain": {"txt"},
        "text/html": {"html", "htm"},
        "text/css": {"css"},
        "text/javascript": {"js"},
        "application/x-javascript": {"js"},
        "text/x-typescript": {"ts"},
        "application/x-typescript": {"ts"},
        "text/csv": {"csv"},
        "text/markdown": {"md"},
        "text/x-python": {"py"},
        "application/x-python-code": {"py"},
        "application/json": {"json"},
        "text/xml": {"xml"},
        "application/rtf": {"rtf"},
        "text/rtf": {"rtf"},
    },
    handler=create_text_document_block,
)
