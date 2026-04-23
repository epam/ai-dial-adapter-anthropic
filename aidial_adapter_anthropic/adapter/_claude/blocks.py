import json
from collections.abc import Iterable

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
    Content as ToolResultContent,
)

from aidial_adapter_anthropic._utils.resource import Resource
from aidial_adapter_anthropic.dial._attachments import AttachmentProcessor
from aidial_adapter_anthropic.dial._message import HumanToolResultMessage


def create_text_block(text: str) -> TextBlockParam:
    return TextBlockParam(text=text, type="text")


def create_image_block(resource: Resource) -> ImageBlockParam:
    return ImageBlockParam(
        source=Base64ImageSourceParam(
            data=resource.data_base64,
            media_type=resource.type,  # type: ignore
            type="base64",
        ),
        type="image",
    )


def create_text_document_block(
    resource: Resource, *, enable_citations: bool = False
) -> RequestDocumentBlockParam:
    return RequestDocumentBlockParam(
        source=PlainTextSourceParam(
            data=resource.data.decode("utf-8"),
            media_type="text/plain",
            type="text",
        ),
        type="document",
        citations=CitationsConfigParam(enabled=enable_citations),
    )


def create_pdf_document_block(
    resource: Resource, *, enable_citations: bool = False
) -> RequestDocumentBlockParam:
    return RequestDocumentBlockParam(
        source=Base64PDFSourceParam(
            data=resource.data_base64,
            media_type="application/pdf",
            type="base64",
        ),
        type="document",
        citations=CitationsConfigParam(enabled=enable_citations),
    )


def create_tool_use_block(call: ToolCall) -> ContentBlockParam:
    return ToolUseBlockParam(
        id=call.id,
        name=call.function.name,
        input=json.loads(call.function.arguments),
        type="tool_use",
    )


def create_tool_result_block(
    message: HumanToolResultMessage,
    custom_blocks: Iterable[ToolResultContent] | None = None,
) -> ToolResultBlockParam:
    """Create a tool result block with text content and optional custom blocks.
    Args:
        message: The tool result message containing the tool_use_id and text content.
        custom_blocks: Optional iterable of custom content blocks (images, documents, etc.)
                      to include in the tool result. Only blocks valid for tool results
                      (TextBlock, ImageBlock, DocumentBlock, etc.) are accepted.
    Returns:
        A ToolResultBlockParam with the tool result and all content blocks.
    """
    content_blocks: list[ToolResultContent] = []

    # Add custom blocks first (if provided)
    if custom_blocks:
        content_blocks.extend(custom_blocks)

    # Add text content block
    content_blocks.append(create_text_block(message.content))

    return ToolResultBlockParam(
        tool_use_id=message.id,
        type="tool_result",
        content=content_blocks,
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
