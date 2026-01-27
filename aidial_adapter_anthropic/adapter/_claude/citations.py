from typing import Callable, assert_never

from anthropic.types.beta import (
    BetaCitationCharLocation as CitationCharLocation,
)
from anthropic.types.beta import (
    BetaCitationContentBlockLocation as CitationContentBlockLocation,
)
from anthropic.types.beta import (
    BetaCitationPageLocation as CitationPageLocation,
)
from anthropic.types.beta import (
    BetaCitationSearchResultLocation as CitationSearchResultLocation,
)
from anthropic.types.beta import (
    BetaCitationsWebSearchResultLocation as CitationsWebSearchResultLocation,
)
from anthropic.types.beta import BetaTextCitation as TextCitation

from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.resource import DialResource


def _add_document_citation(
    consumer: Consumer,
    get_document: Callable[[int], DialResource | None],
    document_index: int,
):
    resource = get_document(document_index)
    document = None if resource is None else resource.to_attachment()

    # NOTE: multiple citations to the same document are merged into one citation
    # until we find a better API to handle citations embedded in text.
    display_index = consumer.add_citation_attachment(
        document_id=document_index, document=document
    )

    # NOTE: avoid adding citation URLs into the generated content,
    # since such references aren't easily portable (e.g. when a conversion is duplicated).
    consumer.append_content(f"[{display_index}]")


def create_citations(
    consumer: Consumer,
    get_document: Callable[[int], DialResource | None],
    citation: TextCitation,
):
    match citation:
        case CitationCharLocation(
            document_index=document_index
        ) | CitationPageLocation(document_index=document_index):
            _add_document_citation(consumer, get_document, document_index)

        # Custom document aren't supported yet
        case CitationContentBlockLocation():
            pass
        # web search isn't supported yet
        case CitationsWebSearchResultLocation():
            pass
        case CitationSearchResultLocation():
            pass
        case _:
            assert_never(citation)
