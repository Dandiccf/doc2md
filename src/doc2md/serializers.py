"""Custom markdown serializers for image description placement."""

from __future__ import annotations

import json
import re
from typing import Any

from docling_core.transforms.serializer.base import BasePictureSerializer
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownPictureSerializer,
)
from docling_core.types.doc.document import (
    DoclingDocument,
    NodeItem,
    PictureItem,
)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def _parse_description(text: str) -> tuple[str, str]:
    """Parse structured JSON description. Returns (summary, detail).

    Falls back to ("", text) for plain text descriptions.
    """
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "summary" in data and "detail" in data:
            return data["summary"].strip(), data["detail"].strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return "", text


def _sanitize_alt_text(text: str) -> str:
    """Make text safe for use inside markdown ![alt](...) syntax."""
    alt = (
        text
        .replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )
    alt = re.sub(r"[\x00-\x1F\x7F]+", " ", alt)
    return alt.strip()


class DescriptionEnrichedImageSerializer(MarkdownPictureSerializer):
    """Custom picture serializer that places descriptions after the image link,
    with optional structured JSON support for alt text enrichment."""

    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: Any,
        doc: DoclingDocument,
        **kwargs: Any,
    ):
        # Get the standard output (caption + image link)
        parent_res = super().serialize(
            item=item, doc_serializer=doc_serializer, doc=doc, **kwargs,
        )
        text = parent_res.text

        desc = ""
        if (
            item.meta is not None
            and hasattr(item.meta, "description")
            and item.meta.description is not None
            and item.meta.description.text
        ):
            desc = strip_think_tags(item.meta.description.text).strip()

        if desc and "![Image](" in text:
            summary, detail = _parse_description(desc)

            # Inject summary into alt text (if available)
            if summary:
                alt = _sanitize_alt_text(summary)
                text = text.replace("![Image](", f"![{alt}](", 1)

            # Full description as blockquote
            blockquote = "\n".join(f"> {line}" for line in detail.splitlines())
            text = text + "\n\n" + blockquote
        elif desc:
            # Description exists but image tag doesn't match "![Image](" pattern
            blockquote = "\n".join(f"> {line}" for line in desc.splitlines())
            text = text + "\n\n" + blockquote

        return create_ser_result(text=text, span_source=item)


# Keep backward-compatible alias
DescriptionAfterImageSerializer = DescriptionEnrichedImageSerializer


class DescriptionEnrichedImageDocSerializer(MarkdownDocSerializer):
    """Markdown serializer that uses our custom picture serializer
    and tells the base serializer that PictureItems handle their own meta."""

    picture_serializer: BasePictureSerializer = DescriptionEnrichedImageSerializer()

    def _item_wraps_meta(self, item: NodeItem) -> bool:
        """Prevent the base serializer from prepending meta for PictureItems."""
        if isinstance(item, PictureItem):
            return True
        return super()._item_wraps_meta(item)


# Keep backward-compatible alias
DescriptionAfterImageDocSerializer = DescriptionEnrichedImageDocSerializer
