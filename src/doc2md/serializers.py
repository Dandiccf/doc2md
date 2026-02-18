"""Custom markdown serializers for image description placement."""

from __future__ import annotations

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


class DescriptionAfterImageSerializer(MarkdownPictureSerializer):
    """Custom picture serializer that places descriptions after the image link."""

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

        parts = [parent_res.text]

        # Append description after the image
        if (
            item.meta is not None
            and hasattr(item.meta, "description")
            and item.meta.description is not None
            and item.meta.description.text
        ):
            desc = strip_think_tags(item.meta.description.text).strip()
            if desc:
                blockquote = "\n".join(f"> {line}" for line in desc.splitlines())
                parts.append(blockquote)

        return create_ser_result(
            text="\n\n".join(p for p in parts if p),
            span_source=item,
        )


class DescriptionAfterImageDocSerializer(MarkdownDocSerializer):
    """Markdown serializer that uses our custom picture serializer
    and tells the base serializer that PictureItems handle their own meta."""

    picture_serializer: BasePictureSerializer = DescriptionAfterImageSerializer()

    def _item_wraps_meta(self, item: NodeItem) -> bool:
        """Prevent the base serializer from prepending meta for PictureItems."""
        if isinstance(item, PictureItem):
            return True
        return super()._item_wraps_meta(item)
