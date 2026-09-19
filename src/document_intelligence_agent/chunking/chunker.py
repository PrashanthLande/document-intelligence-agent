from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling.datamodel.document import DoclingDocument


@dataclass
class Chunk:
    text: str
    document_id: str
    page_numbers: list[int]
    section: str | None
    item_types: list[str]
    provenance: list[dict[str, Any]]


class DocumentChunker:
    """Create provenance-aware chunks from a DoclingDocument."""

    def chunk(
            self,
            document: DoclingDocument,
            document_id: str | Path,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        current_section: str | None = None

        for item, _level in document.iterate_items():
            item_type = type(item).__name__
            text = getattr(item, "text", None)

            if not isinstance(text, str) or not text:
                continue

            if item_type == "SectionHeaderItem":
                current_section = text

            provenance: list[dict[str, Any]] = []

            for prov in getattr(item, "prov", []):
                provenance.append(
                    {
                        "page": prov.page_no,
                        "bbox": {
                            "left": prov.bbox.l,
                            "top": prov.bbox.t,
                            "right": prov.bbox.r,
                            "bottom": prov.bbox.b,
                        },
                        "coord_origin": prov.bbox.coord_origin.value,
                        "charspan": prov.charspan,
                    }
                )

            page_numbers = sorted(
                {entry["page"] for entry in provenance}
            )

            chunks.append(
                Chunk(
                    text=text,
                    document_id=str(document_id),
                    page_numbers=page_numbers,
                    section=current_section,
                    item_types=[item_type],
                    provenance=provenance,
                )
            )

        return chunks