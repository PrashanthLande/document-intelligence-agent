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

            if item_type == "SectionHeaderItem":
                text = getattr(item, "text", None)

                if isinstance(text, str) and text:
                    current_section = text

                continue

            if item_type == "TableItem":
                text = self._table_to_text(item)
            else:
                text = getattr(item, "text", None)

            if not isinstance(text, str) or not text:
                continue

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

            page_numbers = sorted({entry["page"] for entry in provenance})

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

    def _table_to_text(self, table: Any) -> str:
        """Convert a Docling TableItem into deterministic retrieval text."""

        table_data = getattr(table, "data", None)

        if table_data is None:
            return ""

        rows = getattr(table_data, "table_cells", None)

        if not rows:
            return ""

        num_rows = getattr(table_data, "num_rows", None)
        num_cols = getattr(table_data, "num_cols", None)

        if not num_rows or not num_cols:
            return ""

        grid: list[list[str]] = [
            ["" for _ in range(num_cols)]
            for _ in range(num_rows)
        ]

        for cell in rows:
            row_index = getattr(cell, "start_row_offset_idx", None)
            col_index = getattr(cell, "start_col_offset_idx", None)

            if row_index is None or col_index is None:
                continue

            if not (0 <= row_index < num_rows and 0 <= col_index < num_cols):
                continue

            text = getattr(cell, "text", "")

            if not isinstance(text, str):
                text = str(text)

            grid[row_index][col_index] = text.strip()

        lines = [
            " | ".join(cell for cell in row if cell)
            for row in grid
        ]

        lines = [line for line in lines if line]

        return "\n".join(lines)