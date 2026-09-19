from pathlib import Path

from document_intelligence_agent.ingestion.loader import DocumentLoader


FIXTURE = Path("tests/fixtures/docling_technical_report.pdf")


def main() -> None:
    loader = DocumentLoader()
    document = loader.load(FIXTURE)

    print(f"Pages: {len(document.pages)}")
    print(f"Pictures: {len(document.pictures)}")
    print(f"Tables: {len(document.tables)}")

    print("\n--- Representative items ---")

    seen = set()

    for item, level in document.iterate_items():
        item_type = type(item).__name__

        if item_type in seen:
            continue

        if item_type not in {
            "TextItem",
            "SectionHeaderItem",
            "TableItem",
            "PictureItem",
        }:
            continue

        seen.add(item_type)

        print(f"\n=== {item_type} ===")
        print(f"text: {getattr(item, 'text', None)!r}")
        print(f"label: {getattr(item, 'label', None)!r}")
        print(f"prov: {getattr(item, 'prov', None)!r}")
        print(f"parent: {getattr(item, 'parent', None)!r}")
        print(f"self_ref: {getattr(item, 'self_ref', None)!r}")

        if len(seen) == 4:
            break


if __name__ == "__main__":
    main()