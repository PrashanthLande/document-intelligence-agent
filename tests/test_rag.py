from document_intelligence_agent.rag.generator import RAGGenerator


def test_rag_generator():
    generator = RAGGenerator()

    evidence = [
        {
            "text": "Docling is a document processing framework.",
            "metadata": {
                "document_id": "example.pdf",
                "page_numbers": "1",
                "section": "Introduction",
                "item_types": "TextItem",
                "provenance": '[{"page": 1, "bbox": {"left": 10, "top": 20, "right": 100, "bottom": 40}, "coord_origin": "BOTTOMLEFT", "charspan": [0, 10]}]',
            },
        }
    ]
    result = generator.generate(
        query="What is Docling?",
        evidence=evidence,
    )

    assert result["answer"]
    assert isinstance(result["answer"], str)

    assert result["sources"]
    assert len(result["sources"]) == 1

    source = result["sources"][0]

    assert source["document_id"] == "example.pdf"
    assert source["page_numbers"] == "1"
    assert source["section"] == "Introduction"
    assert source["provenance"] == evidence[0]["metadata"]["provenance"]