from document_intelligence_agent.rag.generator import RAGGenerator


def test_rag_generator():
    generator = RAGGenerator()

    evidence = [
        {
            "text": (
                "Docling is a document understanding toolkit "
                "for converting and processing documents."
            ),
            "metadata": {
                "document_id": "example.pdf",
                "page_numbers": "1",
                "section": "Introduction",
            },
        }
    ]

    answer = generator.generate(
        query="What is Docling?",
        evidence=evidence,
    )

    assert answer
    assert isinstance(answer, str)