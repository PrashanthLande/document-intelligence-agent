from document_intelligence_agent.reranking.reranker import (
    DocumentReranker,
)


def test_reranker():
    results = [
        {
            "id": "1",
            "text": "Docling is a document understanding toolkit.",
            "metadata": {},
            "distance": 0.5,
        },
        {
            "id": "2",
            "text": "The weather forecast predicts rain tomorrow.",
            "metadata": {},
            "distance": 0.4,
        },
        {
            "id": "3",
            "text": "Docling supports document parsing and understanding.",
            "metadata": {},
            "distance": 0.6,
        },
    ]

    reranker = DocumentReranker()

    reranked = reranker.rerank(
        query="What is Docling?",
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2
    assert all("rerank_score" in result for result in reranked)
    assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"]

    print("\n--- Reranked results ---")

    for i, result in enumerate(reranked, start=1):
        print(f"\nResult {i}")
        print(f"Rerank score: {result['rerank_score']}")
        print(f"Text: {result['text']}")