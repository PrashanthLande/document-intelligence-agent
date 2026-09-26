from sentence_transformers import CrossEncoder


class DocumentReranker:
    """Rerank retrieved document chunks using a cross-encoder."""

    def __init__(
            self,
            model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
            self,
            query: str,
            results: list[dict],
            top_k: int = 5,
    ) -> list[dict]:
        """Rerank retrieval results for a query."""
        pairs = [
            (query, result["text"])
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):
            reranked.append(
                {
                    **result,
                    "rerank_score": float(score),
                }
            )

        reranked.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]