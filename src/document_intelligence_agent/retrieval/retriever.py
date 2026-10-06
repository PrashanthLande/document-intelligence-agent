from typing import Any

from document_intelligence_agent.embeddings.embedder import DocumentEmbedder
from document_intelligence_agent.vector_store.chroma_store import (
    ChromaVectorStore,
)


class Retriever:
    """Retrieve relevant document chunks for a query."""

    def __init__(
            self,
            embedder: DocumentEmbedder,
            vector_store: ChromaVectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
            self,
            query: str,
            top_k: int = 5,
            document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k chunks relevant to a query.

        If `document_ids` is given, only those documents are searched.
        """
        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_ids=document_ids,
        )

        retrieved_chunks: list[dict[str, Any]] = []

        for index in range(len(results["ids"][0])):
            retrieved_chunks.append(
                {
                    "id": results["ids"][0][index],
                    "text": results["documents"][0][index],
                    "metadata": results["metadatas"][0][index],
                    "distance": results["distances"][0][index],
                }
            )

        return retrieved_chunks