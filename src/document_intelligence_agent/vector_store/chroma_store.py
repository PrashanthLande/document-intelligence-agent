import chromadb
from chromadb.api.types import QueryResult

from document_intelligence_agent.chunking.chunker import Chunk


class ChromaVectorStore:
    """Store document chunks and their embeddings in ChromaDB."""

    def __init__(
            self,
            collection_name: str = "document_chunks",
    ) -> None:
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
            self,
            chunks: list[Chunk],
            embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        ids = [
            f"{chunk.document_id}:{index}"
            for index, chunk in enumerate(chunks)
        ]

        documents = [chunk.text for chunk in chunks]

        metadatas = [
            {
                "document_id": chunk.document_id,
                "page_numbers": ",".join(
                    str(page) for page in chunk.page_numbers
                ),
                "section": chunk.section or "",
                "item_types": ",".join(chunk.item_types),
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
            self,
            query_embedding: list[float],
            top_k: int = 5,
    ) -> QueryResult:
        """Search for the most relevant chunks."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )