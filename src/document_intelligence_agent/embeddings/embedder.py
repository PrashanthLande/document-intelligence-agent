from sentence_transformers import SentenceTransformer

from document_intelligence_agent.chunking.chunker import Chunk


class DocumentEmbedder:
    """Generate embeddings for document chunks."""

    def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Generate one normalized embedding vector for each chunk."""
        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate one normalized embedding vector for a query."""
        return self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()