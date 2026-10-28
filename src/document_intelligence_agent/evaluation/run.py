"""Measure retrieval quality on a question set with known answer locations.

Usage:
    python -m document_intelligence_agent.evaluation.run eval/docling_technical_report.json
"""

import argparse
import uuid
from dataclasses import dataclass
from pathlib import Path

from document_intelligence_agent.chunking.chunker import DocumentChunker
from document_intelligence_agent.embeddings.embedder import DocumentEmbedder
from document_intelligence_agent.evaluation.dataset import (
    EvalDataset,
    is_relevant,
    load_dataset,
)
from document_intelligence_agent.evaluation.metrics import (
    first_relevant_rank,
    hit_at_k,
    mean_reciprocal_rank,
)
from document_intelligence_agent.ingestion.cache import CachingDocumentLoader
from document_intelligence_agent.ingestion.loader import DocumentLoader
from document_intelligence_agent.reranking.reranker import DocumentReranker
from document_intelligence_agent.retrieval.retriever import Retriever
from document_intelligence_agent.vector_store.chroma_store import (
    ChromaVectorStore,
)

REPORTED_KS = (1, 3, 5, 10, 20)


@dataclass
class QuestionResult:
    id: str
    question: str
    # Whether any chunk in the index is relevant at all.
    in_index: bool
    retrieval_rank: int | None
    reranked_rank: int | None


def evaluate(
        dataset: EvalDataset,
        retrieval_k: int,
        rerank_k: int,
        cache_dir: Path,
) -> tuple[list[QuestionResult], int]:
    """Index the dataset's documents, then rank evidence for every question.

    Returns the per-question results and the number of indexed chunks.
    """
    loader = CachingDocumentLoader(DocumentLoader(), cache_dir)
    chunker = DocumentChunker()
    embedder = DocumentEmbedder()
    reranker = DocumentReranker()
    store = ChromaVectorStore(collection_name=f"eval-{uuid.uuid4().hex}")

    all_chunks = []

    for document in dataset.documents:
        if not document.path.exists():
            raise FileNotFoundError(
                f"{document.path} is missing. Download it from "
                f"{document.source_url}"
            )

        chunks = chunker.chunk(
            loader.load(document.path),
            document_id=document.document_id,
        )
        store.add(chunks=chunks, embeddings=embedder.embed_chunks(chunks))
        all_chunks.extend(chunks)

    retriever = Retriever(embedder=embedder, vector_store=store)
    results = []

    for question in dataset.questions:
        in_index = any(
            is_relevant(question, c.document_id, c.page_numbers, c.text)
            for c in all_chunks
        )

        retrieved = retriever.retrieve(question.question, top_k=retrieval_k)
        reranked = reranker.rerank(question.question, retrieved, top_k=rerank_k)

        results.append(
            QuestionResult(
                id=question.id,
                question=question.question,
                in_index=in_index,
                retrieval_rank=first_relevant_rank(
                    [_is_relevant_result(question, r) for r in retrieved]
                ),
                reranked_rank=first_relevant_rank(
                    [_is_relevant_result(question, r) for r in reranked]
                ),
            )
        )

    return results, len(all_chunks)


def _is_relevant_result(question, result: dict) -> bool:
    """Apply `is_relevant` to a retriever result (metadata as stored)."""
    metadata = result["metadata"]
    pages = [int(p) for p in metadata["page_numbers"].split(",") if p]

    return is_relevant(question, metadata["document_id"], pages, result["text"])


def format_report(
        results: list[QuestionResult],
        chunk_count: int,
        retrieval_k: int,
        rerank_k: int,
) -> str:
    stages = [
        ("retrieval", [r.retrieval_rank for r in results], retrieval_k),
        ("reranked", [r.reranked_rank for r in results], rerank_k),
    ]
    header = "".join(f"{f'hit@{k}':>8}" for k in REPORTED_KS) + f"{'MRR':>8}"

    lines = [
        f"{len(results)} questions, {chunk_count} chunks "
        f"(retrieve {retrieval_k} -> rerank {rerank_k})",
        "",
        f"{'stage':<10}{header}",
    ]

    for name, ranks, list_size in stages:
        # hit@k is only meaningful up to the number of results returned.
        cells = "".join(
            f"{hit_at_k(ranks, k):>8.2f}" if k <= list_size else f"{'-':>8}"
            for k in REPORTED_KS
        )
        lines.append(f"{name:<10}{cells}{mean_reciprocal_rank(ranks):>8.2f}")

    not_in_index = [r for r in results if not r.in_index]
    misses = [r for r in results if r.in_index and r.reranked_rank is None]

    lines += ["", "Answer not in any chunk (parsing/chunking problem):"]
    lines += [f"  {r.id}: {r.question}" for r in not_in_index] or ["  none"]

    lines += ["", "Missed after reranking:"]
    lines += [
        f"  {r.id}: {r.question} (retrieval rank: {r.retrieval_rank or '-'})"
        for r in misses
    ] or ["  none"]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--rerank-k", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/eval"))
    args = parser.parse_args()

    results, chunk_count = evaluate(
        load_dataset(args.dataset),
        retrieval_k=args.retrieval_k,
        rerank_k=args.rerank_k,
        cache_dir=args.cache_dir,
    )

    print(format_report(results, chunk_count, args.retrieval_k, args.rerank_k))


if __name__ == "__main__":
    main()
