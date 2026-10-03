"""Gradio UI for document question answering with source evidence."""

import time
import uuid
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import gradio as gr

from document_intelligence_agent.chunking.chunker import DocumentChunker
from document_intelligence_agent.embeddings.embedder import DocumentEmbedder
from document_intelligence_agent.ingestion.loader import DocumentLoader
from document_intelligence_agent.rag.generator import RAGGenerator
from document_intelligence_agent.rag.pipeline import RAGPipeline
from document_intelligence_agent.reranking.reranker import DocumentReranker
from document_intelligence_agent.retrieval.retriever import Retriever
from document_intelligence_agent.vector_store.chroma_store import (
    ChromaVectorStore,
)

# Models are loaded once at startup and shared by all users.
loader = DocumentLoader()
chunker = DocumentChunker()
embedder = DocumentEmbedder()
reranker = DocumentReranker()
generator = RAGGenerator()


T = TypeVar("T")

StatusUpdate = tuple[str, str | None]


def run_with_elapsed_time(
        task: Callable[[], T],
        step: str,
) -> Generator[StatusUpdate, None, T]:
    """Run a blocking task in a worker thread, yielding a status update every second.

    Returns the task's result once it finishes; exceptions propagate.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task)
        start = time.monotonic()

        while not future.done():
            elapsed = time.monotonic() - start
            yield f"{step} ({elapsed:.0f}s elapsed)", None
            time.sleep(1)

        return future.result()


def process_document(file_path: str | None) -> Iterator[StatusUpdate]:
    """Ingest an uploaded file into its own vector store collection.

    Each upload gets a fresh collection so users never search each other's
    documents and re-uploading the same file does not collide on chunk IDs.
    Yields (status message, collection name); the collection name is only
    set on the final update, once the document is ready to query.
    """
    if file_path is None:
        yield "Please upload a document first.", None
        return

    path = Path(file_path)

    chunks = yield from run_with_elapsed_time(
        lambda: chunker.chunk(loader.load(path), document_id=path.name),
        f"Step 1/2: Reading and analysing **{path.name}**. "
        "This usually takes 1-2 minutes",
    )

    if not chunks:
        yield f"No text could be extracted from **{path.name}**.", None
        return

    collection_name = f"doc-{uuid.uuid4().hex}"
    store = ChromaVectorStore(collection_name=collection_name)

    embeddings = yield from run_with_elapsed_time(
        lambda: embedder.embed_chunks(chunks),
        f"Step 2/2: Indexing {len(chunks)} chunks",
    )
    store.add(chunks=chunks, embeddings=embeddings)

    yield (
        f"Processed **{path.name}**: {len(chunks)} chunks indexed. "
        "Ask a question below.",
        collection_name,
    )


def answer_question(
        question: str,
        collection_name: str | None,
) -> tuple[str, str]:
    """Answer a question against this session's document.

    Returns (answer markdown, sources markdown).
    """
    if collection_name is None:
        return "Please upload and process a document first.", ""

    if not question.strip():
        return "Please enter a question.", ""

    store = ChromaVectorStore(collection_name=collection_name)
    pipeline = RAGPipeline(
        retriever=Retriever(embedder=embedder, vector_store=store),
        reranker=reranker,
        generator=generator,
    )

    result = pipeline.answer(question)

    return result["answer"], format_sources(result["evidence"])


def format_sources(evidence: list[dict]) -> str:
    """Render reranked evidence as a markdown list."""
    lines = []

    for index, item in enumerate(evidence, start=1):
        metadata = item["metadata"]
        snippet = item["text"][:300].replace("\n", " ")

        lines.append(
            f"**{index}. {metadata.get('document_id')}**, "
            f"page {metadata.get('page_numbers') or '?'}, "
            f"section: {metadata.get('section') or 'n/a'}\n\n"
            f"> {snippet}"
        )

    return "\n\n".join(lines)


with gr.Blocks(title="Document Intelligence Agent") as demo:
    gr.Markdown(
        "# Document Intelligence Agent\n"
        "Upload a document, then ask questions about it. "
        "Answers come with the evidence they were based on."
    )

    # Vector store collection for the current session's upload.
    collection_state = gr.State(None)

    file_input = gr.File(label="Document", type="filepath")
    process_button = gr.Button("Process document", variant="primary")
    status = gr.Markdown()

    question_input = gr.Textbox(label="Question", placeholder="Ask about the document...")
    ask_button = gr.Button("Ask", variant="primary")

    answer_output = gr.Markdown(label="Answer")

    with gr.Accordion("Sources", open=True):
        sources_output = gr.Markdown()

    process_button.click(
        process_document,
        inputs=[file_input],
        outputs=[status, collection_state],
    )

    gr.on(
        triggers=[ask_button.click, question_input.submit],
        fn=answer_question,
        inputs=[question_input, collection_state],
        outputs=[answer_output, sources_output],
    )


if __name__ == "__main__":
    demo.launch()
