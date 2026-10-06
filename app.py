"""Gradio UI for document question answering with source evidence.

The screen has three panels, like NotebookLM:
  Sources (left)  - upload documents and choose which ones to search
  Chat (middle)   - questions and answers, with history
  Evidence (right) - the sources used for the latest answer
"""

import html
import logging
import time
import uuid
from collections.abc import Callable, Collection, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeVar

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

logger = logging.getLogger(__name__)

# Models are loaded once at startup and shared by all users.
loader = DocumentLoader()
chunker = DocumentChunker()
embedder = DocumentEmbedder()
reranker = DocumentReranker()
generator = RAGGenerator()


class SourceInfo(TypedDict):
    """One uploaded document in the current session."""

    status: Literal["queued", "indexing", "ready", "failed"]
    chunk_count: int
    selected: bool


# Session sources keyed by display name, which is also the document ID
# stored with every chunk in the vector store.
Sources = dict[str, SourceInfo]

EVIDENCE_PLACEHOLDER = (
    '<p class="evidence-empty">'
    "The sources behind the latest answer will appear here."
    "</p>"
)

SNIPPET_CHARS = 300

STYLESHEET = Path(__file__).parent / "app.css"

T = TypeVar("T")


def run_with_elapsed_time(
        task: Callable[[], T],
        step: str,
        *unchanged: Any,
) -> Generator[tuple[Any, ...], None, T]:
    """Run a blocking task in a worker thread, yielding a progress update every second.

    Each update is (progress message, *unchanged), so the caller's other
    outputs stay as they are. Returns the task's result once it finishes;
    exceptions propagate.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task)
        start = time.monotonic()

        while not future.done():
            elapsed = time.monotonic() - start
            yield f"{step} ({elapsed:.0f}s elapsed)", *unchanged
            time.sleep(1)

        return future.result()


def unique_name(name: str, taken: Collection[str]) -> str:
    """Return `name`, or `name (2)`, `name (3)`, ... if it is already taken."""
    if name not in taken:
        return name

    path = Path(name)
    counter = 2

    while f"{path.stem} ({counter}){path.suffix}" in taken:
        counter += 1

    return f"{path.stem} ({counter}){path.suffix}"


def process_documents(
        file_paths: list[str] | None,
        collection_name: str | None,
        sources: Sources,
) -> Iterator[tuple[str, str | None, Sources]]:
    """Index uploaded files, one by one, into this session's collection.

    Each session has its own collection, so users never search each other's
    documents. Yields (progress message, collection name, sources).
    """
    if not file_paths:
        return

    if collection_name is None:
        collection_name = f"session-{uuid.uuid4().hex}"

    store = ChromaVectorStore(collection_name=collection_name)

    # Show every new file in the sources list straight away.
    sources = dict(sources)
    queued: list[tuple[str, Path]] = []

    for file_path in file_paths:
        path = Path(file_path)
        name = unique_name(path.name, sources)
        sources[name] = SourceInfo(status="queued", chunk_count=0, selected=False)
        queued.append((name, path))

    yield "", collection_name, sources

    for name, path in queued:
        sources = {**sources, name: {**sources[name], "status": "indexing"}}
        yield "", collection_name, sources

        try:
            chunks = yield from run_with_elapsed_time(
                lambda: chunker.chunk(loader.load(path), document_id=name),
                f"Reading and analysing **{name}**. "
                "This usually takes 1-2 minutes",
                gr.skip(),
                gr.skip(),
            )

            if chunks:
                embeddings = yield from run_with_elapsed_time(
                    lambda: embedder.embed_chunks(chunks),
                    f"Indexing {len(chunks)} chunks of **{name}**",
                    gr.skip(),
                    gr.skip(),
                )
                store.add(chunks=chunks, embeddings=embeddings)
        except Exception:
            logger.exception("Failed to process %s", name)
            chunks = []

        if chunks:
            info = SourceInfo(
                status="ready",
                chunk_count=len(chunks),
                selected=True,
            )
        else:
            info = SourceInfo(status="failed", chunk_count=0, selected=False)

        sources = {**sources, name: info}
        yield "", collection_name, sources


def set_source_selected(
        name: str,
        selected: bool,
        sources: Sources,
) -> Sources:
    """Tick or untick one source for searching."""
    return {**sources, name: {**sources[name], "selected": selected}}


def remove_source(
        name: str,
        collection_name: str | None,
        sources: Sources,
) -> Sources:
    """Remove one source from the session and from the vector store."""
    if collection_name is not None and sources[name]["status"] == "ready":
        ChromaVectorStore(collection_name=collection_name).delete_document(name)

    return {
        other: info
        for other, info in sources.items()
        if other != name
    }


def searchable_document_ids(sources: Sources) -> list[str]:
    """Return the sources that are ready and ticked for searching."""
    return [
        name
        for name, info in sources.items()
        if info["status"] == "ready" and info["selected"]
    ]


def status_label(info: SourceInfo) -> str:
    """Short status text shown under a source's name."""
    if info["status"] == "queued":
        return "Waiting..."

    if info["status"] == "indexing":
        return "Indexing..."

    if info["status"] == "failed":
        return "Could not read this file"

    return f"{info['chunk_count']} chunks"


def set_question_enabled(sources: Sources) -> tuple[dict, dict]:
    """Enable the question box and Ask button only when there is something to search."""
    ready = bool(searchable_document_ids(sources))
    return gr.update(interactive=ready), gr.update(interactive=ready)


def submit_question(
        question: str,
        history: list[dict],
) -> tuple[Any, Any, str]:
    """Show the question in the chat right away and clear the input box.

    Returns (input box, chat history, question to answer).
    """
    question = question.strip()

    if not question:
        return gr.skip(), gr.skip(), ""

    return "", history + [{"role": "user", "content": question}], question


def answer_question(
        question: str,
        history: list[dict],
        collection_name: str | None,
        sources: Sources,
) -> tuple[Any, Any]:
    """Answer the submitted question against the ticked sources.

    Returns (chat history, evidence HTML).
    """
    if not question:
        return gr.skip(), gr.skip()

    document_ids = searchable_document_ids(sources)

    if collection_name is None or not document_ids:
        reply = "Please tick at least one ready source to search."
        return history + [{"role": "assistant", "content": reply}], gr.skip()

    store = ChromaVectorStore(collection_name=collection_name)
    pipeline = RAGPipeline(
        retriever=Retriever(embedder=embedder, vector_store=store),
        reranker=reranker,
        generator=generator,
    )

    result = pipeline.answer(question, document_ids=document_ids)

    return (
        history + [{"role": "assistant", "content": result["answer"]}],
        format_evidence(question, result["evidence"]),
    )


def format_pages(page_numbers: str) -> str:
    """Turn stored page numbers like "1,2" into "Pages 1–2"."""
    pages = sorted(int(page) for page in page_numbers.split(",") if page)

    if not pages:
        return "Page unknown"

    if len(pages) == 1:
        return f"Page {pages[0]}"

    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"Pages {pages[0]}–{pages[-1]}"

    return "Pages " + ", ".join(str(page) for page in pages)


def shorten(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Cut text to at most `limit` characters, at a word boundary."""
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit].rsplit(" ", 1)[0] + "…"


def format_evidence(question: str, evidence: list[dict]) -> str:
    """Render reranked evidence as HTML cards, one per source.

    All document text is escaped, since it comes from user uploads.
    """
    cards = []

    for index, item in enumerate(evidence, start=1):
        metadata = item["metadata"]
        details = format_pages(metadata.get("page_numbers") or "")

        if metadata.get("section"):
            details += f" · {metadata['section']}"

        cards.append(
            '<div class="evidence-card">'
            '<div class="evidence-header">'
            f'<span class="evidence-number">{index}</span>'
            f'<span class="evidence-document">'
            f"{html.escape(str(metadata.get('document_id')))}</span>"
            "</div>"
            f'<div class="evidence-details">{html.escape(details)}</div>'
            f'<p class="evidence-snippet">{html.escape(shorten(item["text"]))}</p>'
            "</div>"
        )

    return (
        '<p class="evidence-question">Evidence for: '
        f"<strong>{html.escape(question)}</strong></p>"
        + "".join(cards)
    )


with gr.Blocks(title="Document Intelligence Agent") as demo:
    gr.Markdown(
        "# Document Intelligence Agent\n"
        "Add documents, then ask questions about them. "
        "Every answer comes with the evidence it was based on."
    )

    # Per-session state.
    collection_state = gr.State(None)  # vector store collection name
    sources_state = gr.State({})  # Sources
    pending_question = gr.State("")  # question waiting for its answer

    with gr.Row():
        # Panel 1: sources.
        with gr.Column(
                scale=1,
                min_width=260,
                variant="panel",
                elem_classes="app-panel",
        ):
            gr.Markdown("### Sources")
            upload_button = gr.UploadButton(
                "+ Add sources",
                file_count="multiple",
                type="filepath",
                size="sm",
            )
            progress = gr.Markdown()

            # Re-drawn every time the sources change: one row per document,
            # with a checkbox to search it and a button to remove it.
            @gr.render(inputs=sources_state)
            def render_sources(sources: Sources) -> None:
                if not sources:
                    gr.Markdown("No sources yet.")
                    return

                busy = any(
                    info["status"] in ("queued", "indexing")
                    for info in sources.values()
                )

                for name, info in sources.items():
                    with gr.Row(equal_height=True, elem_classes="source-row"):
                        checkbox = gr.Checkbox(
                            label=name,
                            info=status_label(info),
                            value=info["selected"],
                            interactive=info["status"] == "ready" and not busy,
                            container=False,
                            scale=1,
                        )
                        remove_button = gr.Button(
                            "✕",
                            size="sm",
                            variant="secondary",
                            scale=0,
                            min_width=0,
                            interactive=not busy,
                            elem_classes="remove-source",
                        )

                    checkbox.input(
                        partial(set_source_selected, name),
                        inputs=[checkbox, sources_state],
                        outputs=[sources_state],
                    )
                    remove_button.click(
                        partial(remove_source, name),
                        inputs=[collection_state, sources_state],
                        outputs=[sources_state],
                    )

        # Panel 2: chat.
        with gr.Column(scale=2, variant="panel", elem_classes="app-panel"):
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(
                show_label=False,
                height=560,
                placeholder="Add sources on the left, then ask a question about them.",
            )

            with gr.Row():
                question_input = gr.Textbox(
                    show_label=False,
                    placeholder="Ask about your sources...",
                    interactive=False,
                    container=False,
                    scale=5,
                )
                ask_button = gr.Button(
                    "Ask",
                    variant="primary",
                    interactive=False,
                    scale=1,
                    min_width=80,
                )

        # Panel 3: evidence for the latest answer.
        with gr.Column(scale=2, variant="panel", elem_classes="app-panel"):
            gr.Markdown("### Evidence")
            evidence_output = gr.HTML(
                EVIDENCE_PLACEHOLDER,
                elem_classes="evidence-panel",
            )

    upload_button.upload(
        process_documents,
        inputs=[upload_button, collection_state, sources_state],
        outputs=[progress, collection_state, sources_state],
    )

    sources_state.change(
        set_question_enabled,
        inputs=[sources_state],
        outputs=[question_input, ask_button],
    )

    # Two steps, so the question shows up in the chat before the
    # (slow) answer is ready.
    gr.on(
        triggers=[ask_button.click, question_input.submit],
        fn=submit_question,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot, pending_question],
    ).then(
        answer_question,
        inputs=[pending_question, chatbot, collection_state, sources_state],
        outputs=[chatbot, evidence_output],
    )


if __name__ == "__main__":
    demo.launch(css_paths=STYLESHEET)
