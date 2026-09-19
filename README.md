# Document Intelligence Agent

An agentic document intelligence system for understanding documents, retrieving relevant evidence, and answering questions with source-level provenance.

## Overview

The goal of this project is to build a document-grounded AI agent that can answer questions from user-provided documents while identifying where the answer came from.

Supported documents will eventually include PDFs, scanned documents, images, DOCX, PPTX, and other structured or unstructured formats.

The system will return not only an answer, but also relevant source information such as:

- Document name
- Page number
- Section
- Paragraph
- Table
- Source location/provenance

## Architecture

```text
                    Document
                       │
                       ▼
                 ┌───────────┐
                 │  Docling  │
                 └─────┬─────┘
                       │
                       ▼
              Structured Document
                       │
                       ▼
              ┌─────────────────┐
              │    Chunking     │
              └────────┬────────┘
                       │
                       ▼
             Provenance-aware Chunks
                       │
                       ▼
              ┌─────────────────┐
              │   Embeddings    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Vector DB     │
              │    ChromaDB     │
              └────────┬────────┘
                       │
                       ▼
                  Retrieval
                       │
                       ▼
                  Reranking
                       │
                       ▼
                Best Evidence
                       │
                       ▼
               ┌──────────────┐
               │  LangGraph   │
               │    Agent     │
               └──────┬───────┘
                      │
                      ▼
             Grounded Answer
                      │
                      ▼
             Answer + Provenance
````

## Development Roadmap

* [x] Phase 0 — Foundation
* [ ] Phase 1 — Ingestion
* [ ] Phase 2 — Chunking
* [ ] Phase 3 — Embeddings
* [ ] Phase 4 — Vector Database
* [ ] Phase 5 — Retrieval
* [ ] Phase 6 — Reranking
* [ ] Phase 7 — RAG
* [ ] Phase 8 — LangGraph Agent
* [ ] Phase 9 — UI/API
* [ ] Phase 10 — Evaluation

## Tech Stack

| Component           | Technology            |
| ------------------- | --------------------- |
| Document Processing | Docling               |
| Chunking            | Docling HybridChunker |
| Embeddings          | TBD                   |
| Vector Database     | ChromaDB              |
| Retrieval           | TBD                   |
| Reranking           | TBD                   |
| Agent Orchestration | LangGraph             |
| API                 | TBD                   |
| UI                  | TBD                   |
| Language            | Python                |

## Project Status

🚧 **Early development — Phase 0**

This project is being developed incrementally, with each phase focusing on understanding, implementing, testing, and evaluating a specific part of the document intelligence pipeline.

## Test Documents

Test documents are not included in this repository.

Place your own test documents in:

```text
tests/fixtures/

