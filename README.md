# Document Intelligence Agent

An agentic document intelligence system for understanding documents, retrieving relevant evidence, and answering questions with source-level provenance.

## Overview

The goal of this project is to build a document-grounded AI system that can answer questions from user-provided documents while identifying where the answer came from.

The system is designed to support documents such as:

- PDFs
- Scanned documents
- Images
- DOCX
- PPTX
- Other structured or unstructured document formats

The system should return not only an answer, but also relevant source information such as:

- Document name
- Page number
- Section
- Paragraph
- Table
- Source location / provenance

## Architecture

The current document intelligence pipeline follows a deterministic retrieval-augmented generation flow:

```text
                    Document
                       │
                       ▼
                ┌─────────────┐
                │   Docling   │
                └──────┬──────┘
                       │
                       ▼
              Structured Document
                       │
                       ▼
                ┌─────────────┐
                │   Chunking  │
                └──────┬──────┘
                       │
                       ▼
             Provenance-aware Chunks
                       │
                       ▼
                ┌─────────────┐
                │  Embeddings │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │  ChromaDB   │
                └──────┬──────┘
                       │
                       ▼
                  Retrieval
                       │
                       ▼
                  Reranking
                       │
                       ▼
                Relevant Evidence
                       │
                       ▼
                     RAG
                       │
                       ▼
              Grounded Answer
                       │
                       ▼
             Answer + Provenance
````

The components are intentionally separated so that each part can be understood, tested, and replaced independently.

An orchestration layer may be introduced later if the system develops requirements such as branching workflows, iterative retrieval, query rewriting, tool usage, stateful execution, or human-in-the-loop interactions. No orchestration framework is required by the current deterministic pipeline.

## Core Components

### Document Ingestion

Docling processes supported documents and converts them into a structured document representation while preserving available document structure and provenance.

### Chunking

The structured document is converted into retrieval-ready, provenance-aware chunks.

Chunks retain information such as:

* Document
* Page
* Section
* Item type
* Source location
* Bounding box
* Character span

Tables are represented as retrieval text while retaining their source provenance.

### Embeddings

Document chunks are converted into vector representations.

The same embedding model is used to represent user queries so that semantic similarity can be measured between queries and document chunks.

### Vector Store

ChromaDB stores document embeddings together with the metadata required to retrieve and trace source evidence.

### Retrieval

A user query is embedded and used to retrieve potentially relevant document chunks from the vector store.

### Reranking

Retrieved candidates are reranked using a cross-encoder so that the most relevant evidence can be selected for answer generation.

### RAG

The highest-ranked evidence is provided to the language model as context.

The generation step is instructed to:

* Use only the supplied document evidence
* Avoid unsupported information
* State when the available evidence is insufficient
* Return source information associated with the evidence

### Provenance

Provenance is a core design requirement.

The system preserves source information through the retrieval pipeline so that an answer can be traced back to the originating document content.

The intended provenance information includes:

* Document
* Page
* Section
* Item type
* Bounding box
* Character span

This provides the foundation for a future evidence viewer that can identify the relevant region of the original document.

## Design Principles

### Separation of Concerns

Each component has a focused responsibility.

```text
Ingestion      → document processing
Chunking       → retrieval-ready representation
Embeddings     → vector representation
Vector Store   → vector persistence and search
Retrieval      → candidate evidence selection
Reranking      → relevance refinement
RAG            → grounded answer generation
```

Components should not depend unnecessarily on presentation or orchestration concerns.

### Technology Follows Requirements

Libraries and frameworks are introduced when they solve an actual system requirement.

The architecture therefore does not require an agent framework merely because the project is described as agentic.

If future requirements introduce complex workflow orchestration, an orchestration framework can be evaluated at that point.

### Provenance First

Retrieval quality alone is not sufficient.

The system must retain enough information to explain where retrieved evidence originated in the source document.

### Independently Testable Components

Each component should be independently understandable and testable before being composed into the larger system.

## Technology Stack

| Component           | Technology                            |
| ------------------- | ------------------------------------- |
| Document Processing | Docling                               |
| Chunking            | Custom provenance-aware chunking      |
| Embeddings          | Sentence Transformers                 |
| Vector Database     | ChromaDB                              |
| Retrieval           | Custom retrieval component            |
| Reranking           | Sentence Transformers CrossEncoder    |
| RAG                 | OpenAI Responses API                  |
| Language            | Python                                |
| UI / API            | Planned                               |
| Orchestration       | To be evaluated based on requirements |

## Project Structure

```text
src/
└── document_intelligence_agent/
    ├── ingestion/
    │   └── loader.py
    ├── chunking/
    │   └── chunker.py
    ├── embeddings/
    │   └── embedder.py
    ├── vector_store/
    │   └── chroma_store.py
    ├── retrieval/
    │   └── retriever.py
    ├── reranking/
    │   └── reranker.py
    └── rag/
        ├── generator.py
        └── pipeline.py
```

## Testing

The project uses `pytest` for component and integration testing.

Tests cover the document intelligence pipeline at multiple levels, including:

* Document ingestion
* Chunking
* Embeddings
* Vector storage
* Retrieval
* Reranking
* RAG generation
* End-to-end RAG execution

Test documents are not included in the repository.

Place local test documents in:

```text
tests/fixtures/
```

## Development Philosophy

The system is developed incrementally, with each component being implemented and tested before being composed with the next part of the pipeline.

The architecture is intentionally designed so that individual technologies can be evaluated and replaced as requirements and evaluation results evolve.

