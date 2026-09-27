# System Architecture

## Overview

Document Intelligence Agent is designed as a document-grounded AI system.

The system processes a document, converts it into structured and provenance-aware content, indexes that content for retrieval, and uses retrieved evidence to generate grounded answers.

The architecture separates document processing, retrieval, answer generation, application concerns, and future orchestration concerns.

## High-Level Architecture

```text
                         Presentation
                    ┌──────────────────┐
                    │      UI / API    │
                    └────────┬─────────┘
                             │
                             ▼
                       Application
                             │
                             ▼
                    Orchestration Layer
                    (when required)
                             │
                             ▼
                    Domain Components
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
      Ingestion          Retrieval            RAG
          │                  │                  │
          ▼                  │                  │
      Chunking               │                  │
          │                  │                  │
          ▼                  │                  │
     Embeddings              │                  │
          │                  │                  │
          ▼                  │                  │
     Vector Store ───────────┘                  │
                             │                  │
                             └──── Reranking ───┘
                                                │
                                                ▼
                                      Grounded Answer
                                                │
                                                ▼
                                      Answer + Provenance
````

## Architectural Layers

### 1. Presentation

The presentation layer will provide user-facing interfaces such as a web UI or API.

Its responsibilities include:

* Document upload
* User question input
* Answer display
* Source display
* Evidence visualization

Presentation code should not contain document processing, retrieval, or generation logic.

### 2. Application

The application layer will translate external requests into application-level use cases.

Potential responsibilities include:

* Accepting a question and document scope
* Coordinating a document question-answering request
* Preparing responses for the presentation layer
* Managing application-level request and response models

The application layer should depend on domain capabilities rather than UI implementation details.

### 3. Orchestration

The orchestration layer represents workflow coordination across multiple components.

The current retrieval and RAG workflow is deterministic and does not require an orchestration framework.

An orchestration layer becomes useful if the system requires capabilities such as:

* Branching workflows
* Iterative retrieval
* Query rewriting
* Multiple retrieval strategies
* Tool usage
* Stateful execution
* Human-in-the-loop decisions
* Retry or recovery workflows
* Durable execution

An orchestration framework will be evaluated only when such requirements exist.

The orchestration layer should coordinate domain components rather than absorb their responsibilities.

### 4. Domain Components

The domain components contain the core document intelligence capabilities.

```text
Ingestion
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Store
    ↓
Retrieval
    ↓
Reranking
    ↓
RAG
```

Each component should have a focused responsibility and remain independently testable.

## Components

### Document Ingestion

Docling processes supported documents and converts them into a structured document representation.

The ingestion layer should preserve document structure and provenance wherever available.

### Chunking

The structured document is converted into retrieval-ready chunks.

Chunks preserve useful contextual and provenance information such as:

* Document
* Page
* Section
* Item type
* Source location
* Bounding box
* Character span

Tables are converted into deterministic retrieval text while retaining their source provenance.

### Embeddings

Each chunk is converted into a vector representation.

The same embedding space is used to represent user queries so that semantic similarity can be measured between queries and document chunks.

### Vector Store

Embeddings and their associated metadata are stored in a vector database.

The current implementation uses ChromaDB.

The vector store is an infrastructure component and should not determine the behavior of higher-level retrieval or answer-generation logic.

### Retrieval

A user query is converted into an embedding and used to retrieve potentially relevant document chunks.

The retrieval component should focus on selecting candidate evidence and should not contain answer-generation or presentation logic.

### Reranking

Retrieved candidates are reranked to identify the evidence most relevant to the user's question.

The reranker operates on retrieved candidates and does not own document ingestion or answer generation.

### RAG

The RAG layer uses selected evidence to generate a grounded answer.

Its responsibilities include:

* Constructing the evidence context
* Generating a grounded answer
* Returning source information associated with the evidence

The RAG layer should not depend directly on UI concerns.

## Provenance

Provenance is a core architectural requirement.

The system should retain enough metadata to identify where retrieved evidence originated in the source document.

Relevant provenance may include:

* Document identifier
* Page number
* Section
* Item type
* Bounding box
* Character span

The provenance should survive the pipeline from document extraction through retrieval and answer generation.

This information provides the foundation for a future evidence viewer capable of identifying the source region associated with an answer.

## Separation of Concerns

The architecture follows a single-responsibility approach.

```text
Presentation
    → User interaction

Application
    → Use cases and request/response coordination

Orchestration
    → Workflow and state coordination when required

Ingestion
    → Document processing

Chunking
    → Retrieval-ready document representation

Embeddings
    → Vector representation

Vector Store
    → Vector storage and similarity search

Retrieval
    → Candidate evidence selection

Reranking
    → Relevance refinement

RAG
    → Grounded answer generation
```

A component should not take ownership of responsibilities belonging to another layer merely for convenience.

## Dependency Direction

The intended dependency direction is:

```text
Presentation
      ↓
Application
      ↓
Orchestration
      ↓
Domain Components
      ↓
Infrastructure
```

Infrastructure implementations such as ChromaDB and model clients should remain replaceable behind focused component interfaces where practical.

Domain components should not depend on presentation concerns.

## Design Principles

### Technology Follows Requirements

A technology should be introduced because it solves a demonstrated requirement.

The architecture therefore does not require an orchestration framework for the current deterministic RAG pipeline.

### Independent Components

Each component should be independently understandable, testable, and replaceable.

### Provenance Preservation

Provenance should be preserved throughout the pipeline rather than reconstructed after answer generation.

### Replaceable Infrastructure

Model providers, vector databases, and other infrastructure should be replaceable without requiring changes to unrelated domain components.

### Incremental Architecture

The system should evolve by adding capabilities when requirements justify them rather than introducing infrastructure speculatively.

