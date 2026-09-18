
# System Architecture

## Overview

Document Intelligence Agent is designed as a document-grounded AI system.

The system processes a document, converts it into structured and provenance-aware content, indexes that content for retrieval, and uses the retrieved evidence to generate grounded answers.

## High-Level Flow

```text
Document
   │
   ▼
Docling
   │
   ▼
Structured Document
   │
   ▼
Chunking
   │
   ▼
Provenance-aware Chunks
   │
   ▼
Embeddings
   │
   ▼
Vector Database
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
LangGraph Agent
   │
   ▼
Grounded Answer
   │
   ▼
Answer + Provenance
````

## Components

### 1. Document Ingestion

Docling is responsible for processing supported documents and converting them into a structured representation.

The ingestion layer will preserve document structure and provenance wherever available.

### 2. Chunking

The structured document is converted into retrieval-ready chunks.

Chunks should preserve useful contextual information such as:

* Document
* Page
* Section
* Paragraph
* Table
* Source location

### 3. Embeddings

Each chunk is converted into a vector representation.

The same embedding space will be used to represent user queries so that semantic similarity can be measured between queries and document chunks.

### 4. Vector Database

Embeddings and their associated metadata are stored in a vector database.

The initial implementation will use ChromaDB.

### 5. Retrieval

When a user asks a question, the question is converted into an embedding and used to retrieve potentially relevant document chunks.

### 6. Reranking

Retrieved candidates are reranked to identify the evidence most relevant to the user's question.

### 7. RAG

The highest-quality evidence is provided to the language model as context.

The generated answer should remain grounded in the retrieved document evidence.

### 8. LangGraph Agent

LangGraph will orchestrate the document question-answering workflow.

The agent layer will be introduced after the core retrieval pipeline is working independently.

### 9. Provenance

The system should retain enough metadata to identify where retrieved evidence originated in the source document.

The final response should expose relevant provenance such as page, section, paragraph, or table.

## Design Principle

The system is built incrementally.

Each component should be independently understandable, testable, and replaceable before being connected to the next component.


