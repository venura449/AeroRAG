# AeroRAG: Privacy-Preserving Two-Stage Clinical RAG for Asthma Management

AeroRAG is a fully local, offline Retrieval-Augmented Generation (RAG) framework designed to query complex medical guidelines without compromising data privacy. Grounded strictly in the Global Initiative for Asthma (GINA) 2025 Strategy Report, AeroRAG utilizes a custom Two-Stage Retrieval Architecture to provide highly accurate, verifiable, and hallucination-resistant clinical decision support on consumer-grade hardware (4GB VRAM).

## Key Features

* **100% Offline & Private:** Zero cloud dependencies. No patient queries or medical documents are ever sent to external APIs (e.g., OpenAI or Google), ensuring absolute data sovereignty and compliance with healthcare privacy paradigms.
* **Two-Stage Retrieval (Advanced RAG):** * *Stage 1:* Dense passage retrieval utilizing `all-MiniLM-L6-v2` embeddings and ChromaDB to extract a broad set of candidate document chunks.
  * *Stage 2:* A local Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) evaluates query-context pairs, re-ranking and selecting only the top 3 highest-scoring contexts.
* **Hardware Democratization:** Optimized to run entirely on low-end consumer hardware (tested on an NVIDIA RTX 2050 4GB laptop GPU) using partial GPU/CPU layer offloading via `llama.cpp`.
* **Hallucination Guardrails:** Strict prompt engineering forces a hard evidence boundary. If an answer cannot be mathematically or contextually derived from the retrieved chunks, the system safely refuses rather than hallucinating dangerous medical advice.
* **Traceable Citations:** All generated insights are dynamically appended with exact, verifiable source file paths and page-level citation tags.
* **Token-Optimized Streaming:** Leverages server-sent events (SSE) to stream output tokens instantly, significantly minimizing user-perceived latency.

## System Architecture

```text
[User Query] 
     │
     ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Dense Semantic Retrieval                      │
│ - Embed query via all-MiniLM-L6-v2                     │
│ - Fetch Top 10 candidate chunks from ChromaDB          │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Cross-Encoder Re-Ranking                      │
│ - Evaluate [Query, Chunk] pairs using MS-MARCO         │
│ - Mathematically score semantic relevance              │
│ - Filter down to top 3 highest-scoring contexts        │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 3: Constrained Generation (Local Inference)      │
│ - Inject context + prompt guardrails into Gemma-4B     │
│ - Stream token output with exact page citations        │
└────────────────────────────────────────────────────────┘
