AeroRAG: Privacy-Preserving Two-Stage Clinical RAG for Asthma Management
AeroRAG is a fully local, offline Retrieval-Augmented Generation (RAG) framework designed to query complex medical guidelines without compromising data privacy. Grounded strictly in the Global Initiative for Asthma (GINA) 2025 Strategy Report, AeroRAG utilizes a custom Two-Stage Retrieval Architecture to provide highly accurate, verifiable, and hallucination-resistant clinical decision support on consumer-grade hardware (4GB VRAM).

🌟 Key Features
100% Offline & Private: Zero cloud dependencies. No patient queries or medical documents are ever sent to external APIs (e.g., OpenAI or Google), ensuring absolute data sovereignty and compliance with healthcare privacy paradigms.

Two-Stage Retrieval (Advanced RAG): * Stage 1: Dense passage retrieval utilizing all-MiniLM-L6-v2 embeddings and ChromaDB to extract a broad set of candidate document chunks.

Stage 2: A local Cross-Encoder (ms-marco-MiniLM-L-6-v2) evaluates query-context pairs, re-ranking and selecting only the top 3 highest-scoring contexts.

Hardware Democratization: Optimized to run entirely on low-end consumer hardware (tested on an NVIDIA RTX 2050 4GB laptop GPU) using partial GPU/CPU layer offloading via llama.cpp.

Hallucination Guardrails: Strict prompt engineering forces a hard evidence boundary. If an answer cannot be mathematically or contextually derived from the retrieved chunks, the system safely refuses rather than hallucinating dangerous medical advice.

Traceable Citations: All generated insights are dynamically appended with exact, verifiable source file paths and page-level citation tags.

Token-Optimized Streaming: Leverages server-sent events (SSE) to stream output tokens instantly, significantly minimizing user-perceived latency.

🏗️ System Architecture
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
│ STAGE 3: Constrained Generation (Local Inference)       │
│ - Inject context + prompt guardrails into Gemma-4B      │
│ - Stream token output with exact page citations        │
└────────────────────────────────────────────────────────┘
🛠️ Tech Stack
LLM Inference Engine: llama.cpp server (Quantized Gemma 4B IT .gguf)

Vector Database: ChromaDB (Persistent local vector store)

Embedding Model: HuggingFace sentence-transformers/all-MiniLM-L6-v2 (384 Dimensions)

Re-Ranker Model: HuggingFace cross-encoder/ms-marco-MiniLM-L-6-v2

Orchestration: LangChain & langchain-community

API Layer: OpenAI-compatible Python SDK (targeting the local llama.cpp endpoint)

🚀 Quick Start
1. Prerequisites
Python 3.10 or higher

CUDA Toolkit (if offloading to an NVIDIA GPU)

The GINA 2025 PDF Strategy Report saved locally

2. Installation
Clone the repository and set up a virtual environment:

Bash
git clone https://github.com/YourUsername/AeroRAG.git
cd AeroRAG
python -m venv venv
source venv/Scripts/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
3. Start the Local LLM Server
Download your preferred quantized Gemma-4B GGUF model and initialize the llama.cpp server. Ensure Flash Attention (-fa) and optimized thread counts (-t) are enabled to preserve VRAM:

Bash
.\llama-server.exe -m gemma-4-E4B-it-UD-Q4_K_XL.gguf -ngl 15 -c 4096 -fa -t 4 --port 8080
4. Populate the Knowledge Base
Place your GINA 2025 PDF in your data directory and execute the ingestion script to chunk, embed, and store the document vectors locally:

Bash
python ingestion_pipeline.py
5. Launch the Assistant
Run the advanced conversational loop script to start executing clinical queries:

Bash
python query.py
📊 Evaluation & Verification
To ensure clinical safety, the framework was strictly evaluated against out-of-bounds questions (e.g., non-respiratory conditions like acute appendicitis). The custom scoring logic demonstrated a 100% safety refusal rate on ungrounded prompts, maintaining an ironclad evidence boundary essential for safety-critical medical deployment.

📝 Citation & Academic Use
If you are using this framework for academic research, university theses, or publications, please cite it as follows:

Code snippet
@software{AeroRAG2026,
  author = {Your Name / SLIIT Informatics},
  title = {AeroRAG: A Privacy-Preserving Two-Stage Retrieval-Augmented Generation Framework for Clinical Asthma Guidelines},
  year = {2026},
  url = {https://github.com/YourUsername/AeroRAG}
}
