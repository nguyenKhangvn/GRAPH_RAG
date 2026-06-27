<!-- generated-by: gsd-doc-writer -->
# GraphRAG Tourism Chatbot

A Graph Retrieval-Augmented Generation (GraphRAG) chatbot that answers tourism questions about Gia Lai and Quy Nhon (Binh Dinh), Vietnam. It combines a Neo4j knowledge graph with LLM-powered generation to provide accurate, grounded answers with map visualization support.

## Architecture

```
User Query (Vietnamese)
    -> Intent Analysis (LLM)
    -> Knowledge Graph Retrieval (Neo4j + vector/fulltext search)
    -> Graph Traversal (1-hop, multi-hop)
    -> LLM Answer Generation
    -> Structured Response + Map Display
```

The system follows a Clean Architecture pattern with clear separation:

- **`graph_rag/`** -- Core RAG pipeline, modules, services, and configuration
- **`api.py`** -- FastAPI backend (REST + SSE streaming)
- **`FE/`** -- React + Vite frontend
- **`data_processing/`** -- Crawlers and data ingestion scripts
- **`data_v1/`, `data_v2/`** -- Raw and enriched tourism datasets
- **`data_neo4j_v1/`, `data_neo4j_v2/`, `data_neo4j_v3/`** -- Neo4j import data
- **`rag_baseline/`** -- Baseline RAG implementation for benchmarking
- **`tools/`** -- Audit and benchmarking utilities

## Prerequisites

- Python >= 3.9
- Neo4j >= 5.18 (with APOC plugin recommended)
- Node.js >= 18 (for the frontend)
- At least one LLM API key (Gemini, OpenAI, Groq, xAI, DeepSeek, or MiMo)

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd craw

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install Python dependencies
pip install -r graph_rag/requirements.txt

# 4. Create the .env file with your credentials
#    Required keys depend on which LLM provider you use.
#    See "Configuration" below for the full list.

# 5. (Optional) Set up Neo4j indexes
python graph_rag/setup_indexes.py

# 6. Install frontend dependencies
cd FE
npm install
cd ..
```

## Configuration

Create a `.env` file in the project root (or inside `graph_rag/`) with the following variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `NEO4J_URI` | Yes | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | Yes | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | -- | Neo4j password |
| `GEMINI_API_KEY` | One LLM key required | -- | Google Gemini API key |
| `OPENAI_API_KEY` | Optional | -- | OpenAI API key |
| `GROQ_API_KEY` | Optional | -- | Groq API key |
| `XAI_API_KEY` | Optional | -- | xAI (Grok) API key |
| `DEEPSEEK_API_KEY` | Optional | -- | DeepSeek API key |
| `MIMO_API_KEY` | Optional | -- | MiMo API key |
| `LLM_MODEL_NAME` | No | `deepseek-chat` | Default LLM model |
| `PIPELINE_LLM_MODEL_NAME` | No | falls back to `LLM_MODEL_NAME` | Model used in the RAG pipeline |
| `EMBEDDING_MODEL_NAME` | No | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Sentence embedding model |
| `APP_ENV` | No | `dev` | Environment name (`dev`, `staging`, `production`) |
| `CORS_ORIGINS` | No | `http://localhost:5173,http://localhost:8000` | Comma-separated allowed CORS origins |

## Quick Start

```bash
# 1. Start the backend API
python api.py
# The API runs at http://localhost:8000

# 2. In another terminal, start the frontend
cd FE
npm run dev
# The frontend runs at http://localhost:5173

# 3. Or run the interactive CLI directly
python graph_rag/main.py
```

## Usage

### REST API

Send a chat query:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Khach san Bien Xanh o dau?", "chat_history": []}'
```

### Interactive CLI

```bash
python graph_rag/main.py
# Then type questions directly at the prompt
```

### Example Queries

| Query | Intent |
|---|---|
| "Khach san Bien Xanh o dau?" | Entity lookup |
| "Cho toi danh sach dac san Gia Lai" | Discovery list |
| "Lap lich trinh 2 ngay o Quy Nhon" | Tour planning |
| "So sanh nha hang A va B" | Comparison |
| "Tu Pleiku den Quy Nhon bao xa?" | Distance |

## Project Structure

```
graph_rag/
  core/               # Abstract interfaces, state definitions, constants
  config/             # Centralized config loader (env vars + JSON)
  services/           # Singleton services: Neo4j, LLM providers, directions
  modules/
    query_analysis/   # Intent classification and query understanding
    query_planning/   # Intent routing and sub-query planning
    retrieval/        # Seed retrieval, agentic retrieval, vector/fulltext search, Cypher generation
    graph/            # Graph traversal logic
    context/          # Structured context builder
    generation/       # LLM answer generation
    validation/       # Completeness gate
    pipeline_support/ # Location grounding, distance intent, admin region mapping
    tour_plan/        # Tour route optimization
  pipeline/           # RAGPipeline orchestrator
  prompts/            # Prompt templates for LLM calls
  utils/              # Text normalization and helpers
  tests/              # Test suite
api.py                # FastAPI backend (REST)
api_sse.py            # FastAPI backend (SSE streaming)
FE/                   # React + Vite frontend
data_processing/      # Data ingestion scripts (accommodation, dishes, events, etc.)
tools/                # Audit and benchmarking scripts
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

<!-- VERIFY: License information not found in repository. Confirm license type. -->
No LICENSE file detected. All rights reserved by the project owner unless otherwise stated.
