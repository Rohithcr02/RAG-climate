# RAG for Climate Challenges

Search and query climate research documents using hybrid retrieval + LLM-generated answers with source citations.

## What it does

1. You feed it PDFs (climate reports, policy docs, technical papers)
2. It chunks and indexes them in a local ChromaDB vector store
3. You ask questions in a Streamlit UI
4. It retrieves the best chunks using vector search + BM25, fuses the results, and generates a cited answer via Groq (Llama 3.3 70B)

## Project structure

```
├── app.py                 # Streamlit UI
├── config.py              # Prompts, model settings, constants
├── llm.py                 # Groq client + answer generation
├── html_renderer.py       # HTML/CSS/JS for cited answers
├── ingest.py              # PDF ingestion into ChromaDB
├── retrieve.py            # Hybrid search (vector + BM25 + RRF)
├── eval/
│   ├── generate_test_set.py   # Generate golden Q&A test set via LLM
│   ├── metrics.py             # Custom citation accuracy metrics
│   ├── run_eval.py            # RAGAS + custom eval runner
│   └── report.py              # Print/export eval results
├── requirements.txt
├── .env.example
└── chroma_db/             # Local vector store (auto-generated)
```

## Setup

**Requirements:** Python 3.8+, a [Groq API key](https://console.groq.com/)

```bash
# Clone
git clone https://github.com/asaikiranb/RAG-climate.git
cd RAG-climate

# Install
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

## Usage

### 1. Ingest your PDFs

Put PDF files in a `data/` folder, then run:

```bash
python ingest.py
```

This extracts text, chunks it (1000 tokens, 200 overlap), generates embeddings, and stores everything in local ChromaDB.

### 2. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Type a question or click one of the example queries.

### 3. Run evals (optional)

Evaluate retrieval quality and answer accuracy:

```bash
# Generate a golden test set (LLM writes ground truth from your docs)
python -m eval.generate_test_set

# Run full evaluation (RAGAS metrics + citation accuracy)
python -m eval.run_eval

# View results
python -m eval.report --markdown
```

Eval metrics:
- **RAGAS**: faithfulness, answer relevancy, context precision, context recall
- **Custom**: citation validity, citation coverage, source grounding

## How retrieval works

```
Query
  ├── Vector search (semantic, via ChromaDB)
  ├── BM25 search (keyword, via rank-bm25)
  └── Reciprocal Rank Fusion (merges both)
        → Top 5 chunks → LLM generates cited answer
```

## Configuration

All settings live in `config.py`:
- `LLM_MODEL` : which Groq model to use (default: `llama-3.3-70b-versatile`)
- `LLM_TEMPERATURE`, `LLM_MAX_TOKENS` : generation params
- `SYSTEM_PROMPT` : controls answer style and citation behavior
- `EXAMPLE_QUERIES` : the example buttons shown in the UI

Chunk size and overlap can be changed in `ingest.py` when calling `chunk_text()`.

## Costs

Groq free tier: 14,400 requests/day. No paid APIs required.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY not found` | Create `.env` file, paste your key |
| `No PDF files found` | Put PDFs in `data/` folder |
| `Could not connect to collection` | Run `python ingest.py` first |
| Empty results | Try broader queries, check ingestion output |
