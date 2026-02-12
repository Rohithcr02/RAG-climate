# Retrieval Augmented Generation for Climate Challenges

A Retrieval-Augmented Generation (RAG) application designed for HVAC mechanics to search and query technical documentation using AI.

## Features

- **Hybrid Search**: Combines vector similarity (semantic search) with BM25 (keyword search) using Reciprocal Rank Fusion
- **Smart Chunking**: Processes PDFs with token-aware chunking (1000 tokens per chunk, 200 token overlap)
- **Brand Filtering**: Filter results by equipment manufacturer
- **Source Citations**: Always shows document name and page number for answers
- **Safety-Aware**: LLM includes safety warnings where relevant
- **Cloud Storage**: Uses ChromaDB cloud for scalable vector storage

## Tech Stack

- **PDF Processing**: PyMuPDF
- **Embeddings**: all-MiniLM-L6-v2 via sentence-transformers
- **Vector Database**: ChromaDB Cloud
- **Keyword Search**: BM25 (rank-bm25)
- **LLM**: Groq API (Llama 3 70B)
- **UI**: Streamlit

## Project Structure

```
.
├── app.py              # Streamlit UI application
├── ingest.py           # PDF ingestion and embedding pipeline
├── retrieve.py         # Hybrid search and retrieval logic
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (create this)
├── .env.example        # Template for environment variables
├── data/               # Place PDF files here for ingestion
└── README.md           # This file
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Groq API key ([get one here](https://console.groq.com/))
- ChromaDB Cloud account ([sign up here](https://www.trychroma.com/))

### 2. Installation

```bash
# Clone or download the repository
cd RAG-climate

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# Groq API Key
GROQ_API_KEY=your_groq_api_key_here

# ChromaDB Cloud Configuration
CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_HOST=your_chroma_host_here
CHROMA_COLLECTION_NAME=hvac_documents
```

**How to get ChromaDB Cloud credentials:**
1. Sign up at [trychroma.com](https://www.trychroma.com/)
2. Create a new database
3. Copy the API key and host URL
4. Paste them into your `.env` file

### 4. Add PDF Documents

Place your HVAC technical manuals (PDF files) in the `data/` folder:

```bash
# Example structure:
data/
  ├── Carrier_ServiceManual.pdf
  ├── Trane_TroubleshootingGuide.pdf
  └── Lennox_InstallationManual.pdf
```

### 5. Ingest Documents

Run the ingestion script to process PDFs and store them in ChromaDB:

```bash
python ingest.py
```

This will:
- Extract text from all PDFs in the `data/` folder
- Split text into 1000-token chunks with 200-token overlap
- Generate embeddings using all-MiniLM-L6-v2
- Store chunks with metadata (filename, page number) in ChromaDB Cloud

**Note**: The first run will download the embedding model (~80MB).

### 6. Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Search Interface

1. **Enter a Question**: Type your technical question in the search bar
   - Example: "How do I diagnose a refrigerant leak?"

2. **Filter by Brand** (Optional): Use the sidebar dropdown to filter results by manufacturer
   - Helps narrow down results to specific equipment

3. **Adjust Number of Sources**: Use the slider to control how many document chunks are retrieved (3-10)

4. **View Results**:
   - **Answer**: AI-generated response with source citations
   - **Sources**: List of documents and page numbers used
   - **Retrieved Context**: Expandable section showing the actual text chunks

### Example Questions

- "How do I troubleshoot a compressor failure?"
- "What are the steps for refrigerant leak detection?"
- "How to check capacitor functionality?"
- "What safety precautions are needed when handling refrigerants?"
- "How do I diagnose a frozen evaporator coil?"

## How It Works

### 1. Document Ingestion (`ingest.py`)

```
PDF Files → PyMuPDF → Text Extraction → Token-Based Chunking
    ↓
Sentence Transformer (all-MiniLM-L6-v2) → Embeddings
    ↓
ChromaDB Cloud Storage (with metadata)
```

### 2. Retrieval (`retrieve.py`)

```
User Query
    ↓
┌─────────────────┬─────────────────┐
│  Vector Search  │   BM25 Search   │
│   (Semantic)    │   (Keyword)     │
└────────┬────────┴────────┬────────┘
         │                 │
         └────────┬────────┘
                  ↓
    Reciprocal Rank Fusion (RRF)
                  ↓
          Top 5 Chunks
```

**Reciprocal Rank Fusion** merges results using:
```
RRF(document) = Σ(1 / (k + rank))
```
where k=60 and rank is the position in each search method's results.

### 3. Answer Generation (`app.py`)

```
Retrieved Chunks → Context Assembly
    ↓
Groq API (Llama 3 70B) + System Prompt
    ↓
Answer with Citations + Safety Warnings
```

## System Prompt

The LLM uses this system prompt:

> You are an HVAC technical assistant for AC mechanics. Answer using ONLY the provided context. Cite source document and page number. If the context doesn't have the answer, say so. Include safety warnings where relevant.

## Testing Retrieval

Test the retrieval system independently:

```bash
python retrieve.py
```

This will:
- Show available brands in your collection
- Run a sample query
- Display search results with scores

## Troubleshooting

### "No PDF files found in ./data"
- Make sure you've placed PDF files in the `data/` folder
- Check that files have `.pdf` extension

### "Could not connect to collection"
- Verify your ChromaDB credentials in `.env`
- Check that `CHROMA_HOST` and `CHROMA_API_KEY` are correct
- Ensure you've run `ingest.py` to create the collection

### "GROQ_API_KEY not found"
- Make sure you've created the `.env` file
- Verify the API key is correct
- Check there are no extra spaces or quotes

### Empty search results
- Verify documents were ingested successfully (check `ingest.py` output)
- Try broader search terms
- Remove brand filter to search all documents

### Model download issues
- The first run downloads the embedding model (~80MB)
- Ensure you have internet connection
- Check disk space

## Performance Tips

1. **Batch Ingestion**: The ingestion script processes documents in batches of 100 chunks for efficiency

2. **Caching**: The retriever caches the BM25 index to avoid reloading on every search

3. **Chunk Size**: 1000 tokens balances context quality and retrieval precision
   - Smaller chunks = more precise but may lose context
   - Larger chunks = more context but less precise matching

4. **Top-K Results**: Default is 5 chunks, increase for more comprehensive answers

## Customization

### Change Chunk Size

Edit `ingest.py`:
```python
chunks = self.chunk_text(page_data['text'], chunk_size=1000, overlap=200)
```

### Change LLM Model

Edit `app.py`:
```python
model="llama3-70b-8192",  # Change to other Groq models
```

### Modify System Prompt

Edit the `SYSTEM_PROMPT` variable in `app.py` to change how the LLM responds.

### Change Number of Retrieved Chunks

Edit `retrieve.py` in `vector_search()` and `bm25_search()`:
```python
top_k=20  # Increase for more candidate chunks before RRF
```

## API Costs

- **Groq**: Free tier includes 14,400 requests/day for Llama 3 70B
- **ChromaDB Cloud**: Free tier includes 50k documents (check current limits)

## Security Notes

- Never commit your `.env` file to version control
- Keep API keys confidential
- The `.env` file is already in `.gitignore`

## License

This project is for educational and internal use. Ensure you have proper rights to the PDF documents you ingest.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify all credentials are correct
4. Ensure documents are properly formatted PDFs

## Future Enhancements

Potential improvements:
- [ ] Add conversation history for follow-up questions
- [ ] Support for image extraction from PDFs
- [ ] Multi-turn dialogue capabilities
- [ ] Export answers to PDF reports
- [ ] Admin panel for document management
- [ ] User feedback and answer rating system
