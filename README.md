# Page RAG MVP

Chrome extension plus FastAPI backend for grounded Q&A on the current webpage using local RAG, plus a separate "Save key points" flow for structured notes.

## What it does

- `Live Q&A`: extracts the current page, cleans and chunks it, builds a per-page in-memory FAISS index, retrieves top chunks, and answers from only those chunks.
- `Save key points`: sends cleaned page content through a separate notes prompt to generate markdown or JSON study notes.

## Project layout

- `backend/`: FastAPI app with `/process`, `/ask`, and `/save`
- `extension/`: Manifest V3 Chrome extension with popup UI and content extraction

## Backend setup

1. Install dependencies:

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Make sure Ollama is running and the models are available:

   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

3. Start the API:

   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

Optional environment variables:

- `OLLAMA_BASE_URL`
- `OLLAMA_LLM_MODEL`
- `OLLAMA_EMBED_MODEL`

## Extension setup

1. Open `chrome://extensions`
2. Enable Developer mode
3. Click `Load unpacked`
4. Select the `extension/` folder

The popup defaults to `http://127.0.0.1:8000`.

## API summary

- `POST /process`: build an in-memory FAISS index for a page
- `POST /ask`: retrieve the top chunks and answer using only those chunks
- `POST /save`: generate structured notes from cleaned page text

## Notes

- The backend keeps embeddings in memory only, keyed by a stable page hash.
- If retrieval finds no relevant chunk, `/ask` returns `Not in page`.
- HTML-aware chunking uses headings and content blocks when HTML is provided, and falls back to text heuristics otherwise.
