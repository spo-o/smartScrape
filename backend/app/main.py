from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .rag import STORE, answer_question, create_page_index, generate_notes
from .schemas import AskRequest, AskResponse, PagePayload, ProcessResponse, SaveRequest, SaveResponse


app = FastAPI(title="Page RAG Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
async def process_page(payload: PagePayload) -> ProcessResponse:
    try:
        page_index = await create_page_index(
            url=payload.url,
            title=payload.title,
            text=payload.text,
            html=payload.html,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to process page with Ollama: {exc}") from exc

    STORE.put(page_index)
    return ProcessResponse(
        page_id=page_index.page_id,
        chunk_count=len(page_index.chunks),
        cleaned_characters=len(page_index.cleaned_text),
        sections=sorted({chunk.section for chunk in page_index.chunks})[:20],
    )


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    page_index = STORE.get(payload.page_id)
    if not page_index:
        raise HTTPException(status_code=404, detail="Page not processed or expired")

    try:
        answer, sources = await answer_question(page_index, payload.question, top_k=payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to answer with Ollama: {exc}") from exc

    return AskResponse(answer=answer, sources=sources, page_id=payload.page_id)


@app.post("/save", response_model=SaveResponse)
async def save_notes(payload: SaveRequest) -> SaveResponse:
    try:
        page_index = await create_page_index(
            url=payload.url,
            title=payload.title,
            text=payload.text,
            html=payload.html,
        )
        STORE.put(page_index)
        output = await generate_notes(
            title=payload.title or payload.url,
            text=page_index.cleaned_text,
            output_format=payload.format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to generate notes with Ollama: {exc}") from exc

    return SaveResponse(output=output.strip(), format=payload.format)
