from pydantic import BaseModel, Field


class PagePayload(BaseModel):
    url: str
    title: str = ""
    text: str = Field(default="", description="Raw page text from document.body.innerText")
    html: str = Field(default="", description="Optional page HTML for better section-aware chunking")


class ProcessResponse(BaseModel):
    page_id: str
    chunk_count: int
    cleaned_characters: int
    sections: list[str]


class AskRequest(BaseModel):
    page_id: str
    question: str
    top_k: int = 4


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    page_id: str


class SaveRequest(PagePayload):
    format: str = Field(default="markdown", pattern="^(markdown|json)$")


class SaveResponse(BaseModel):
    output: str
    format: str
