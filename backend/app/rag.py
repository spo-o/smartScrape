from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable

import faiss
import httpx
import numpy as np
from bs4 import BeautifulSoup, NavigableString, Tag


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
MIN_RELEVANCE_SCORE = 0.18


def stable_page_id(url: str, text: str) -> str:
    digest = hashlib.sha256(f"{url}\n{text[:5000]}".encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class Chunk:
    index: int
    section: str
    text: str

    @property
    def label(self) -> str:
        return f"{self.section} [chunk {self.index}]"


@dataclass
class PageIndex:
    page_id: str
    url: str
    title: str
    cleaned_text: str
    chunks: list[Chunk]
    index: faiss.IndexFlatIP
    embeddings: np.ndarray


class PageStore:
    def __init__(self) -> None:
        self._store: dict[str, PageIndex] = {}

    def put(self, page_index: PageIndex) -> None:
        self._store[page_index.page_id] = page_index

    def get(self, page_id: str) -> PageIndex | None:
        return self._store.get(page_id)


STORE = PageStore()


def clean_and_chunk(text: str, html: str) -> tuple[str, list[Chunk]]:
    if html.strip():
        sections = _sections_from_html(html)
    else:
        sections = _sections_from_text(text)

    chunks: list[Chunk] = []
    cleaned_parts: list[str] = []
    chunk_index = 0

    for section_name, raw_block in sections:
        normalized = _normalize_text(raw_block)
        if not normalized:
            continue
        cleaned_parts.append(f"{section_name}\n{normalized}")
        for piece in _split_large_block(normalized):
            chunks.append(Chunk(index=chunk_index, section=section_name, text=piece))
            chunk_index += 1

    if not chunks:
        fallback_text = _normalize_text(text)
        if fallback_text:
            cleaned_parts.append(fallback_text)
            for piece in _split_large_block(fallback_text):
                chunks.append(Chunk(index=chunk_index, section="Page", text=piece))
                chunk_index += 1

    return "\n\n".join(cleaned_parts), chunks


def _sections_from_html(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in ["script", "style", "noscript", "nav", "aside", "footer", "header", "form", "button", "svg"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    body = soup.body or soup
    sections: list[tuple[str, str]] = []
    current_heading = "Introduction"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        text = _normalize_text("\n".join(buffer))
        if text:
            sections.append((current_heading, text))
        buffer = []

    for element in body.descendants:
        if isinstance(element, NavigableString):
            continue
        if not isinstance(element, Tag):
            continue
        if element.name in {"h1", "h2", "h3"}:
            flush()
            heading_text = _normalize_text(element.get_text(" ", strip=True))
            if heading_text:
                current_heading = heading_text
            continue
        if element.name in {"p", "li", "blockquote"}:
            text = _normalize_text(element.get_text(" ", strip=True))
            if text:
                buffer.append(text)
            continue
        if element.name in {"pre", "code"}:
            code_text = _normalize_text(element.get_text("\n", strip=True))
            if code_text:
                buffer.append(f"Code:\n{code_text}")

    flush()
    return sections


def _sections_from_text(text: str) -> list[tuple[str, str]]:
    lines = [line.strip() for line in text.splitlines()]
    sections: list[tuple[str, str]] = []
    current_heading = "Page"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        block = _normalize_text("\n".join(buffer))
        if block:
            sections.append((current_heading, block))
        buffer = []

    for line in lines:
        if not line:
            if buffer and buffer[-1] != "":
                buffer.append("")
            continue
        if _looks_like_heading(line):
            flush()
            current_heading = line[:120]
            continue
        buffer.append(line)

    flush()
    return sections


def _looks_like_heading(line: str) -> bool:
    if len(line) > 80:
        return False
    words = line.split()
    if not words or len(words) > 12:
        return False
    return line.isupper() or line.endswith(":") or line == line.title()


def _normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_large_block(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(sentence) <= max_chars:
            current = sentence
            continue
        pieces.extend(_force_split(sentence, max_chars))
        current = ""

    if current:
        pieces.append(current)

    return _apply_overlap(pieces)


def _force_split(text: str, max_chars: int) -> list[str]:
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _apply_overlap(pieces: Iterable[str]) -> list[str]:
    pieces = list(pieces)
    if len(pieces) < 2:
        return pieces
    with_overlap: list[str] = [pieces[0]]
    for piece in pieces[1:]:
        prefix = with_overlap[-1][-CHUNK_OVERLAP:]
        with_overlap.append(f"{prefix}\n{piece}")
    return with_overlap


async def embed_texts(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for text in texts:
            payload = await _embed_one(client, text)
            vectors.append(payload)

    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    return matrix


async def _embed_one(client: httpx.AsyncClient, text: str) -> list[float]:
    try:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("embeddings"):
            return payload["embeddings"][0]
    except httpx.HTTPError:
        pass

    fallback = await client.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
    )
    fallback.raise_for_status()
    payload = fallback.json()
    return payload["embedding"]


async def create_page_index(url: str, title: str, text: str, html: str) -> PageIndex:
    cleaned_text, chunks = clean_and_chunk(text=text, html=html)
    if not chunks:
        raise ValueError("No usable page content extracted")

    embeddings = await embed_texts([chunk.text for chunk in chunks])
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return PageIndex(
        page_id=stable_page_id(url, cleaned_text or text),
        url=url,
        title=title,
        cleaned_text=cleaned_text,
        chunks=chunks,
        index=index,
        embeddings=embeddings,
    )


async def retrieve_chunks(page_index: PageIndex, question: str, top_k: int) -> list[tuple[Chunk, float]]:
    queries = expand_question_queries(page_index, question)
    query_embeddings = await embed_texts(queries)

    scored_results: dict[int, float] = {}
    search_k = min(max(top_k * 2, 6), len(page_index.chunks))

    for query_embedding in query_embeddings:
        scores, indices = page_index.index.search(np.array([query_embedding]), search_k)
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            score = float(score)
            if score < MIN_RELEVANCE_SCORE:
                continue
            best = scored_results.get(int(idx))
            if best is None or score > best:
                scored_results[int(idx)] = score

    ranked = sorted(scored_results.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [(page_index.chunks[idx], score) for idx, score in ranked]


def expand_question_queries(page_index: PageIndex, question: str) -> list[str]:
    queries = [question.strip()]
    normalized = re.sub(r"\s+", " ", question).strip()
    if not normalized:
        return queries

    subject = _subject_from_title(page_index.title or page_index.url)
    lower = normalized.lower()

    if subject and re.search(r"\b(he|she|they|his|her|their)\b", lower):
        replaced = re.sub(r"\bhe\b|\bshe\b|\bthey\b", subject, normalized, flags=re.IGNORECASE)
        replaced = re.sub(
            r"\bhis\b|\bher\b|\btheir\b",
            f"{subject}'s",
            replaced,
            flags=re.IGNORECASE,
        )
        queries.append(replaced)

    if "ipl" in lower and "team" in lower:
        queries.append(f"{subject} IPL team" if subject else normalized)
        queries.append(
            re.sub(r"\bdoes\s+", "", normalized, flags=re.IGNORECASE).replace("play for", "plays for")
        )
        if subject:
            queries.append(f"Which IPL team does {subject} play for?")
            queries.append(f"{subject} plays for which IPL team")

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in queries:
        cleaned = candidate.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _subject_from_title(title: str) -> str:
    title = re.sub(r"\s*[-|]\s*Wikipedia.*$", "", title).strip()
    title = re.sub(r"\s*-\s*.*$", "", title).strip()
    if not title:
        return ""
    words = title.split()
    if len(words) > 6:
        return ""
    return title


async def answer_question(page_index: PageIndex, question: str, top_k: int = 4) -> tuple[str, list[str]]:
    retrieved = await retrieve_chunks(page_index, question, top_k)
    if not retrieved:
        return "Not in page", []

    context_blocks = [f"{chunk.label}\n{chunk.text}" for chunk, _ in retrieved]
    prompt = (
        "You answer questions using only the supplied webpage context.\n"
        "Rules:\n"
        "- If the answer is not fully supported by the context, reply exactly: Not in page\n"
        "- Be concise.\n"
        "- Do not use outside knowledge.\n\n"
        f"Question: {question}\n\n"
        "Context:\n"
        + "\n\n".join(context_blocks)
    )

    answer = await generate_text(prompt)
    answer = answer.strip()
    if not answer:
        answer = "Not in page"
    sources = [chunk.label for chunk, _ in retrieved]
    return answer, sources


async def generate_notes(title: str, text: str, output_format: str) -> str:
    if output_format == "json":
        requested_format = (
            'Return valid JSON with keys "bullet_points", "definitions", '
            '"key_concepts", and "examples". Each value must be an array of strings.'
        )
    else:
        requested_format = (
            "Return markdown with sections: Bullet Points, Definitions, Key Concepts, and Examples. "
            "Use short bullet lists and keep it export-ready."
        )

    prompt = (
        "Create structured study notes from the provided webpage content.\n"
        "Do not answer questions or add outside facts.\n"
        f"{requested_format}\n\n"
        f"Title: {title}\n\n"
        "Content:\n"
        f"{text[:18000]}"
    )
    return await generate_text(prompt)


async def generate_text(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
        response.raise_for_status()
        payload = response.json()

    if "response" in payload:
        return payload["response"]
    return json.dumps(payload)
