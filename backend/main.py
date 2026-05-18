from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str


class StatsRequest(BaseModel):
    text: str = ""


class StatsResponse(BaseModel):
    characters: int
    characters_no_spaces: int
    lines: int
    words: int


class UploadResponse(BaseModel):
    filename: str
    content: str
    stats: StatsResponse


@lru_cache(maxsize=1)
def get_rag_graph():
    """Import and cache the compiled RAG graph once per backend process."""
    from RAG import graph

    return graph


def build_stats(text: str) -> StatsResponse:
    return StatsResponse(
        characters=len(text),
        characters_no_spaces=sum(1 for char in text if not char.isspace()),
        lines=0 if text == "" else text.count("\n") + 1,
        words=len(text.split()),
    )


app = FastAPI(title="Language Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    def invoke_rag() -> str:
        graph = get_rag_graph()
        result = graph.invoke({"messages": [{"role": "user", "content": message}]})
        return result["messages"][-1].content

    try:
        answer = await asyncio.to_thread(invoke_rag)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(answer=answer)


@app.post("/stats", response_model=StatsResponse)
def stats(request: StatsRequest):
    return build_stats(request.text)


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    allowed_suffixes = {".txt", ".md", ".csv"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload .txt, .md, or .csv files.",
        )

    raw_content = await file.read()
    try:
        content = raw_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not decode file as UTF-8 text.",
        ) from exc

    return UploadResponse(
        filename=file.filename or "uploaded-file",
        content=content,
        stats=build_stats(content),
    )
