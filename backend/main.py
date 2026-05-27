from __future__ import annotations

import asyncio
import csv
import json
import re
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import nagisa
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str


class StatsRequest(BaseModel):
    text: str = ""


class AnnotateRequest(BaseModel):
    text: str = ""


class TranslationRequest(BaseModel):
    word: str = Field(..., min_length=1)


class AnnotatedToken(BaseModel):
    text: str
    jlpt_level: int | None = None
    reading: str | None = None


class AnnotationResponse(BaseModel):
    tokens: list[AnnotatedToken]


class TranslationResponse(BaseModel):
    word: str
    translation: str
    reading: str | None = None


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


@lru_cache(maxsize=1)
def get_jlpt_vocab() -> dict[str, dict[str, int | str]]:
    """Load JLPT vocabulary for fast surface-form and reading lookups."""
    vocab_path = Path(__file__).parent.parent / "documents" / "JLPT_vocab_ALL.csv"
    vocab: dict[str, dict[str, int | str]] = {}

    with vocab_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            level = int(row["Level"])
            entry = {"level": level, "reading": row["Reading"]}
            vocab.setdefault(row["Kanji"], entry)
            vocab.setdefault(row["Reading"], entry)

    return vocab


JAPANESE_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaffー々〆〤]+")
JITENDEX_URL = (
    "https://github.com/stephenmk/stephenmk.github.io/releases/latest/download/"
    "jitendex-yomitan.zip"
)
JITENDEX_ZIP_PATH = Path(__file__).parent / "data" / "jitendex-yomitan.zip"
JITENDEX_CACHE_PATH = Path(__file__).parent / "data" / "jitendex-cache.json"


def build_annotations(text: str) -> AnnotationResponse:
    vocab = get_jlpt_vocab()
    tokens: list[AnnotatedToken] = []
    cursor = 0

    for match in JAPANESE_RUN_RE.finditer(text):
        if match.start() > cursor:
            tokens.append(AnnotatedToken(text=text[cursor:match.start()]))

        tagged = nagisa.tagging(match.group())
        words = tagged.words
        postags = tagged.postags
        index = 0
        while index < len(words):
            best_surface = words[index]
            best_entry = vocab.get(best_surface)
            best_end = index + 1

            for end in range(min(len(words), index + 8), index, -1):
                surface = "".join(words[index:end])
                entry = vocab.get(surface)
                if entry:
                    best_surface = surface
                    best_entry = entry
                    best_end = end
                    break

            if best_end == index + 1 and (
                postags[index] in {"助詞", "助動詞", "補助記号"}
                or re.fullmatch(r"[\u3040-\u309f]", best_surface)
            ):
                best_entry = None

            tokens.append(
                AnnotatedToken(
                    text=best_surface,
                    jlpt_level=best_entry["level"] if best_entry else None,
                    reading=best_entry["reading"] if best_entry else None,
                )
            )
            index = best_end

        cursor = match.end()

    if cursor < len(text):
        tokens.append(AnnotatedToken(text=text[cursor:]))

    return AnnotationResponse(tokens=tokens)


def ensure_jitendex_zip() -> Path:
    """Download the Jitendex Yomitan dictionary once and reuse it locally."""
    if not JITENDEX_ZIP_PATH.exists():
        JITENDEX_ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(JITENDEX_URL, JITENDEX_ZIP_PATH)

    return JITENDEX_ZIP_PATH


def collect_text(value: Any) -> str:
    """Extract readable text from Yomitan structured-content nodes."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(filter(None, (collect_text(item).strip() for item in value)))
    if isinstance(value, dict):
        return collect_text(value.get("content", ""))
    return ""


def collect_glosses(value: Any, in_glossary: bool = False) -> list[str]:
    """Find concise glossary items inside Jitendex structured content."""
    glosses: list[str] = []

    if isinstance(value, list):
        for item in value:
            glosses.extend(collect_glosses(item, in_glossary))
    elif isinstance(value, dict):
        data = value.get("data")
        content_type = data.get("content") if isinstance(data, dict) else None
        next_in_glossary = in_glossary or content_type == "glossary"

        if in_glossary and value.get("tag") == "li":
            text = collect_text(value).strip()
            if text:
                glosses.append(text)
        else:
            glosses.extend(collect_glosses(value.get("content"), next_in_glossary))

    return glosses


@lru_cache(maxsize=1)
def get_jitendex_data() -> dict[str, dict[str, list[str] | str]]:
    """Load Jitendex into a compact exact-match lookup and cache it on disk."""
    if JITENDEX_CACHE_PATH.exists():
        with JITENDEX_CACHE_PATH.open(encoding="utf-8") as file:
            return json.load(file)

    dictionary: dict[str, list[str]] = {}
    readings: dict[str, str] = {}

    with zipfile.ZipFile(ensure_jitendex_zip()) as archive:
        term_banks = sorted(
            name for name in archive.namelist() if name.startswith("term_bank_")
        )
        for name in term_banks:
            entries = json.loads(archive.read(name).decode("utf-8"))
            for entry in entries:
                expression = entry[0]
                reading = entry[1]
                definitions = entry[5]
                if reading:
                    readings.setdefault(expression, reading)
                    readings.setdefault(reading, reading)

                glosses = []
                for definition in definitions:
                    glosses.extend(collect_glosses(definition))

                clean_glosses = []
                for gloss in glosses:
                    clean_gloss = re.sub(r"\s+", " ", gloss).strip()
                    if clean_gloss and clean_gloss not in clean_glosses:
                        clean_glosses.append(clean_gloss)

                for key in {expression, reading} - {""}:
                    existing = dictionary.setdefault(key, [])
                    for gloss in clean_glosses[:3]:
                        if gloss not in existing:
                            existing.append(gloss)
                    del existing[3:]

    data = {"dictionary": dictionary, "readings": readings}
    JITENDEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JITENDEX_CACHE_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)

    return data


@lru_cache(maxsize=1)
def get_jitendex_readings() -> dict[str, str]:
    return get_jitendex_data()["readings"]


@lru_cache(maxsize=512)
def translate_japanese_word(word: str) -> str:
    glosses = get_jitendex_data()["dictionary"].get(word)
    if not glosses:
        return "Translation unavailable"

    return "; ".join(glosses)[:160]


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


@app.post("/annotate", response_model=AnnotationResponse)
def annotate(request: AnnotateRequest):
    return build_annotations(request.text)


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    word = request.word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="Word must not be empty.")

    try:
        translation = await asyncio.to_thread(translate_japanese_word, word)
    except Exception:
        translation = "Translation unavailable"

    reading = get_jitendex_readings().get(word)
    return TranslationResponse(word=word, translation=translation, reading=reading)


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
