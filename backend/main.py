from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from sudachipy import dictionary as sudachi_dictionary
from sudachipy import tokenizer as sudachi_tokenizer
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


class SimplifyRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target_level: str = Field(..., pattern=r"^N[1-5]$")


class SimplifyResponse(BaseModel):
    text: str
    target_level: str
    annotations: list[AnnotatedToken] = Field(default_factory=list)


class AnnotatedToken(BaseModel):
    text: str
    base_form: str | None = None
    part_of_speech: str | None = None
    is_content: bool = False
    is_proper_noun: bool = False
    jlpt_level: int | None = None
    reading: str | None = None
    frequency_rank: int | None = None
    kanji_levels: list[int] = Field(default_factory=list)
    unknown_kanji_count: int = 0


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
def get_simplification_model():
    """Create a small cached chat model for text simplification requests."""
    return init_chat_model("gemma4:31b-cloud", model_provider="ollama", temperature=0.2)


@lru_cache(maxsize=1)
def get_frequency_vocab() -> dict[str, int]:
    """Load word-frequency ranks. Higher rank means rarer/more difficult."""
    frequency_path = (
        Path(__file__).parent.parent
        / "documents"
        / "jpn_newscrawl_2019_1M-words.txt"
    )
    frequencies: dict[str, int] = {}

    if not frequency_path.exists():
        return frequencies

    with frequency_path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rank_text, word = parts[:2]
            if not word or word in frequencies:
                continue
            try:
                frequencies[word] = int(rank_text)
            except ValueError:
                continue

    return frequencies


@lru_cache(maxsize=1)
def get_sudachi_tokenizer():
    """Create the Sudachi tokenizer once; used for surfaces, POS, and lemmas."""
    return sudachi_dictionary.Dictionary().create()


@lru_cache(maxsize=1)
def get_jlpt_vocab() -> dict[str, dict[str, int | str]]:
    """Load JLPT vocabulary for fast surface-form and reading lookups."""
    vocab_path = Path(__file__).parent.parent / "documents" / "JLPT_vocab_ALL.csv"
    vocab: dict[str, dict[str, int | str]] = {}
    kana_candidates: dict[str, dict[str, int | str]] = {}

    def is_kana_key(key: str) -> bool:
        return bool(re.fullmatch(r"[\u3040-\u30ffー]+", key))

    def keep_easiest_entry(
        target: dict[str, dict[str, int | str]],
        key: str,
        entry: dict[str, int | str],
    ) -> None:
        """Keep the easiest (and, on ties, shortest-reading) entry for a key.

        Many headwords are homographs with several CSV rows (私 → あたし/N1,
        わたし/N5; 今日 → こんにち/N3, きょう/N5). The previous code kept whichever
        row came first in the file for kanji keys, which surfaced rare/formal
        readings at the wrong JLPT level. Preferring the highest level number
        (= easiest JLPT) picks the common everyday word, and the shortest-reading
        tiebreak favours the basic reading (わたし over わたくし, にほん over にっぽん)
        instead of an arbitrary file-order choice.
        """
        current_entry = target.get(key)
        if current_entry is None:
            target[key] = entry
            return
        current_level = int(current_entry["level"])
        if level > current_level:
            target[key] = entry
        elif level == current_level and len(str(entry["reading"])) < len(
            str(current_entry["reading"])
        ):
            target[key] = entry

    with vocab_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            level = int(row["Level"])
            surface = row["Kanji"]
            reading = row["Reading"]
            entry = {"level": level, "reading": reading}

            if is_kana_key(surface):
                keep_easiest_entry(kana_candidates, surface, entry)
            else:
                # The CSV stores する-verbs with a する-suffixed reading
                # (勉強 → べんきょうする). For a bare kanji surface that does not
                # itself end in する, drop the suffix so the displayed reading
                # matches the surface (勉強 → べんきょう) instead of trailing する.
                surface_entry = entry
                if reading.endswith("する") and not surface.endswith("する"):
                    surface_entry = {"level": level, "reading": reading[:-2]}
                keep_easiest_entry(vocab, surface, surface_entry)

            keep_easiest_entry(kana_candidates, reading, entry)

    vocab.update(kana_candidates)
    return vocab


@lru_cache(maxsize=1)
def get_jlpt_kanji() -> dict[str, int]:
    """Load JLPT kanji levels keyed by individual kanji characters."""
    kanji_path = Path(__file__).parent.parent / "documents" / "JLPT_kanji_ALL.csv"
    kanji_levels: dict[str, int] = {}

    if not kanji_path.exists():
        return kanji_levels

    with kanji_path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            kanji, level_text = row[:2]
            try:
                kanji_levels[kanji] = int(level_text)
            except ValueError:
                continue

    return kanji_levels


KANJI_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
HIRAGANA_ONLY_RE = re.compile(r"[\u3040-\u309fー]+")
JAPANESE_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaffー々〆〤]+")
JITENDEX_URL = (
    "https://github.com/stephenmk/stephenmk.github.io/releases/latest/download/"
    "jitendex-yomitan.zip"
)
JITENDEX_ZIP_PATH = Path(__file__).parent / "data" / "jitendex-yomitan.zip"
JITENDEX_CACHE_PATH = Path(__file__).parent / "data" / "jitendex-cache.json"


def build_annotations(text: str) -> AnnotationResponse:
    vocab = get_jlpt_vocab()
    frequencies = get_frequency_vocab()
    kanji_lookup = get_jlpt_kanji()
    sudachi = get_sudachi_tokenizer()
    tokens: list[AnnotatedToken] = []
    cursor = 0

    for match in JAPANESE_RUN_RE.finditer(text):
        if match.start() > cursor:
            tokens.append(AnnotatedToken(text=text[cursor:match.start()]))

        morphemes = list(
            sudachi.tokenize(match.group(), sudachi_tokenizer.Tokenizer.SplitMode.A)
        )
        words = [morpheme.surface() for morpheme in morphemes]
        lemmas = [morpheme.dictionary_form() for morpheme in morphemes]
        pos_details = [morpheme.part_of_speech() for morpheme in morphemes]
        postags = [pos_detail[0] for pos_detail in pos_details]
        index = 0
        while index < len(words):
            best_surface = words[index]
            allow_lemma_lookup = not HIRAGANA_ONLY_RE.fullmatch(best_surface)
            best_base_form = (
                lemmas[index] if allow_lemma_lookup and lemmas[index] != best_surface else None
            )
            best_pos_details = [pos_details[index]]
            best_entry = vocab.get(best_surface) or (
                vocab.get(lemmas[index]) if allow_lemma_lookup else None
            )
            best_lookup_key = (
                best_surface
                if best_surface in frequencies or not allow_lemma_lookup
                else lemmas[index]
            )
            best_end = index + 1

            for end in range(min(len(words), index + 8), index, -1):
                if end > index + 1 and any(
                    pos in {"助詞", "助動詞", "補助記号"} for pos in postags[index:end]
                ):
                    continue

                surface = "".join(words[index:end])
                base_form = "".join(lemmas[index:end])
                allow_lemma_lookup = not HIRAGANA_ONLY_RE.fullmatch(surface)
                entry = vocab.get(surface) or (vocab.get(base_form) if allow_lemma_lookup else None)
                if entry:
                    best_surface = surface
                    best_base_form = base_form if allow_lemma_lookup and base_form != surface else None
                    best_entry = entry
                    best_lookup_key = surface if surface in frequencies or not allow_lemma_lookup else base_form
                    best_pos_details = pos_details[index:end]
                    best_end = end
                    break

            is_grammar_token = best_end == index + 1 and postags[index] in {
                "助詞",
                "助動詞",
                "補助記号",
            }
            if is_grammar_token:
                best_base_form = None
                best_entry = None
                best_lookup_key = ""
            elif best_entry is None and HIRAGANA_ONLY_RE.fullmatch(best_surface):
                best_lookup_key = ""

            kanji_chars = KANJI_RE.findall(best_surface)
            kanji_levels = [kanji_lookup[char] for char in kanji_chars if char in kanji_lookup]
            unknown_kanji_count = len(kanji_chars) - len(kanji_levels)
            is_proper_noun = any(
                len(pos_detail) > 1 and pos_detail[1] == "固有名詞"
                for pos_detail in best_pos_details
            )
            is_auxiliary_hiragana_verb = (
                best_entry is None
                and HIRAGANA_ONLY_RE.fullmatch(best_surface)
                and any(pos_detail[0] == "動詞" for pos_detail in best_pos_details)
            )
            is_content = not is_grammar_token and not is_auxiliary_hiragana_verb and any(
                pos_detail[0] in {"名詞", "動詞", "形容詞", "副詞"}
                for pos_detail in best_pos_details
            )

            tokens.append(
                AnnotatedToken(
                    text=best_surface,
                    base_form=best_base_form,
                    part_of_speech="/".join(pos_detail[0] for pos_detail in best_pos_details),
                    is_content=is_content,
                    is_proper_noun=is_proper_noun,
                    jlpt_level=best_entry["level"] if best_entry else None,
                    reading=best_entry["reading"] if best_entry else None,
                    frequency_rank=frequencies.get(best_lookup_key),
                    kanji_levels=kanji_levels,
                    unknown_kanji_count=unknown_kanji_count,
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


SIMPLIFICATION_CACHE: dict[tuple[str, str], str] = {}


def build_simplification_prompt(source_text: str, target_level: str) -> str:
    return f"""
Rewrite this Japanese text for JLPT {target_level}. Keep meaning, names, numbers, and tone. Use natural {target_level}-level vocabulary/grammar. Return only Japanese text, no notes.

{source_text}
""".strip()


@lru_cache(maxsize=128)
def simplify_text_cached(source_text: str, target_level: str) -> str:
    """Generate and cache simplified variants by exact source text and target level."""
    prompt = build_simplification_prompt(source_text, target_level)
    model = get_simplification_model()
    response = model.invoke([{"role": "user", "content": prompt}])
    simplified_text = response.content.strip()
    SIMPLIFICATION_CACHE[(source_text, target_level)] = simplified_text
    return simplified_text


def stream_simplified_text(source_text: str, target_level: str):
    """Yield simplified text chunks from Ollama as soon as tokens arrive."""
    cached_text = SIMPLIFICATION_CACHE.get((source_text, target_level))
    if cached_text is not None:
        for index in range(0, len(cached_text), 8):
            yield cached_text[index : index + 8]
        return

    prompt = build_simplification_prompt(source_text, target_level)
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    payload = json.dumps(
        {
            "model": "kimi-k2.6:cloud",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "think": False,
            "options": {"temperature": 0.2},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{ollama_host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    chunks: list[str] = []

    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.strip():
                continue
            event = json.loads(line.decode("utf-8"))
            if event.get("error"):
                raise RuntimeError(event["error"])
            content = event.get("message", {}).get("content", "")
            if not content:
                continue
            chunks.append(content)
            yield content

    SIMPLIFICATION_CACHE[(source_text, target_level)] = "".join(chunks).strip()


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


@app.post("/simplify", response_model=SimplifyResponse)
async def simplify(request: SimplifyRequest):
    source_text = request.text.strip()
    if not source_text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        simplified_text = await asyncio.to_thread(
            simplify_text_cached, source_text, request.target_level
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not simplified_text:
        raise HTTPException(status_code=500, detail="Simplification returned no text.")

    annotations = build_annotations(simplified_text).tokens
    return SimplifyResponse(
        text=simplified_text,
        target_level=request.target_level,
        annotations=annotations,
    )


@app.post("/simplify/stream")
async def simplify_stream(request: SimplifyRequest):
    source_text = request.text.strip()
    if not source_text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    async def event_stream():
        simplified_chunks: list[str] = []
        try:
            for chunk in stream_simplified_text(source_text, request.target_level):
                simplified_chunks.append(chunk)
                yield json.dumps({"type": "chunk", "text": chunk}, ensure_ascii=False) + "\n"
                await asyncio.sleep(0.025)

            simplified_text = "".join(simplified_chunks).strip()
            if not simplified_text:
                yield json.dumps(
                    {"type": "error", "detail": "Simplification returned no text."},
                    ensure_ascii=False,
                ) + "\n"
                return

            annotations = [token.model_dump() for token in build_annotations(simplified_text).tokens]
            yield json.dumps(
                {
                    "type": "done",
                    "text": simplified_text,
                    "target_level": request.target_level,
                    "annotations": annotations,
                },
                ensure_ascii=False,
            ) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
