# Language Assistant — How It Works

Interview preparation deep-dive: architecture, tools, data, and the reasoning behind the
design decisions. Everything here is grounded in the actual code (file references are
given throughout).

---

## 1. Elevator pitch

**Readr** is a reading and vocabulary assistant for language learners of **Japanese** and
**Hebrew**. A learner pastes (or uploads) a text and gets:

1. **Inline difficulty annotation** — every word is color-coded by JLPT level (Japanese,
   N5–N1) or CEFR level (Hebrew, A1–C1), with hover tooltips showing the reading,
   dictionary translation, and level.
2. **A difficulty score** (0–100) for the whole text, computed from vocabulary
   percentiles, kanji density, sentence length, and grammar signals, plus a per-level
   word-distribution chart.
3. **An "easier version"** — an LLM rewrites the text at a chosen target level (e.g.
   "rewrite this N2 article at N4"), streamed token-by-token into the UI and then
   re-annotated so the learner can verify the rewrite really is easier.
4. **A tutor chat** — a RAG (retrieval-augmented generation) chatbot that answers
   vocabulary questions grounded in curated JLPT/CEFR datasets rather than the model's
   memory, switchable between a local model (Ollama) and a hosted one (Groq).

Three moving parts: a **React/Vite frontend**, a **FastAPI backend**, and a
**LangGraph agentic-RAG pipeline** backed by **Chroma** vector stores.

---

## 2. Tech stack at a glance

| Layer | Technology | Why |
|---|---|---|
| Frontend | React 19 + Vite | Fast dev loop, single-page app; `react-markdown` + `remark-gfm` for rendering tutor answers |
| API | FastAPI + Uvicorn + Pydantic | Async endpoints, automatic request validation, typed response models, easy CORS |
| RAG orchestration | LangChain 1.x + LangGraph | Graph-based control flow (retrieve → grade → rewrite → answer) instead of a fixed chain |
| Vector store | Chroma (persisted on disk) | Local, zero-infra, per-language persist directories |
| Embeddings | Ollama `embeddinggemma` | Local, free, and **pinned** — the persisted vectors depend on it (see §6.3) |
| Chat LLMs | Ollama `gemma4:31b-cloud` (default) or Groq `llama-3.3-70b-versatile` | User-selectable provider toggle; local-first with a hosted fallback |
| Japanese NLP | SudachiPy + sudachidict-core | Morphological analysis: segmentation, part-of-speech, lemmas (Japanese has no spaces) |
| Japanese dictionary | Jitendex (Yomitan zip) | High-quality open JMdict-derived dictionary for tooltip translations |
| Package management | `uv` (Python), npm (JS) | Reproducible environments (`uv.lock`) |

Repository layout:

```
RAG.py                 # LangGraph RAG pipeline (language-agnostic, config-driven)
backend/main.py        # FastAPI app: chat, annotate, simplify, translate, upload, stats
frontend/src/App.jsx   # Entire React UI (single component + helpers)
interface.py           # Legacy Streamlit UI (first prototype, superseded by React)
documents/             # Source datasets (CSVs + frequency list)
chroma_db_*/           # Persisted per-language vector stores
backend/data/          # Jitendex zip + preprocessed JSON cache
```

---

## 3. The data (and why it matters)

The whole product concept rests on **curated difficulty-labelled vocabulary lists**:

| File | Content | Used for |
|---|---|---|
| `documents/JLPT_vocab_ALL.csv` | Japanese word, reading, English meaning, JLPT level (columns `Kanji,Reading,Meaning,Level`) | RAG corpus + annotation lookup |
| `documents/JLPT_kanji_ALL.csv` | Individual kanji → JLPT level | Per-character kanji difficulty in annotations |
| `documents/hebrew_cefr_final.csv` | Rank, Hebrew word, English gloss, CEFR level, confidence, source (`model_predicted`) | RAG corpus + annotation + tooltip translation |
| `documents/jpn_newscrawl_2019_1M-words.txt` | Leipzig Corpora word-frequency list (rank → word) | Difficulty estimate for words *not* on any JLPT list |
| `backend/data/jitendex-yomitan.zip` | Jitendex dictionary in Yomitan format | Japanese tooltip translations + readings |

Two things worth mentioning in an interview:

- **The Hebrew CEFR levels are model-predicted** (the CSV carries a `confidence` column
  and `source=model_predicted`). There is no free official CEFR-tagged Hebrew frequency
  list, so one was generated — a pragmatic data-engineering decision, made transparent
  in the data.
- **The frequency list is a fallback difficulty signal**: if a Japanese word isn't in the
  JLPT lists, its newscrawl frequency rank is bucketed into an approximate level
  (≤2,000 → ~N3-ish, ≤5,000 → harder, >10,000 → hardest). Same idea for Hebrew ranks
  → CEFR buckets. This avoids the earlier bug of "unknown word = maximum difficulty".

---

## 4. Architecture overview

```
┌─────────────────────────────┐        ┌──────────────────────────────────────┐
│  React SPA (Vite, :5173)    │  HTTP  │  FastAPI backend (:8000)             │
│                             │───────▶│                                      │
│  • contentEditable reader   │        │  POST /chat      → LangGraph RAG     │
│  • difficulty scoring (JS)  │        │  POST /annotate  → Sudachi/regex     │
│  • streaming reader (fetch) │        │  POST /simplify/stream → Ollama NDJSON│
│  • tooltips + charts        │        │  POST /translate → Jitendex/CSV      │
└─────────────────────────────┘        │  POST /upload, /stats, GET /health   │
                                       └───────────┬──────────────────────────┘
                                                   │
                     ┌─────────────────────────────┼───────────────────────────┐
                     ▼                             ▼                           ▼
          LangGraph graph (RAG.py)       SudachiPy tokenizer         Ollama / Groq LLMs
          Chroma + embeddinggemma        + in-memory CSV lookups     (chat + simplify)
```

A deliberate split of responsibilities:

- **The backend owns facts** (tokenization, dictionary lookups, level tags, LLM calls).
- **The frontend owns judgement** (the difficulty formula, charts, tooltips) — scoring
  runs client-side over the annotation tokens, so tweaking weights needs no backend
  round-trip and re-scoring is instant when switching between source and easier text.

---

## 5. The RAG pipeline (`RAG.py`) — the centerpiece

### 5.1 It's *agentic* RAG, not a fixed chain

The graph is built with LangGraph's `StateGraph(MessagesState)` and has four nodes:

```
START ─▶ generate_query_or_respond ──(tools_condition)──▶ END   (no retrieval needed)
                    ▲                        │
                    │                        ▼
             rewrite_question ◀──(grade: "no")── retrieve (ToolNode)
                                             │
                                     (grade: "yes" / cap hit)
                                             ▼
                                      generate_answer ─▶ END
```

1. **`generate_query_or_respond`** — the LLM sees the user message with the retriever
   *bound as a tool* (`bind_tools`). It decides itself whether to call the retriever
   (and with what query) or to answer directly. Small talk like "hello" never hits the
   vector store. This is the "agentic" part: retrieval is a tool call, not a mandatory
   step.
2. **`retrieve`** — a LangGraph `ToolNode` executes the retriever tool: a Chroma
   similarity search returning the top **k=6** chunks, joined into a context string.
3. **`grade_documents`** — a *separate grader model at temperature 0* judges relevance
   with a forced one-word `yes`/`no` answer. Relevant → `generate_answer`; not →
   `rewrite_question`. (An earlier version used free-form grading output and broke on
   parsing — commit `9b557e5` "Fix RAG grading output parsing"; the fix is the strict
   "Respond with exactly one word" instruction plus `score.startswith("yes")`.)
4. **`rewrite_question`** — the LLM reformulates the question into a better vocabulary
   lookup and loops back to step 1.
5. **`generate_answer`** — a "mentor" prompt: answer from the retrieved context, admit
   ignorance if the context doesn't cover it, max three sentences.

### 5.2 The rewrite loop is capped at one iteration — a real production lesson

`grade_documents` counts `HumanMessage`s in the state to detect how many rewrites have
happened and **forces `generate_answer` after one rewrite**. The code comment explains
why: every extra loop costs 3–4 sequential LLM calls, and an unbounded loop kept chat
requests spinning until LangGraph's recursion limit. Commit `3b862f6` ("Fix slow chat")
introduced the cap. Interview takeaway: *self-correcting agent loops need explicit
budgets; grader models are noisy and will happily loop forever.*

### 5.3 One config-driven graph factory for two languages

There is no duplicated Japanese/Hebrew pipeline. `LANGUAGE_CONFIG` holds per-language
document paths, Chroma directory, tool name/description, mentor prompt, and rewrite
prompt; `build_graph(language, provider)` assembles the graph from it. The retriever
tool's `name` and `description` are set per language — important because the *LLM reads
the tool description* to decide when and how to call it. Adding a third language means
adding one config entry and a CSV.

`build_graph` and `get_retriever` are wrapped in `functools.lru_cache`, so each
(language, provider) graph and each language's retriever are built exactly once per
process — the pattern used all over the backend for lazy singletons.

### 5.4 Vector store construction details

- CSVs are loaded with LangChain's `CSVLoader` (one document per row) and split with
  `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=50)`
  — small chunks because each unit of meaning is a single vocabulary row.
- **Rebuild detection counts embeddings, not files**: a leftover `chroma.sqlite3` with
  zero vectors used to pass a naive "directory not empty" check, silently yielding an
  empty retriever. The fix checks `vectorstore._collection.count() == 0`.
- **Batched inserts (200 docs/batch)**: Ollama's embedding runner crashes on one huge
  embed request, so the corpus is inserted in batches.

---

## 6. The provider toggle (Ollama vs Groq)

- The UI has a **Local / Groq** segmented control; the choice travels with every
  `/chat` request and selects the chat model:
  `ollama → gemma4:31b-cloud`, `groq → llama-3.3-70b-versatile` (overridable via
  `GROQ_MODEL`). Models are created via LangChain's provider-agnostic
  `init_chat_model`, so swapping providers is pure configuration.
- The backend **fails fast with a 400** ("Groq mode needs a GROQ_API_KEY…") instead of
  letting the request die deep inside LangChain with a cryptic error.
- **Key decision — embeddings never switch providers.** The comment in `RAG.py` says it
  outright: the persisted Chroma stores were built with `embeddinggemma` vectors, so the
  chat provider toggle must not touch retrieval. Mixing embedding models within one
  index makes similarity scores meaningless; switching would require a full re-index.
  Separating the *generation* model (swappable) from the *embedding* model (pinned) is
  the key insight here.
- Temperatures are intentional: **0.2** for generation (mostly factual, slight
  naturalness), **0** for the grader (deterministic classification).

---

## 7. The annotation engine (`backend/main.py`)

### 7.1 Japanese — the hard case

Japanese has no spaces, so annotation needs real morphological analysis:

1. **Regex pre-scan** (`JAPANESE_RUN_RE`) splits the text into Japanese runs vs.
   passthrough text (Latin, digits, whitespace), so only Japanese gets tokenized.
2. **SudachiPy in SplitMode.A** (shortest units) produces surfaces, dictionary forms
   (lemmas), and part-of-speech details per morpheme.
3. **Greedy longest-match re-merging**: mode A over-segments compounds, so the code
   tries to merge up to **8 consecutive morphemes** (longest first) and looks each
   candidate up in the JLPT vocabulary — matching both surface and lemma. First hit
   wins. Merged spans containing particles/auxiliaries/punctuation POS tags are skipped
   so grammar never gets glued into a "word".
4. **POS-based classification**: particles/auxiliaries are grammar tokens (never
   scored); a token is a *content word* if any morpheme is a noun, verb, adjective, or
   adverb; proper nouns (固有名詞) are flagged and later **excluded from difficulty
   scoring** (a name isn't "hard vocabulary").
5. **Per-token payload** (`AnnotatedToken`): JLPT level, reading, frequency rank,
   per-kanji levels + count of unlisted kanji, lemma, POS.

Two subtle correctness fixes worth telling as stories:

- **Homograph resolution** (`keep_easiest_entry`, commit `2923279`): headwords like 私
  appear in the CSV as both あたし/N1 and わたし/N5. The old code kept whichever row came
  first in the file, so everyday words showed up as advanced. The fix prefers the
  **easiest JLPT level**, tie-broken by **shortest reading** (わたし over わたくし) — a
  principled rule instead of file-order luck.
- **する-verb readings**: the CSV stores 勉強 with reading べんきょうする; for a bare 勉強
  surface the する suffix is stripped so the displayed reading matches the surface.
- Hiragana-only surfaces skip lemma lookup (too ambiguous), and hiragana-only auxiliary
  verbs (いる, なる…) are excluded from content-word scoring.

### 7.2 Hebrew — deliberately simpler

Hebrew is space-separated, so no morphological analyzer is used. A regex
(`HEBREW_RUN_RE`, which also tolerates geresh/gershayim/hyphens inside words) finds
words; `normalize_hebrew_word` strips **niqqud and cantillation marks**
(U+0591–U+05C7) so vocalized text still matches the unvocalized CSV; lookups try the
raw surface first, then the normalized form. Bidi control characters (LRM/RLM,
embedding/isolate marks) are stripped during normalization because LLM output and
copy-pasted text often carry them and they wreck RTL rendering.

An honest limitation to volunteer: without a Hebrew morphological analyzer, inflected
forms and prefixed forms (ו־, ה־, ב־…) miss the lexicon unless that exact form is
listed. The right next step would be a tool like HebSpacy or YAP.

### 7.3 Text normalization before annotation/simplification

`normalize_text_for_language` flattens LLM layout artifacts (markdown tables, pipes,
bullet lists, one-word-per-line output) into prose, collapses whitespace *between
Japanese characters* (models love inserting spaces there), and for Hebrew removes stray
verse numbers and bidi controls. This exists because commit history shows the "easier
text" repeatedly came back garbled/table-shaped (`e632c64`, `a3f7e60`) — prompt
instructions alone ("no tables, no columns") were not reliable, so the output is
sanitized defensively. *Lesson: never trust LLM formatting; normalize on the boundary.*

---

## 8. Text simplification ("Easier version")

- Endpoint pair: `/simplify` (blocking, cached via `lru_cache(maxsize=128)`) and
  `/simplify/stream` (the one the UI uses).
- The prompt is compact and constraint-heavy: keep meaning/names/numbers/tone, target
  the exact JLPT/CEFR level, return *only* text in the target language, no tables/notes.
- **Streaming is hand-rolled NDJSON over Ollama's raw `/api/chat` HTTP API** rather than
  going through LangChain — chunks are yielded the moment tokens arrive, each wrapped
  as `{"type": "chunk", "text": ...}` on its own line. When generation finishes, the
  backend normalizes the full text, **re-annotates it**, and emits a final
  `{"type": "done", text, annotations}` event; errors become `{"type": "error"}` events
  in-stream. `StreamingResponse` sets `X-Accel-Buffering: no` so proxies don't buffer.
- A module-level `SIMPLIFICATION_CACHE` dict is shared between the streaming and
  blocking paths (keyed by exact source text + level + language); a cache hit is
  *replayed* in 8-character chunks so the UI experience stays consistent.
- The frontend reads the stream with `fetch` + `ReadableStream.getReader()`, splits on
  newlines, updates state per chunk (throttled by `requestAnimationFrame`), and swaps in
  the fully-annotated version on the `done` event.
- Validation is defense-in-depth: Pydantic regex on `target_level`
  (`^(N[1-5]|A1|A2|B1|B2|C1)$`) *plus* a language/level cross-check (Japanese must use
  N-levels, Hebrew CEFR levels) returning a clean 400.

The product loop this enables: read → see difficulty → generate easier version →
**the rewrite is itself annotated and scored**, so the learner (and the developer)
can verify the model actually hit the target level.

---

## 9. Dictionary lookups (`/translate`)

- **Japanese**: Jitendex ships as a Yomitan zip of `term_bank_*.json` files whose
  definitions are nested "structured content" (HTML-ish trees). On first use the
  backend downloads the zip, walks every entry, recursively extracts only the `li`
  items inside `glossary`-tagged nodes (`collect_glosses`), dedupes, **keeps max 3
  glosses per key**, indexes by both expression and reading, and writes the whole thing
  to `jitendex-cache.json`. Every later startup is one JSON load. Classic
  **preprocess-once, serve-fast** pattern; per-word results additionally get
  `lru_cache(maxsize=512)`.
- **Hebrew**: translation comes straight from the CEFR CSV gloss, formatted as
  `gloss (level)`.
- The endpoint **never 500s** — any failure degrades to `"Translation unavailable"`,
  because a tooltip is not worth an error banner.
- Frontend side: tooltip translations are fetched lazily on hover, cached in state, and
  a `pendingTranslationsRef` set deduplicates in-flight requests. If Jitendex returns a
  reading the token annotations are patched in place so furigana appears on subsequent
  hovers.

---

## 10. Difficulty scoring (frontend, `App.jsx`)

Runs entirely client-side over the annotation tokens. Only **content words that are not
proper nouns** count.

**Japanese score (0–100)** = `vocab·0.50 + kanji·0.20 + sentence-length·0.15 + grammar·0.15`

- *Vocabulary*: each word mapped to a 1–6 level (JLPT inverted: N5→1 … N1→5; unlisted
  words bucketed by frequency rank, capped at 6). The score averages the **mean, 75th
  and 90th percentile** — so a few advanced words raise the score noticeably, but the
  tail doesn't dominate. A **type-token-ratio bonus** (lexical diversity) is added,
  confidence-scaled by text length (`min(1, tokens/40)`) so tiny texts can't game it.
- *Kanji*: 70% level score (per-token blend of average and max kanji level, unknown
  kanji counted as hardest) + 30% kanji density.
- *Sentence length*: mean content-words per sentence vs. a baseline (6 for Japanese, 7
  for Hebrew), linearly scaled.
- *Grammar*: regex counts of classical/formal patterns (べき, ざる, をば, とて…) weighted
  1.4× plus passive/causative forms (られる, させられる…), normalized per sentence.

**Hebrew score** = `vocab·0.75 + sentence-length·0.25` (no kanji/grammar analog), with
CEFR levels mapped linearly to 0–100 and rank-based fallback buckets.

Percentiles-over-averages is the talking point: a text isn't "intermediate" because it
averages out — it's hard if its *hardest frequent words* are hard.

The UI also derives a **suggested target level** from the distribution: dominant level
minus/plus one step, so the "easier version" defaults to a sensible goal.

---

## 11. Frontend implementation notes

- **A single `App.jsx` component** (~1,300 lines) with helper functions — deliberate
  simplicity for a bootcamp-scale project; a stated next step would be splitting into
  components/hooks.
- **Config-driven bilingualism** mirrors the backend: a `languageConfigs` object holds
  labels, level scales, `dir` (`rtl` for Hebrew!), `lang` attributes, default targets,
  and CSS-class mappers. The whole UI flips RTL and re-labels itself off one state key.
- **The reader is a `contentEditable` div**, not a `<textarea>` — required to render
  color-coded `<span>`s inline while staying editable. Annotated spans are rebuilt via
  `replaceChildren()` + `document.createElement` (imperative on purpose: React
  reconciliation and `contentEditable` fight each other). Tooltip hit-testing uses one
  delegated `mousemove` handler reading `data-token-index` instead of per-span
  listeners.
- **Annotation is debounced 350 ms** with an `AbortController` cancelling stale
  requests — type fast, annotate once.
- **Race-condition hygiene**: `easierTextSource` remembers which source text produced
  the easier version and shows "Source text changed. Regenerate…" when stale; a
  `pendingSimplifyLevel` queue defers simplification until tokenization completes;
  changing the target level regenerates automatically.
- Tutor answers render as **markdown** (`react-markdown` + GFM) because the models
  return lists/tables; user messages render as plain text (no injection surface).

---

## 12. API surface summary

| Endpoint | Purpose | Notable behavior |
|---|---|---|
| `POST /chat` | RAG tutor answer | Runs the LangGraph graph in `asyncio.to_thread` so the sync graph doesn't block the event loop; 400 if Groq chosen without key |
| `POST /annotate` | Tokenize + level-tag text | Sudachi (ja) / regex+normalization (he) |
| `POST /simplify` | Blocking simplification | `lru_cache`d |
| `POST /simplify/stream` | Streaming simplification | NDJSON chunks → final `done` event with annotations |
| `POST /translate` | Word tooltip lookup | Jitendex (ja) / CEFR CSV (he); never errors |
| `POST /upload` | Text file upload | `.txt/.md/.csv` only, UTF-8 validated, returns content + stats |
| `POST /stats` | Character/word/line counts | Simple, kept from the first prototype |
| `GET /health` | Liveness | — |

CORS is locked to the Vite dev origins (localhost/127.0.0.1 on 5173/5174). Heavy
resources (tokenizer, vocab maps, frequency list, Jitendex, graphs, models) are all
`lru_cache`d lazy singletons — first request warms them, restarts stay cheap.

---

## 13. How the project evolved (git history as a narrative)

1. `9effe67` — RAG base script with Ollama (pure Python, no UI).
2. `9b557e5` — first RAG hardening: grading output parsing fixed via constrained output.
3. `4db62f9`/`5e2fa0c` — **Streamlit prototype** (`interface.py`): fastest way to a
   usable chat UI for validating the RAG loop.
4. `48d678c` — **replaced Streamlit with FastAPI + React**: Streamlit's rerun model
   can't do inline annotated editors, hover tooltips, or token streaming.
5. `27b2c76` — JLPT underlines + Jitendex dictionary integration.
6. `ff23b48` — difficulty scoring + "easier text" workflow (the core product loop).
7. `04d4110` — scoring fix: unknown words no longer rated maximum difficulty
   (frequency-rank fallback instead).
8. `b0f708b` — simplification model swap (kimi-k2.6 → gemma4) after quality issues.
9. `2923279` — homograph fix: easiest-JLPT-entry rule.
10. `e34e998`–`bdd7b27` — Hebrew mode hardening: bidi/niqqud normalization, garbled
    output sanitization, ledger ordering, markdown chat rendering.
11. `12134ca` — Groq/Ollama provider toggle end-to-end.
12. `3b862f6` — performance: rebuild-empty-vector-store detection + rewrite-loop cap.

The pattern to narrate: *prototype fast (script → Streamlit), rewrite when the UI
ceiling is hit, then iterate on correctness and performance with small, well-scoped
fixes — each commit message names a user-visible symptom.*

---

## 14. Key design decisions — quick reference for interview answers

1. **Agentic RAG over naive RAG**: the model decides *whether* to retrieve
   (tool-calling), retrieved docs are *graded*, and bad retrievals trigger *query
   rewriting*. But with a hard budget (one rewrite) because latency is a feature.
2. **Grader at temperature 0 with one-word constrained output** — treat LLM-as-judge as
   a classifier, not a conversation.
3. **Embedding model pinned independently of the chat provider** — persisted vectors
   are only comparable to queries embedded by the same model.
4. **Config-driven multilinguality** on both ends (one graph factory, one React config
   object) instead of parallel code paths.
5. **Right tool per language**: full morphological analysis for Japanese (Sudachi),
   lightweight regex + Unicode normalization for Hebrew — complexity where the language
   demands it.
6. **Deterministic facts, statistical judgement**: dictionaries/CSVs/tokenizers produce
   token facts server-side; the fuzzy difficulty formula lives client-side where it's
   cheap to iterate.
7. **Preprocess-once caches everywhere**: Jitendex zip → JSON cache; Chroma persisted
   and rebuilt only when the collection is actually empty; `lru_cache` for models,
   graphs, vocab maps, translations, simplifications.
8. **Streaming UX for the slowest operation** (simplification) via NDJSON, with an
   in-stream error protocol and a replayed cache so hits and misses feel the same.
9. **Sanitize LLM output at the boundary** — normalization functions exist because
   prompts alone couldn't stop tables/spaces/bidi garbage.
10. **Graceful degradation**: translation never 500s; Groq misconfiguration is a clear
    400; upload validates type and encoding before touching content.

---

## 15. Honest limitations & "what would you do next" answers

- **No tests.** First priority: pytest for the annotation edge cases (homographs,
  する-verbs, merged compounds, niqqud) and the graph routing; they're pure functions
  and easy to cover.
- **No streaming for chat** — `/chat` blocks until the graph finishes; LangGraph
  supports `stream()` and the UI already has streaming plumbing from `/simplify`.
- **Single-turn RAG**: each chat message is answered independently (`state["messages"][0]`
  is assumed to be the question); real conversation memory (checkpointer/threads) is a
  natural extension.
- **Hebrew morphology**: prefixed/inflected forms miss the lexicon; needs a real
  analyzer.
- **In-process, unbounded dict caches** (`SIMPLIFICATION_CACHE`, translations) — fine
  single-user, would move to Redis/TTL for multi-user.
- **No auth/rate limiting**, dev-only CORS — it's a local learning tool by design.
- **App.jsx monolith** — would split into `Reader`, `Chat`, `Ledger` components and
  custom hooks (`useAnnotations`, `useSimplifyStream`).
- **Difficulty formula is heuristic**, hand-tuned weights — could be validated against
  graded readers or learner data.

---

## 16. Thirty-second architecture answer (memorize this)

> "It's a language-learning reading assistant for Japanese and Hebrew. A React frontend
> talks to a FastAPI backend. The tutor chat is an agentic RAG pipeline built with
> LangGraph: the model decides whether to query a Chroma vector store of JLPT/CEFR
> vocabulary, a temperature-zero grader checks the retrieved rows, and a capped
> query-rewrite loop recovers from bad retrievals. Text annotation uses SudachiPy for
> Japanese morphology with a greedy longest-match against the JLPT lists, plus a
> frequency-list fallback; Hebrew uses Unicode normalization against a CEFR list. The
> flagship feature streams an LLM rewrite of any text at a chosen difficulty level over
> NDJSON, then re-annotates the result so you can verify it actually got easier. Chat
> runs on a local Ollama model or Groq via a toggle — but embeddings are pinned to one
> local model, because the persisted vectors depend on it."
