# Language Assistant

A Japanese vocabulary assistant with a FastAPI backend, LangGraph RAG system, and React/Vite frontend.

## Run the backend

```bash
uv run uvicorn backend.main:app --reload
```

The API runs at <http://localhost:8000>.

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at <http://localhost:5173> and talks to the backend at `http://localhost:8000`.

To use another API URL, create `frontend/.env.local`:

```bash
VITE_API_BASE_URL=http://localhost:8000
```
