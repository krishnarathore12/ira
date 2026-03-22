# IRA backend (HiMem + FastAPI)

## Prerequisites

- Python 3.11 or 3.12
- Docker (for Qdrant and OpenSearch)
- OpenAI API key

## 1. Start memory stores

From the **repository root**:

```bash
docker compose up -d
```

This starts Qdrant on `6333` and OpenSearch on `9200`. OpenSearch may take ~30–60 seconds before it accepts requests. Compose uses `DISABLE_SECURITY_PLUGIN=true` and `DISABLE_INSTALL_DEMO_CONFIG=true` so the node starts without the demo installer (which otherwise requires `OPENSEARCH_INITIAL_ADMIN_PASSWORD` and matches HiMem’s HTTP client without auth).

If OpenSearch was stuck **Restarting** with old settings, run `docker compose down`, remove the bad data once with `docker volume rm ira_opensearch_data` (or `docker compose down -v` to drop all project volumes), then `docker compose up -d` again.

If nothing is listening on `9200`, chat still works using **note memory (Qdrant)** only; episode memory is skipped until OpenSearch is up.

Optional env: `OPENSEARCH_HOST` (default `localhost`), `OPENSEARCH_PORT` (default `9200`).

## 2. Configure environment

Create `backend/.env`:

```env
OPENAI_API_KEY=sk-...
# Optional overrides:
# OPENAI_MODEL=gpt-4o-mini
# CHAT_MODEL=gpt-4o-mini
# QDRANT_HOST=localhost
# QDRANT_PORT=6333
# HIMEM_COLLECTION=ira_companion_notes
# HIMEM_EPISODE_INDEX_PREFIX=ira
```

HiMem reads `backend/HiMem/config/base.yaml` by default. Instruction file paths are rewritten automatically to absolute paths under `HiMem/`. To use another file, set `HIMEM_CONFIG_PATH`.

## 3. Install and run the API

```bash
cd backend
uv sync
export PYTHONPATH=HiMem
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The first chat request can be slow (downloading embedding models, etc.). Reranking is disabled in the API config loader for faster startup.

## Endpoints

- `GET /api/health` — liveness
- `POST /api/chat` — JSON body `{ "user_id": "default", "messages": [{ "role": "user"|"assistant", "content": "..." }] }` (last message must be `user`)

## CLI scratch script

`main.py` in this folder is a standalone OpenAI REPL and is not used by the FastAPI app.
