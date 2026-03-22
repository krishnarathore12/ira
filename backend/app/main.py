"""FastAPI entry: HiMem-backed companion chat."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_BACKEND = Path(__file__).resolve().parent.parent
_HIMEM = _BACKEND / "HiMem"
if str(_HIMEM) not in sys.path:
    sys.path.insert(0, str(_HIMEM))

load_dotenv(_BACKEND / ".env")

from app.chat import run_chat  # noqa: E402
from app.auth import auth_router, get_current_user
from app.database import User
from app.memory_service import get_memory
from fastapi import Depends


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    user_id: Optional[str] = Field(default="default", min_length=1)
    messages: list[ChatMessage]


class RetrievedMemoryItem(BaseModel):
    model_config = {"extra": "ignore"}

    source: str
    id: Optional[str] = None
    content: str = ""
    score: Optional[float] = None
    timestamp: Optional[str] = None
    topic: Optional[str] = None
    topic_summary: Optional[str] = None


class NoteWriteItem(BaseModel):
    model_config = {"extra": "ignore"}

    id: Optional[str] = None
    memory: Optional[str] = None
    category: Optional[str] = None
    event: Optional[str] = None


class EpisodeWriteItem(BaseModel):
    model_config = {"extra": "ignore"}

    topic: Optional[str] = None
    topic_summary: Optional[str] = None
    document_id: Optional[str] = None


class MemoryTrace(BaseModel):
    model_config = {"extra": "ignore"}

    query: str
    retrieved: list[RetrievedMemoryItem]
    note_writes: list[NoteWriteItem]
    episode_writes: list[EpisodeWriteItem]


class ChatResponse(BaseModel):
    message: ChatMessage
    memory_trace: MemoryTrace


def _memory_trace_from_dict(d: dict[str, Any]) -> MemoryTrace:
    return MemoryTrace(
        query=d.get("query", ""),
        retrieved=[RetrievedMemoryItem.model_validate(x) for x in d.get("retrieved", [])],
        note_writes=[NoteWriteItem.model_validate(x) for x in d.get("note_writes", [])],
        episode_writes=[EpisodeWriteItem.model_validate(x) for x in d.get("episode_writes", [])],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pre-loading HiMem ML models...")
    get_memory()  # Initializes the heavy sentence-transformers on boot
    print("Models loaded successfully!")
    yield

app = FastAPI(title="IRA Companion API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest, current_user: User = Depends(get_current_user)) -> ChatResponse:
    msgs = [{"role": m.role, "content": m.content} for m in body.messages]
    result = run_chat(user_id=str(current_user.id), messages=msgs)
    return ChatResponse(
        message=ChatMessage(role="assistant", content=result.assistant_text, timestamp=result.timestamp),
        memory_trace=_memory_trace_from_dict(result.memory_trace),
    )

from app.chat import force_save_session, get_active_session_history

@app.get("/api/chat/history")
def get_history(current_user: User = Depends(get_current_user)) -> list[dict]:
    return get_active_session_history(str(current_user.id))

@app.post("/api/chat/save")
def save_chat(current_user: User = Depends(get_current_user)) -> dict:
    trace = force_save_session(str(current_user.id))
    if trace:
        return {"status": "success", "detail": "Session manually saved to memory.", "memory_trace": _memory_trace_from_dict(trace).model_dump()}
    return {"status": "ignored", "detail": "No active session found or empty."}
