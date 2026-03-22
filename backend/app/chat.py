"""Multi-turn chat: retrieve HiMem context, call OpenAI, persist exchange."""

from __future__ import annotations

import os
import uuid
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from openai import OpenAI

from app.memory_service import get_memory
from himem.dataset.model import Session as HiMemSession, Turn
from app.database import SessionLocal, ChatSession, ChatMessage as DBMessage

MAX_MESSAGES = 40
MEMORY_SEARCH_LIMIT = 8
CONTENT_PREVIEW_MAX = 2000
COMPANION_SYSTEM = """You are Ira, a warm and attentive Indian female AI companion. You always speak in Hinglish (a natural mix of Hindi and English) to keep the conversation friendly, emotional, and relatable. You remember what the user shares across conversations \
when relevant memory is provided below. Use memories naturally; do not fabricate details. If memory is empty, \
respond helpfully in Hinglish without claiming to recall past chats."""


def _format_memory_block(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(No matching long-term memories yet.)"
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        ts = r.get("metadata", {}).get("timestamp") or r.get("timestamp", "")
        content = (r.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{i}. [{ts}] {content}")
    return "\n".join(lines) if lines else "(No matching long-term memories yet.)"


def _hit_is_episode(r: dict[str, Any]) -> bool:
    topic = r.get("topic")
    summary = r.get("topic_summary")
    if topic is not None and str(topic).strip():
        return True
    if summary is not None and str(summary).strip():
        return True
    return False


def _serialize_retrieved_hit(r: dict[str, Any]) -> dict[str, Any]:
    source = "episode" if _hit_is_episode(r) else "note"
    content = (r.get("content") or "").strip()
    if len(content) > CONTENT_PREVIEW_MAX:
        content = content[:CONTENT_PREVIEW_MAX] + "…"
    ts = r.get("timestamp")
    if ts is None and isinstance(r.get("metadata"), dict):
        ts = r["metadata"].get("timestamp")
    out: dict[str, Any] = {
        "source": source,
        "id": r.get("id"),
        "content": content,
        "score": r.get("score"),
        "timestamp": ts,
    }
    if source == "episode":
        out["topic"] = r.get("topic")
        out["topic_summary"] = r.get("topic_summary")
    return out


def _parse_add_trace(add_return: dict[str, Any], session_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = add_return.get(session_id) or []
    note_writes: list[dict[str, Any]] = []
    episode_writes: list[dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        if "_ira_episode_writes" in r:
            episode_writes = list(r["_ira_episode_writes"])
            continue
        if "memory" in r and "event" in r:
            note_writes.append(
                {
                    "id": r.get("id"),
                    "memory": r.get("memory"),
                    "category": r.get("category"),
                    "event": r.get("event"),
                }
            )
    return note_writes, episode_writes


def _save_session_to_himem(user_id_str: str, active_session: ChatSession, db: Any) -> tuple[list, list]:
    memory = get_memory()
    messages = db.query(DBMessage).filter(DBMessage.chat_session_id == active_session.id).order_by(DBMessage.timestamp.asc()).all()
    
    if not messages:
        return [], []
        
    turns = []
    for m in messages:
        turns.append(Turn(dia_id=f"{active_session.session_id}_{m.id}", role=m.role, content=m.content))
        
    dt_str = active_session.last_activity.replace(tzinfo=timezone.utc).isoformat()
    
    himem_session = HiMemSession(
        session_id=active_session.session_id,
        date_time=dt_str,
        turns=turns
    )
    
    try:
        add_return = memory.add(user_id_str, himem_session, construction_mode="all", enable_knowledge_alignment=False)
        print(f"Successfully batch saved session {active_session.session_id}")
        return _parse_add_trace(add_return, active_session.session_id)
    except Exception as e:
        print(f"Failed to batch save memory: {e}")
        return [], []

def force_save_session(user_id_str: str) -> dict | None:
    db = SessionLocal()
    try:
        active_session = db.query(ChatSession).filter(
            ChatSession.user_id == int(user_id_str)
        ).order_by(ChatSession.last_activity.desc()).first()
        
        if active_session:
            note_writes, episode_writes = _save_session_to_himem(user_id_str, active_session, db)
            now = datetime.utcnow()
            new_session = ChatSession(user_id=int(user_id_str), session_id=str(uuid.uuid4()), last_activity=now)
            db.add(new_session)
            db.commit()
            return {
                "query": "(Manual session batch save)",
                "retrieved": [],
                "note_writes": note_writes,
                "episode_writes": episode_writes,
            }
        return None
    finally:
        db.close()

def get_active_session_history(user_id_str: str) -> list[dict]:
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).filter(
            ChatSession.user_id == int(user_id_str)
        ).all()
        
        if not sessions:
            return []
            
        session_ids = [s.id for s in sessions]
        messages = db.query(DBMessage).filter(DBMessage.chat_session_id.in_(session_ids)).order_by(DBMessage.timestamp.asc()).all()
        
        res = []
        for m in messages:
            msg = {
                "id": m.id, 
                "role": m.role, 
                "text": m.content, 
                "timestamp": m.timestamp.isoformat() + "Z" if m.timestamp else None
            }
            if m.memory_trace:
                try:
                    msg["memoryTrace"] = json.loads(m.memory_trace)
                except Exception:
                    pass
            res.append(msg)
        return res
    finally:
        db.close()



@dataclass
class ChatResult:
    assistant_text: str
    memory_trace: dict[str, Any]
    timestamp: str


def run_chat(*, user_id: str, messages: list[dict[str, str]]) -> ChatResult:
    if not messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")

    last_user: str | None = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    if last_user is None:
        raise HTTPException(status_code=422, detail="last message must be from role 'user'")

    memory = get_memory()
    try:
        found = memory.search(
            last_user,
            user_id=user_id,
            mode="hybrid",
            limit=MEMORY_SEARCH_LIMIT,
            rerank=False,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"memory search failed: {e}") from e

    results = found.get("results") or []
    retrieved = [_serialize_retrieved_hit(r) for r in results if isinstance(r, dict)]
    memory_block = _format_memory_block(results)

    current_time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    system_with_memory = (
        f"{COMPANION_SYSTEM}\n\nCurrent time: {current_time_str}\n\n"
        f"--- Relevant long-term memory ---\n{memory_block}\n--- End memory ---"
    )

    trimmed = messages[-MAX_MESSAGES:]
    api_messages: list[dict[str, str]] = [{"role": "system", "content": system_with_memory}]
    for m in trimmed:
        role = m.get("role", "")
        content = m.get("content", "")
        if role not in ("user", "assistant", "system"):
            continue
        if role == "system":
            continue
        api_messages.append({"role": role, "content": content})

    client = OpenAI()
    model = os.environ.get("CHAT_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    try:
        resp = client.chat.completions.create(model=model, messages=api_messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}") from e

    choice = resp.choices[0].message
    assistant_text = choice.content or ""
    if not assistant_text.strip():
        raise HTTPException(status_code=502, detail="empty model response")

    now = datetime.utcnow()
    db = SessionLocal()
    trace = {
        "query": last_user,
        "retrieved": retrieved,
        "note_writes": [],
        "episode_writes": [],
    }
    
    try:
        active_session = db.query(ChatSession).filter(
            ChatSession.user_id == int(user_id)
        ).order_by(ChatSession.last_activity.desc()).first()
        
        active_session_id = None
        
        if active_session:
            delta = now - active_session.last_activity
            if delta.total_seconds() > 30 * 60: # 30 minutes
                _save_session_to_himem(user_id, active_session, db)
                
                session_id_to_use = str(uuid.uuid4())
                new_session = ChatSession(user_id=int(user_id), session_id=session_id_to_use, last_activity=now)
                db.add(new_session)
                db.commit()
                db.refresh(new_session)
                active_session_id = new_session.id
            else:
                active_session.last_activity = now
                db.commit()
                active_session_id = active_session.id
        else:
            session_id_to_use = str(uuid.uuid4())
            new_session = ChatSession(user_id=int(user_id), session_id=session_id_to_use, last_activity=now)
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            active_session_id = new_session.id

        user_msg = DBMessage(chat_session_id=active_session_id, role="user", content=last_user, timestamp=now)
        asst_msg = DBMessage(
            chat_session_id=active_session_id, 
            role="assistant", 
            content=assistant_text, 
            timestamp=now,
            memory_trace=json.dumps(trace)
        )
        db.add(user_msg)
        db.add(asst_msg)
        db.commit()
        
    finally:
        db.close()

    return ChatResult(assistant_text=assistant_text, memory_trace=trace, timestamp=now.isoformat() + "Z")
