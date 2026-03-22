from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidences: List[str]
    category: Optional[str] = None


@dataclass
class Turn:
    dia_id: str
    role: str
    content: str


@dataclass
class Session:
    session_id: str
    date_time: str
    turns: List[Turn]


@dataclass
class Conversation:
    user: str
    sessions: Dict[str, Session]


@dataclass
class Sample:
    """A single sample from the dataset"""
    sample_id: str
    qa: List[QA]
    conversations: List[Conversation]
