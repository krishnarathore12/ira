"""Load HiMem YAML, resolve instruction paths, apply env overrides."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

BACKEND_ROOT = Path(__file__).resolve().parent.parent
HIMEM_ROOT = BACKEND_ROOT / "HiMem"
DEFAULT_BASE_CONFIG = HIMEM_ROOT / "config" / "base.yaml"


def _resolve_path_strings(obj: Any, root: Path) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, str) and (
                key.endswith("_path") or key in ("prompt_path", "topic_recommendation_prompt_path")
            ):
                p = Path(val)
                if not p.is_absolute():
                    obj[key] = str((root / val).resolve())
            else:
                _resolve_path_strings(val, root)
    elif isinstance(obj, list):
        for item in obj:
            _resolve_path_strings(item, root)


def load_himem_config_dict() -> dict[str, Any]:
    path = Path(os.environ.get("HIMEM_CONFIG_PATH", DEFAULT_BASE_CONFIG))
    if not path.is_file():
        raise FileNotFoundError(f"HiMem config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("HiMem config must be a YAML mapping")

    cfg = deepcopy(cfg)
    _resolve_path_strings(cfg, HIMEM_ROOT)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for HiMem LLMs")

    chat_default_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    for provider in ("openai", "qwen", "ollama"):
        block = cfg.get("llm_providers", {}).get(provider)
        if isinstance(block, dict) and isinstance(block.get("config"), dict):
            conf = block["config"]
            if provider in ("openai", "qwen"):
                if conf.get("api_key") in (None, "", "YOUR_API_KEY"):
                    conf["api_key"] = api_key
            if provider == "openai":
                if conf.get("model") in (None, "", "YOUR_MODEL_NAME"):
                    conf["model"] = chat_default_model

    vs = cfg.setdefault("vector_store", {})
    vsc = vs.setdefault("config", {})
    vsc["host"] = os.environ.get("QDRANT_HOST", vsc.get("host", "localhost"))
    vsc["port"] = int(os.environ.get("QDRANT_PORT", str(vsc.get("port", 6333))))
    vsc["collection_name"] = os.environ.get(
        "HIMEM_COLLECTION", vsc.get("collection_name", "ira_companion_notes")
    )
    embed_dims = vsc.get("embedding_model_dims", 768)
    emb = cfg.setdefault("embedder", {})
    embc = emb.setdefault("config", {})
    embc["embedding_dims"] = int(
        os.environ.get("HIMEM_EMBEDDING_DIMS", str(embc.get("embedding_dims", embed_dims)))
    )
    vsc["embedding_model_dims"] = embc["embedding_dims"]

    ep = cfg.setdefault("components", {}).setdefault("episode_memory", {}).setdefault("config", {})
    ep["index_prefix"] = os.environ.get("HIMEM_EPISODE_INDEX_PREFIX", ep.get("index_prefix", "ira"))

    seg = cfg.setdefault("components", {}).setdefault("segmentor", {}).setdefault("config", {})
    seg["collection_name"] = os.environ.get("HIMEM_TOPICS_COLLECTION", seg.get("collection_name", "topics"))

    cfg.pop("reranker", None)

    return cfg
