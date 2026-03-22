import os
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from himem.embeddings.configs import EmbedderConfig
from himem.llms.configs import LlmConfig
from himem.vector_stores.configs import VectorStoreConfig
from himem.configs.rerankers.config import RerankerConfig


class MemoryItem(BaseModel):
    id: str = Field(..., description="The unique identifier for the text data")
    content: str = Field(
        ..., description="The memory deduced from the text data"
    )  # TODO After prompt changes from platform, update this
    hash: Optional[str] = Field(None, description="The hash of the memory")
    # The metadata value can be anything and not just string. Fix it
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the text data")
    score: Optional[float] = Field(None, description="The score associated with the text data")
    created_at: Optional[str] = Field(None, description="The timestamp when the memory was created")
    updated_at: Optional[str] = Field(None, description="The timestamp when the memory was updated")


class Component(BaseModel):
    config: dict


class MemoryConfig(BaseModel):
    llm_providers: Dict[str, LlmConfig] = Field(
        description="Configuration for the language model",
    )
    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Configuration for the embedding model",
        default_factory=EmbedderConfig,
    )
    reranker: Optional[RerankerConfig] = Field(
        description="Configuration for the reranker",
        default=None,
    )
    custom_fact_extraction_prompt: Optional[str] = Field(
        description="Custom prompt for the fact extraction",
        default=None,
    )
    custom_update_memory_prompt: Optional[str] = Field(
        description="Custom prompt for the update memory",
        default=None,
    )
    components: Dict[str, Component]
