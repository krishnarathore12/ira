# Framework structure partially adapted from Mem0 (Apache 2.0 License)
import asyncio
import threading
from concurrent import futures
from typing import Any, Dict, Optional, Awaitable, TypeVar

from loguru import logger
from pydantic import ValidationError

from himem.components.extractor import Extractor
from himem.components.reviewer import Reviewer
from himem.components.segmentor import Segmentor
from himem.configs.base import MemoryConfig
from himem.dataset.model import Session
from himem.exceptions import ValidationError
from himem.memory.episode_store import EpisodeStore
from himem.memory.note_store import NoteStore
from himem.memory.utils import (
    _build_filters_and_metadata, _has_advanced_operators, _process_metadata_filters,
)
from himem.utils.factory import (
    EmbedderFactory,
    LlmFactory,
    RerankerFactory,
)

T = TypeVar("T")


class Memory:
    def __init__(self, config: MemoryConfig):
        self.config = config

        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
        )

        memory_config = self.config.components['memory'].config
        self.enabled_llm_provider = memory_config["llm_provider"]
        self.llm_config = self.config.llm_providers[self.enabled_llm_provider].config
        self.llm = LlmFactory.create(self.enabled_llm_provider, self.llm_config)

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config
            )

        # Initialize components
        self.event_segmentor = Segmentor(config)
        self.reviewer = Reviewer(config)
        self.extractor = Extractor(config)

        # Initialize memories
        self.note_store = NoteStore(config)
        self.episode_store = EpisodeStore(config)

        self.enable_self_evolution = memory_config.get("enable_self_evolution", False)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._start_background_loop()

    def _start_background_loop(self):
        if self._loop is not None:
            return

        def loop_runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = threading.Thread(target=loop_runner, daemon=True, name="Graphiti-Background-Loop")
        self._thread.start()

    def run_sync(self, coro: Awaitable[T]) -> T:
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("Background loop not running")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()  # blocks only the calling thread

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def add(self, user_id, session: Session,
            metadata: Optional[Dict[str, Any]] = None,
            construction_mode: str = "all",
            enable_knowledge_alignment=False,
            ):
        """
        Searches for memories based on a query
        Args:
            user_id (str): User ID.
            session (Session): Session to use.
            metadata (str, optional): Metadata about the data. Defaults to None.
            construction_mode (str, optional): Construction mode. Defaults to "all".
                - all
                - episode
                - note
            enable_knowledge_alignment (int, optional): Whether to use knowledge alignment or not. Defaults to False.
        """
        records = []

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            input_metadata=metadata,
        )
        processed_metadata["timestamp"] = session.date_time

        logger.debug(
            f"user_id: {user_id}, current session:{session.session_id}, construction_mode: {construction_mode}, enable_knowledge_alignment: {enable_knowledge_alignment}")

        if construction_mode == "all" or construction_mode == "note":
            record = self.note_store.add(session, processed_metadata, effective_filters, enable_knowledge_alignment)
            records.extend(record)

        if construction_mode == "all" or construction_mode == "episode":
            # Segment all sessions of the current conversation, including topic inference and keywords extraction
            segments = self.event_segmentor.segment([session])
            episode_trace: list[Dict[str, Any]] = []
            for idx, segment in enumerate(segments):
                if segment is None:
                    logger.warning(f"Segment {idx} was not found in the segments list")
                    continue
                processed_metadata["segment_id"] = f"{session.session_id}_{idx}"
                with futures.ThreadPoolExecutor() as executor:
                    future1 = executor.submit(self.episode_store.add, segment, processed_metadata, session, enable_knowledge_alignment)
                    add_futures = [future1, ]
                    futures.wait(add_futures)

                    doc_id = None
                    for future in add_futures:
                        doc_id = future.result()
                    episode_trace.append(
                        {
                            "topic": segment.get("topic"),
                            "topic_summary": segment.get("topic_summary"),
                            "document_id": doc_id,
                        }
                    )
            records.append({"_ira_episode_writes": episode_trace})

        return {session.session_id: records}

    def search(
            self,
            query: str,
            *,
            user_id: Optional[str] = None,
            mode: Optional[str] = 'hybrid',
            limit: int = 10,
            filters: Optional[Dict[str, Any]] = None,
            threshold: Optional[float] = None,
            rerank: bool = True,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            mode (str, optional): ID of the run to search for. Defaults to hybrid.
                - hybrid
                - note
                - episode
                - best-effort
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None.
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.
            rerank (boolean, optional): Whether to rerank the results. Defaults to True.

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key
        """
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        # Apply enhanced metadata filtering if advanced operators are detected
        if filters and _has_advanced_operators(filters):
            processed_filters = _process_metadata_filters(filters)
            effective_filters.update(processed_filters)
        elif filters:
            # Simple filters, merge directly
            effective_filters.update(filters)

        if mode == "best-effort":
            for key in ['note', 'episode']:
                results = self._search(query, effective_filters, limit, key, rerank, threshold)
                context = ""
                for idx, result in enumerate(results):
                    timestamp = result['metadata'].get('timestamp', "")
                    if not timestamp:
                        logger.warning(f"No timestamp found for key {key} in result {result}")
                    content = result['content']
                    content = content.replace('user:', user_id + ":").strip()
                    context += f"---------Evidence {idx + 1}----------\n"
                    context += f"timestamp: {timestamp}\n"
                    context += f"conversation: \n{content}\n"
                is_correct = self.reviewer.evaluate(query, context)
                if is_correct == 1:
                    break
            if is_correct == 1 and key == 'episode' and self.enable_self_evolution:
                logger.debug("Memory self-evolution enabled. Processing the missed information if possible.")
                users_data = {}
                for idx, result in enumerate(results):
                    if user_id not in users_data:
                        users_data[user_id] = [result]
                    else:
                        users_data[user_id].append(result)
                for user_id, user_data in users_data.items():
                    context = ""
                    for idx, result in enumerate(user_data):
                        timestamp = result['metadata']['timestamp']
                        content = result['content']
                        content = content.replace('user:', user_id + ":").strip()
                        context += f"---------Evidence {idx + 1}----------\n"
                        context += f"timestamp: {timestamp}\n"
                        context += f"conversation: \n{content}\n"
                    # Extract the knowledge related to the question
                    new_infos = self.extractor.knowledge_extraction(query, context)
                    for new_info in new_infos:
                        if "metadata" not in new_info:
                            new_info["metadata"] = {}
                        new_info['metadata']['user_id'] = user_id
                        new_info['metadata']['timestamp'] = new_info.get('timestamp')
                    # Send the knowledge to the note memory for knowledge conflict detection and updating
                    metadata = {'user_id': user_id}
                    filters = {'user_id': user_id}
                    result = self.note_store.update(new_infos, metadata, filters)
                    logger.debug("===========Found missed info========")
                    logger.debug(f"context: \n{context}\nnew_infos: \n{new_infos}\nresult: \n{result}")
            return results
        # Default mode, hybrid search including note and episode
        elif mode == "hybrid":
            results = []
            with futures.ThreadPoolExecutor(max_workers=5) as executor:
                f1 = executor.submit(self._search, query, effective_filters, limit, "note", rerank, threshold)
                f2 = executor.submit(self._search, query, effective_filters, limit, "episode", rerank, threshold)
                results.extend(f1.result())
                results.extend(f2.result())
        else:
            results = self._search(query, effective_filters, limit, mode, rerank, threshold)

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and results:
            try:
                reranked_memories = self.reranker.rerank(query, results, limit)
                results = []
                for reranked_memory in reranked_memories:
                    if (not threshold) or (threshold and reranked_memory["rerank_score"] >= threshold):
                        results.append(reranked_memory)
            except Exception as e:
                logger.warning(f"Reranking failed, {e}, using original results: {results}")

        return {"results": results[:limit]}

    def _search(self, query, filters, limit: int, mode, rerank: bool = False, threshold: float = None):
        if mode == 'episode':
            results = self.episode_store.search(query, filters, limit)
        else:
            results = self.note_store.search(query, filters, limit)
        return results
