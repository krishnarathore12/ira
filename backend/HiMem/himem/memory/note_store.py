import hashlib
import json
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Optional

import pytz
from loguru import logger

from himem.components.extractor import Extractor
from himem.components.knowledge_conflict_detector import KnowledgeConflictDetector
from himem.configs.base import MemoryItem
from himem.dataset.model import Session
from himem.utils.factory import VectorStoreFactory, EmbedderFactory, LlmFactory


class NoteStore:
    def __init__(self, config):
        self.config = config

        current_config = self.config.components['note_memory'].config
        self.enabled_llm_provider = current_config["llm_provider"]
        self.llm_config = self.config.llm_providers[self.enabled_llm_provider].config
        self.llm = LlmFactory.create(self.enabled_llm_provider, self.llm_config)

        self.extractor = Extractor(config)

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
        )

        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.collection_name = self.config.vector_store.config.collection_name

        self.knowledge_conflict_detector = KnowledgeConflictDetector(config)

    def add(self, session: Session, metadata, filters=None, enable_knowledge_alignment=False):
        user_id = metadata['user_id']
        timestamp = metadata['timestamp']

        try:
            new_extracted_notes = self.extractor.extract_notes(user_id, session, enable_knowledge_alignment)
        except Exception as e:
            logger.error(f"Failed to extract notes: {e}")
            raise e

        unique_data = {}
        for item in new_extracted_notes:
            if item['category'] is None:
                item['category'] = 'Fact'
            unique_data[item["content"]] = item
        new_extracted_notes = list(unique_data.values())

        if not new_extracted_notes:
            logger.info("No new facts retrieved from input.")
            return []

        logger.debug(f"New extracted notes: {new_extracted_notes}")

        new_extracted_user_related_notes = []
        new_extracted_event_related_notes = []
        for note in new_extracted_notes:
            if note['category'] == "Event" or note['category'] == "Fact":
                new_extracted_event_related_notes.append(note)
            else:
                new_extracted_user_related_notes.append(note)
        logger.debug(f"New extracted notes: {len(new_extracted_notes)}")

        returned_memories = []
        for idx, note in enumerate(new_extracted_event_related_notes):
            logger.debug("-------------------------------")
            logger.info(f"New extracted event note: {note}")
            try:
                category = note.get("category")
                content = note.get("content")

                processed_metadata = deepcopy(metadata)
                processed_metadata['timestamp'] = note.get("timestamp", timestamp)
                processed_metadata['category'] = category

                memory_id = self._create_memory(
                    data=content,
                    metadata=processed_metadata,
                )
                record = {"id": memory_id, "memory": content,
                          'category': category,
                          "event": "ADD"}
                returned_memories.append(record)
                logger.info(f"memory in action: {record}")
            except Exception as e:
                logger.error(f"Error in new_retrieved_facts: {e}")

        retrieved_previous_memories = self._retrieve_previous_memory(new_extracted_user_related_notes, filters)
        new_memories_with_actions, temp_uuid_mapping = self.knowledge_conflict_detector.resolve(new_extracted_user_related_notes, retrieved_previous_memories)
        logger.debug(f"New memory with actions: {len(new_memories_with_actions)}")
        try:
            result = self._handle_action(new_memories_with_actions, temp_uuid_mapping, metadata)
        except Exception as e:
            logger.warning(f"failed to do action: {e}, retrieved_previous_memories: {retrieved_previous_memories}")
            raise e
        if not result:
            returned_memories.extend(result)
        return returned_memories

    def update(self, new_infos: list, metadata, filters=None):
        retrieved_previous_memories = self._retrieve_previous_memory(new_infos, filters)
        new_memories_with_actions, temp_uuid_mapping = self.knowledge_conflict_detector.resolve(new_infos, retrieved_previous_memories)
        logger.debug(f"New memory with actions: {len(new_memories_with_actions)}")
        try:
            result = self._handle_action(new_memories_with_actions, temp_uuid_mapping, metadata)
        except Exception as e:
            logger.warning(f"failed to do action: {e}, new_memories_with_actions: {new_memories_with_actions}")
            return []
        return result

    def search(self, query, filters, limit: int = 10, threshold: Optional[float] = None, enable_hybrid: bool = False):
        embeddings = self.embedding_model.encode(query)
        if enable_hybrid:
            memories = self.vector_store.hybrid_search(query=query, query_embeddings=embeddings, limit=limit, filters=filters)
        else:
            memories = self.vector_store.search(query=query, query_embeddings=embeddings, limit=limit, filters=filters)

        promoted_payload_keys = [
            "user_id",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                content=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
                metadata=mem.payload.get("metadata"),
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    def _retrieve_previous_memory(self, new_extracted_notes, filters):
        retrieved_old_memory = []
        search_filters = {}
        if filters.get("user_id"):
            search_filters["user_id"] = filters["user_id"]
        for note in new_extracted_notes:
            new_mem = json.dumps(note)
            messages_embeddings = self.embedding_model.encode(new_mem)
            existing_memories = self.vector_store.search(
                query=new_mem,
                query_embeddings=messages_embeddings,
                limit=5,
                filters=search_filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "content": mem.payload.get("data", "")})
        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())

        return retrieved_old_memory

    def _handle_action(self, new_memories_with_actions, temp_uuid_mapping, metadata):
        returned_memories = []
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(f"memories with action: {resp}")
                try:
                    category = resp.get("category")
                    action_text = resp.get("text")
                    if not action_text:
                        logger.warning("Skipping memory entry because of empty `text` field.")
                        continue

                    event_type = resp.get("event")
                    processed_metadata = deepcopy(metadata)
                    processed_metadata['category'] = category

                    memory_id = temp_uuid_mapping.get(resp.get("old_memory"), None)
                    if not memory_id and event_type == "UPDATE":
                        logger.warning(f"Skipping updating memory entry because of lack of memory_id: {resp}")
                        event_type = "ADD"
                    elif not memory_id and event_type == "DELETE":
                        logger.warning(f"Skipping deleting memory entry because of lack of memory_id: {resp}")
                        event_type = "NONE"

                    record = {}
                    if event_type == "ADD":
                        memory_id = self._create_memory(
                            data=action_text,
                            metadata=processed_metadata,
                        )
                        record = {"id": memory_id, "memory": action_text, 'category': category, "event": event_type}
                        returned_memories.append(record)
                    elif event_type == "UPDATE":
                        self._update_memory(
                            memory_id=memory_id,
                            data=action_text,
                            metadata=processed_metadata,
                        )
                        record = {
                            "id": temp_uuid_mapping[resp.get("old_memory")],
                            "memory": action_text,
                            'category': category,
                            "event": event_type,
                            "previous_memory": resp.get("old_memory"),
                        }
                        returned_memories.append(record)
                    elif event_type == "DELETE":
                        self._delete_memory(memory_id=temp_uuid_mapping[resp.get("old_memory")])
                        record = {
                            "id": temp_uuid_mapping[resp.get("old_memory")], "memory": action_text, 'category': category, "event": event_type,
                        }
                        returned_memories.append(record)
                    elif event_type == "NONE":
                        record = {"memory": action_text, 'category': category, "event": event_type, }
                    logger.info(f"memory in action: {record}")
                except Exception as e:
                    logger.error(f"Error processing memory action: {resp}, Error: {e}, temp_uuid_mapping: {temp_uuid_mapping}, resp: {resp}")
        except Exception as e:
            logger.error(f"Error iterating new_memories_with_actions: {e}")
        return returned_memories

    def _create_memory(self, data, metadata=None):
        embeddings = self.embedding_model.encode(data)
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        try:
            self.vector_store.insert(
                docs=[data],
                embeddings=[embeddings],
                ids=[memory_id],
                payloads=[metadata],
            )
        except Exception as e:
            logger.error(f"Error creating memory {data}, metadata:{metadata}, Error: {e}")
            raise e
        return memory_id

    def _update_memory(self, memory_id, data, existing_embeddings=None, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" not in new_metadata and "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if existing_embeddings:
            embeddings = existing_embeddings
        else:
            embeddings = self.embedding_model.encode(data)
        self.vector_store.update(
            vector_id=memory_id,
            doc=data,
            embeddings=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        return memory_id

    def _delete_memory(self, memory_id):
        existing_memory = self.vector_store.get(vector_id=memory_id)
        logger.info(f"Deleting memory with {memory_id}: {existing_memory}")
        self.vector_store.delete(vector_id=memory_id)
        return memory_id
