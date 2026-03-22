import json
import uuid
from copy import deepcopy
from typing import List

from loguru import logger

from himem.utils.base import prefix_exchanges_with_idx, create_llm_instance_from_config
from himem.utils.factory import EmbedderFactory, VectorStoreFactory


class Segmentor:
    def __init__(self, config):
        self.config = config.components['segmentor'].config
        enabled_llm_provider = self.config.get('llm_provider')
        self.llm = create_llm_instance_from_config(config, enabled_llm_provider)

        prompt_path = self.config.get('prompt_path')
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.segment_prompt = f.read()

        self.embedding_model = EmbedderFactory.create(
            config.embedder.provider,
            config.embedder.config,
        )
        self.embedding_dims = config.embedder.config.get("embedding_dims", 768)
        self.collection_name = self.config.get('collection_name')
        dup_vec_config = deepcopy(config.vector_store.config)
        dup_vec_config.collection_name = self.collection_name

        self.vector_store = VectorStoreFactory.create(
            config.vector_store.provider, dup_vec_config,
        )
        topic_recommendation_prompt_path = self.config.get('topic_recommendation_prompt_path')
        with open(topic_recommendation_prompt_path, "r", encoding="utf-8") as f:
            self.topic_recommendation_prompt = f.read()
        self.top_k = 5

        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "segmentation_output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "segment_id": {"type": "integer"},
                                    "start_exchange_number": {"type": "integer"},
                                    "end_exchange_number": {"type": "integer"},
                                    "num_exchanges": {"type": "integer"},
                                    "topic": {"type": "string"},
                                    "topic_summary": {"type": "string"}
                                },
                                "required": [
                                    "segment_id",
                                    "start_exchange_number",
                                    "end_exchange_number",
                                    "num_exchanges",
                                    "topic",
                                    "topic_summary"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["segments"],
                    "additionalProperties": False
                }
            }
        }

    def segment(self, sessions, ) -> List[dict]:
        segments = []
        for session_idx, session in enumerate(sessions):
            timestamp = session.date_time
            turns = session.turns
            exchanges_str_with_idx = prefix_exchanges_with_idx(turns)

            prompt = self.segment_prompt.format(
                text_to_be_segmented=exchanges_str_with_idx
            )
            response, _ = self.llm.generate_response([{'role': 'user', 'content': prompt}],
                                                     response_format=self.response_format)
            response = json.loads(response)
            lines = response["segments"]
            total_num_of_turns = len(turns)

            segmentations = []
            prev_idx = 0
            for line in lines:
                n_ex = int(line["num_exchanges"])
                topic = line["topic"].lower()
                topic = self.advise(topic)
                if (prev_idx + n_ex) >= total_num_of_turns:
                    if prev_idx == total_num_of_turns:
                        exchanges = turns[prev_idx - 1:]
                    else:
                        exchanges = turns[prev_idx:]
                else:
                    exchanges = turns[prev_idx: prev_idx + n_ex]
                if exchanges:
                    segmentation = {'id': str(uuid.uuid4()),
                                    'timestamp': timestamp,
                                    'exchanges': exchanges,
                                    'topic': topic, 'topic_summary': line["topic_summary"]}
                    segmentations.append(segmentation)
                    prev_idx = prev_idx + n_ex
                else:
                    cause = f"found empty exchanges, prev_idx={prev_idx}, n_ex={n_ex}, total={len(turns)}\nline={line}\nexchanges: {exchanges}\nexchanges_str_with_idx: {exchanges_str_with_idx}"
                    logger.warning(cause)
                    raise ValueError(cause)
            logger.debug(f"{session_idx}-th session is segmented to {len(segmentations)} segments")
            segments.extend(segmentations)

        return segments

    def advise(self, topic: str) -> str:
        topic_embedding = self.embedding_model.encode(topic)
        existing_topics = self.vector_store.search(
            query=topic,
            query_embeddings=topic_embedding,
        )
        existing_topics = [item.payload.get('data') for item in existing_topics if item.score >= 0.5]
        if not existing_topics:
            memory_id = str(uuid.uuid4())
            metadata = {'data': topic}
            self.vector_store.insert(
                docs=[topic],
                embeddings=[topic_embedding],
                ids=[memory_id],
                payloads=[metadata],
            )
            return topic
        prompt = self.topic_recommendation_prompt.format(
            TARGET_TOPIC_IS_HERE=topic,
            EXISTING_TOPICS_ARE_HERE=existing_topics,
        )
        response, _ = self.llm.generate_response([{'role': 'user', 'content': prompt}], )
        return response
