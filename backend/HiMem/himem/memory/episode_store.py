import hashlib
import json
import os
from copy import deepcopy

from opensearchpy import OpenSearch, exceptions
from loguru import logger

try:
    from urllib3.exceptions import MaxRetryError, NewConnectionError

    _URLLIB3_CONN_ERRORS = (MaxRetryError, NewConnectionError)
except ImportError:  # pragma: no cover
    _URLLIB3_CONN_ERRORS = ()


def _opensearch_connection_failed(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    if _URLLIB3_CONN_ERRORS and isinstance(exc, _URLLIB3_CONN_ERRORS):
        return True
    msg = str(exc).lower()
    return "connection refused" in msg or "failed to establish" in msg or "newconnectionerror" in msg

from himem.dataset.model import Session
from himem.memory.utils import parse_messages
from himem.utils.factory import LlmFactory, EmbedderFactory

MAPPING_DEFINITION = {
    "settings": {
        "index.knn": True  # enable vector index
    },
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "timestamp": {"type": "text"},
            "topic": {"type": "keyword"},
            "topic_summary": {"type": "text"},
            "content": {"type": "text"},
            "content_embedding": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "engine": "lucene",
                    "space_type": "l2",
                    "name": "hnsw",
                    "parameters": {}
                }
            }
        }
    }
}

PIPELINE_NAME = 'hi-rrf-pipeline'
PIPELINE_BODY = {
    "description": "RRF processor for hybrid search score fusion and deduplication.",
    "phase_results_processors": [
        {
            "score-ranker-processor": {
                "combination": {
                    "technique": "rrf"
                }
            }
        }
    ]
}


class EpisodeStore:
    def __init__(self, config):
        self.config = config
        current_config = self.config.components['episode_memory'].config
        _host = os.environ.get("OPENSEARCH_HOST", "localhost")
        _port = int(os.environ.get("OPENSEARCH_PORT", "9200"))
        self.event_store_client = OpenSearch(
            hosts=[{"host": _host, "port": _port}],
            use_ssl=False,
            verify_certs=False,
        )

        self.enabled_llm_provider = current_config["llm_provider"]
        self.llm_config = self.config.llm_providers[self.enabled_llm_provider].config
        self.llm = LlmFactory.create(self.enabled_llm_provider, self.llm_config)

        self.index_prefix = current_config.get("index_prefix", "default")

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
        )

        self.ensure_pipeline_registered_once(PIPELINE_NAME, PIPELINE_BODY)

    def add(self, segment, processed_metadata, session: Session = None, enable_knowledge_alignment=False):
        if not segment:
            raise Exception('No segment')

        metadata = deepcopy(processed_metadata)
        user_id = metadata.get("user_id")
        timestamp = metadata['timestamp']

        exchanges = segment['exchanges']
        content = ""
        for turn in exchanges:
            content += f"{turn.role}:{turn.content}\n"
        content = content.strip()

        if enable_knowledge_alignment and session:
            logger.warning(f"Enabling knowledge alignment in episode memory, but it should not take effect according to the experiment.")
            # content = self.coreference_resolution(content, session)

        # Save them.
        embedding = self.embedding_model.encode(content)
        metadata['segment_id'] = segment['id']

        doc = {'timestamp': timestamp, 'content': content,
               'content_embedding': embedding,
               'topic': segment['topic'], 'topic_summary': segment['topic_summary'],
               'metadata': metadata}

        index_name = f"{self.index_prefix}_{user_id}_corpus".lower()
        try:
            if not self.event_store_client.indices.exists(index=index_name):
                mapping = deepcopy(MAPPING_DEFINITION)
                if hasattr(self.embedding_model, "config") and hasattr(self.embedding_model.config, "embedding_dims"):
                    mapping["mappings"]["properties"]["content_embedding"]["dimension"] = self.embedding_model.config.embedding_dims
                self.event_store_client.indices.create(index=index_name, body=mapping)

            data_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
            _id = hashlib.blake2b(data_str.encode('utf-8'), digest_size=16).hexdigest()

            self.event_store_client.index(index=index_name, id=_id, body=doc, refresh=True)
            return _id
        except Exception as e:
            if _opensearch_connection_failed(e):
                logger.warning("OpenSearch unreachable; skipping episode persist: {}", e)
                return None
            raise

    def search(self, query, filters, limit=5):
        query_embedding = self.embedding_model.encode(query)
        query_embedding = [float(x) for x in query_embedding]

        body = {
            "size": limit,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "knn": {
                                "content_embedding": {
                                    "vector": query_embedding,
                                    "k": 50
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "content",
                                    "topic_summary",
                                ],
                                "type": "best_fields",
                                "operator": "or",
                                "fuzziness": "AUTO"
                            }
                        }
                    ]
                }
            }
        }
        user_id = filters["user_id"]
        index_name = f"{self.index_prefix}_{user_id}_corpus".lower()

        results = []
        try:
            temp = self.event_store_client.search(index=index_name, body=body, params={
                "search_pipeline": PIPELINE_NAME
            })
        except exceptions.NotFoundError:
            return results
        except Exception as e:
            if _opensearch_connection_failed(e):
                logger.warning("OpenSearch unreachable; episode search returns no hits: {}", e)
                return results
            raise
        hits = temp.get("hits", {}).get("hits", [])
        for hit in hits:
            score = hit["_score"]
            source = hit["_source"]
            results.append(
                {'id': hit['_id'], 'timestamp': source['timestamp'], 'metadata': source['metadata'], 'score': score,
                 'content': source["content"], 'topic': source["topic"], 'topic_summary': source["topic_summary"]})
        return results

    def ensure_pipeline_registered_once(self, pipeline_name, pipeline_body):
        """Checks if a pipeline exists and registers it only if it does not."""
        path = f"/_search/pipeline/{pipeline_name}"

        # 1. Check if the pipeline exists using a GET request
        try:
            # GET /_search/pipeline/{pipeline-name}
            self.event_store_client.transport.perform_request(
                method="GET",  # HEAD is more efficient than GET for checking existence
                url=path
            )
            print(f"✅ Search Pipeline **{pipeline_name}** already exists. Skipping creation.")
            return

        except exceptions.NotFoundError:
            # Proceed to creation if the pipeline is not found (404)
            print(f"Pipeline **{pipeline_name}** not found. Proceeding to create...")
        except Exception as e:
            # Handle other possible connection/auth errors
            print(f"Error during existence check for {pipeline_name}: {e}")
            return

        # 2. Register the pipeline if it doesn't exist
        try:
            response = self.event_store_client.transport.perform_request(
                method="PUT",
                url=path,
                body=pipeline_body
            )

            if response and response.get('acknowledged', False):
                print(f"Successfully registered Search Pipeline: **{pipeline_name}**.")
            else:
                print(f"Failed to register pipeline. Response: {response}")

        except Exception as e:
            print(f"An error occurred during pipeline creation: {e}")

    def coreference_resolution(self, content: str, session: Session):
        prompt_path = "config/instructions/coreference_resolution.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        messages = []
        for turn in session.turns:
            if turn.role == "user":
                messages.append({'role': turn.role, 'content': turn.content})
        parsed_messages = parse_messages(messages)

        user_prompt = system_prompt.format(
            text_to_be_aligned=content,
            conversation_is_here=parsed_messages,
        )
        response, _ = self.llm.generate_response(
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return response
