import importlib
from typing import Optional, Dict, Union

from himem.configs.embeddings.base import BaseEmbedderConfig
from himem.configs.llms.base import BaseLlmConfig
from himem.configs.rerankers.base import BaseRerankerConfig
from himem.configs.rerankers.huggingface import HuggingFaceRerankerConfig
from himem.configs.rerankers.llm import LLMRerankerConfig
from himem.configs.rerankers.sentence_transformer import SentenceTransformerRerankerConfig
from himem.configs.rerankers.zero_entropy import ZeroEntropyRerankerConfig


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    provider_to_class = {
        "ollama": "himem.llms.ollama.OllamaLLM",
        "openai": "himem.llms.openai.OpenAILLM",
        "qwen": "himem.llms.openai.OpenAILLM",
        "langchain": "himem.llms.langchain.LangchainLLM",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            llm_instance = load_class(class_type)
            base_config = BaseLlmConfig(**config)
            return llm_instance(base_config)
        else:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")


class EmbedderFactory:
    provider_to_class = {
        "openai": "himem.embeddings.openai.OpenAIEmbedding",
        "ollama": "himem.embeddings.ollama.OllamaEmbedding",
        "huggingface": "himem.embeddings.huggingface.HuggingFaceEmbedding",
        "langchain": "himem.embeddings.langchain.LangchainEmbedding",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "himem.vector_stores.qdrant.Qdrant",
        "chroma": "himem.vector_stores.chroma.ChromaDB",
        "pgvector": "himem.vector_stores.pgvector.PGVector",
        "milvus": "himem.vector_stores.milvus.MilvusDB",
        "pinecone": "himem.vector_stores.pinecone.PineconeDB",
        "elasticsearch": "himem.vector_stores.elasticsearch.ElasticsearchDB",
        "faiss": "himem.vector_stores.faiss.FAISS",
        "langchain": "himem.vector_stores.langchain.Langchain",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")

    @classmethod
    def reset(cls, instance):
        instance.reset()
        return instance


class GraphStoreFactory:
    """
    Factory for creating MemoryGraph instances for different graph store providers.
    Usage: GraphStoreFactory.create(provider_name, config)
    """

    provider_to_class = {
        "graphiti": "himem.graphs.graphiti.GraphitiMemory",
        "default": "himem.graphs.memory_graph.MemoryGraph",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name, cls.provider_to_class["default"])
        try:
            GraphClass = load_class(class_type)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import MemoryGraph for provider '{provider_name}': {e}")
        return GraphClass(config)


class RerankerFactory:
    """
    Factory for creating reranker instances with appropriate configurations.
    Supports provider-specific configs following the same pattern as other factories.
    """

    # Provider mappings with their config classes
    provider_to_class = {
        "sentence_transformer": (
            "himem.reranker.sentence_transformer_reranker.SentenceTransformerReranker",
            SentenceTransformerRerankerConfig),
        "zero_entropy": ("himem.reranker.zero_entropy_reranker.ZeroEntropyReranker", ZeroEntropyRerankerConfig),
        "llm_reranker": ("himem.reranker.llm_reranker.LLMReranker", LLMRerankerConfig),
        "huggingface": ("himem.reranker.huggingface_reranker.HuggingFaceReranker", HuggingFaceRerankerConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseRerankerConfig, Dict]] = None, **kwargs):
        """
        Create a reranker instance based on the provider and configuration.

        Args:
            provider_name: The reranker provider (e.g., 'cohere', 'sentence_transformer')
            config: Configuration object or dictionary
            **kwargs: Additional configuration parameters

        Returns:
            Reranker instance configured for the specified provider

        Raises:
            ImportError: If the provider class cannot be imported
            ValueError: If the provider is not supported
        """
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported reranker provider: {provider_name}")

        class_path, config_class = cls.provider_to_class[provider_name]

        # Handle configuration
        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config = config_class(**config, **kwargs)
        elif not isinstance(config, BaseRerankerConfig):
            raise ValueError(f"Config must be a {config_class.__name__} instance or dict")

        # Import and create the reranker class
        try:
            reranker_class = load_class(class_path)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import reranker for provider '{provider_name}': {e}")

        return reranker_class(config)
