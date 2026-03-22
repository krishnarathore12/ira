import collections
import os
import shutil
import uuid

from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import MatchText
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
)

from himem.vector_stores.base import VectorStoreBase


class Qdrant(VectorStoreBase):
    def __init__(
            self,
            collection_name: str,
            embedding_model_dims: int,
            client: QdrantClient = None,
            host: str = None,
            port: int = None,
            path: str = None,
            url: str = None,
            api_key: str = None,
            on_disk: bool = False,
    ):
        """
        Initialize the Qdrant vector store.

        Args:
            collection_name (str): Name of the collection.
            embedding_model_dims (int): Dimensions of the embedding model.
            client (QdrantClient, optional): Existing Qdrant client instance. Defaults to None.
            host (str, optional): Host address for Qdrant server. Defaults to None.
            port (int, optional): Port for Qdrant server. Defaults to None.
            path (str, optional): Path for local Qdrant database. Defaults to None.
            url (str, optional): Full URL for Qdrant server. Defaults to None.
            api_key (str, optional): API key for Qdrant server. Defaults to None.
            on_disk (bool, optional): Enables persistent storage. Defaults to False.
        """
        if client:
            self.client = client
        else:
            params = {}
            if api_key:
                params["api_key"] = api_key
            if url:
                params["url"] = url
            if host and port:
                params["host"] = host
                params["port"] = port
            if not params:
                params["path"] = path
                if not on_disk:
                    if os.path.exists(path) and os.path.isdir(path):
                        shutil.rmtree(path)

            self.client = QdrantClient(**params)

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.on_disk = on_disk
        self.create_col(embedding_model_dims, on_disk)
        self.dense_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    def create_col(self, vector_size: int, on_disk: bool, distance: Distance = Distance.COSINE):
        """
        Create a new collection.

        Args:
            vector_size (int): Size of the vectors to be stored.
            on_disk (bool): Enables persistent storage.
            distance (Distance, optional): Distance metric for vector similarity. Defaults to Distance.COSINE.
        """
        # Skip creating collection if already exists
        response = self.list_cols()
        for collection in response.collections:
            if collection.name == self.collection_name:
                logger.debug(f"Collection {self.collection_name} already exists. Skipping creation.")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.embedding_model_dims,
                    distance=models.Distance.COSINE
                )
            },

            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                ),
            }
        )

    def get_sparse_vector(self, text: str):
        """Generates a simple sparse vector (term frequency mapping)."""
        tokens = text.lower().split()
        # Create a frequency map {token_index: count}
        sparse_values = collections.Counter(tokens)

        # In a real system, you would map these tokens to a global vocabulary index.
        # For a Qdrant payload/sparse format, you need indices and values.
        # We will use simple integer hashes as placeholder indices for illustration.
        indices = [hash(token) % 100000 for token in sparse_values.keys()]  # Using hash as index placeholder
        values = list(sparse_values.values())

        # Qdrant expects a dict with 'indices' and 'values' for sparse vectors
        return {"indices": indices, "values": values}

    def insert(self, docs: list, embeddings: list, payloads: list = None, ids: list = None):
        """
        Insert vectors into a collection.

        Args:
            docs (list): List of documents to insert.
            embeddings (list): List of documents to insert.
            payloads (list, optional): List of payloads corresponding to vectors. Defaults to None.
            ids (list, optional): List of IDs corresponding to vectors. Defaults to None.
        """
        # logger.info(f"Inserting {len(embeddings)} vectors into collection {self.collection_name}")
        sparse_vecs = [self.get_sparse_vector(doc) for doc in docs]
        points = [
            PointStruct(
                id=uuid.uuid4().hex if ids is None else ids[idx],
                vector={
                    "dense": embeddings[idx],
                    "sparse": models.SparseVector(
                        indices=sparse_vecs[idx]["indices"],
                        values=sparse_vecs[idx]["values"]
                    )
                },
                payload=payloads[idx] if payloads else {},
            )
            for idx, doc in enumerate(docs)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def _create_filter(self, filters: dict) -> Filter:
        """
        Create a Filter object from the provided filters.

        Args:
            filters (dict): Filters to apply.

        Returns:
            Filter: The created Filter object.
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict) and "gte" in value and "lte" in value:
                conditions.append(FieldCondition(key=key, range=Range(gte=value["gte"], lte=value["lte"])))
            elif isinstance(value, dict) and "in" in value:
                search_terms = value["in"]
                for term in search_terms:
                    condition = FieldCondition(
                        key=key,
                        match=MatchText(text=term)  # "like" match for that single term
                    )
                    conditions.append(condition)
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    def search(self, query: str, query_embeddings: list, limit: int = 5, filters: dict = None) -> list:
        """
        Search for similar vectors.

        Args:
            query (str): Query.
            query_embeddings (list): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """
        query_filter = self._create_filter(filters) if filters else None
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embeddings,
            query_filter=query_filter,
            limit=limit,
            using="dense",
        )
        return hits.points

    def hybrid_search(self, query: str, query_embeddings: list, limit: int = 5, filters: dict = None) -> list:
        """
        Search for similar vectors.

        Args:
            query (str): Query.
            query_embeddings (list): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """
        query_filter = self._create_filter(filters) if filters else None
        hits = self.client.query_points(
            collection_name=self.collection_name,
            # 1. Prefetch the results from the SPARSE (lexical) vector index
            prefetch=[
                # 2. Prefetch the results from the DENSE (semantic) vector index
                models.Prefetch(
                    # Pass the dense query vector (all-mpnet-base-v2)
                    query=query_embeddings,
                    # Specify the named vector field to use
                    using="dense",
                    limit=limit,  # Number of candidates to fetch from the dense index
                ),
            ],

            # 3. Specify the final fusion method on the combined results
            query=models.FusionQuery(
                fusion=models.Fusion.RRF  # Reciprocal Rank Fusion
            ),
            query_filter=query_filter,
            limit=limit,  # Final number of results to return after fusion
        )
        return hits.points

    def delete(self, vector_id: int):
        """
        Delete a vector by ID.

        Args:
            vector_id (int): ID of the vector to delete.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=[vector_id],
            ),
        )

    def update(self, vector_id: int, doc: str, embeddings, payload: dict = None):
        """
        Update a vector and its payload.

        Args:
            vector_id (int): ID of the vector to update.
            doc (str): Updated document.
            embeddings (list): Updated document.
            payload (dict, optional): Updated payload. Defaults to None.
        """
        sparse_vec = self.get_sparse_vector(doc)
        point = PointStruct(
            id=vector_id,
            vector={
                "dense": embeddings,
                "sparse": models.SparseVector(
                    indices=sparse_vec["indices"],
                    values=sparse_vec["values"]
                )
            },
            payload=payload if payload else {},
        )

        self.client.upsert(collection_name=self.collection_name, points=[point])

    def get(self, vector_id: int) -> dict:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (int): ID of the vector to retrieve.

        Returns:
            dict: Retrieved vector.
        """
        result = self.client.retrieve(collection_name=self.collection_name, ids=[vector_id], with_payload=True)
        return result[0] if result else None

    def list_cols(self) -> list:
        """
        List all collections.

        Returns:
            list: List of collection names.
        """
        return self.client.get_collections()

    def delete_col(self):
        """Delete a collection."""
        self.client.delete_collection(collection_name=self.collection_name)

    def col_info(self) -> dict:
        """
        Get information about a collection.

        Returns:
            dict: Collection information.
        """
        return self.client.get_collection(collection_name=self.collection_name)

    def list(self, filters: dict = None, limit: int = 100) -> list:
        """
        List all vectors in a collection.

        Args:
            filters (dict, optional): Filters to apply to the list. Defaults to None.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            list: List of vectors.
        """
        query_filter = self._create_filter(filters) if filters else None
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return result

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_col()
        self.create_col(self.embedding_model_dims, self.on_disk)
