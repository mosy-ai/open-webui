# common/vector_store/qdrant.py

import os
import logging
from typing import Optional, List, Union, Any
from qdrant_client import QdrantClient as Qclient
from qdrant_client.http.models import PointStruct
from qdrant_client.http import models
from fastembed import SparseTextEmbedding

# Import shared models and types
from common.models.vector import VectorItem, GetResult, SearchResult, ScoredPoint

log = logging.getLogger(__name__)

# Assume these configuration variables are set via environment or config file.
QDRANT_URI = "http://localhost:16333"
QDRANT_API_KEY = "oNViSToRURpA"

NO_LIMIT = 10000

class QdrantProvider:
    """
    Implements Qdrant-specific vector storage operations, including collection management, 
    upserting points with dense (and optionally sparse) embeddings, searching, and data migration.
    """
    def __init__(self):
        self.QDRANT_URI = QDRANT_URI
        self.QDRANT_API_KEY = QDRANT_API_KEY
        self.client = (
            Qclient(url=self.QDRANT_URI, api_key=self.QDRANT_API_KEY)
            if self.QDRANT_URI
            else None
        )

        # Initialize sparse text embedding (e.g., BM25)
        self.sparse_text_embedding = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )
        
        # Define thresholds (these could be made configurable)
        self.dense_search_threshold = 0.5
        self.sparse_search_threshold = 0.5
        self.fusion_threshold = 0.4

    def _result_to_get_result(self, points: List[Any]) -> GetResult:
        """
        Convert Qdrant result points to a GetResult dataclass.
        """
        ids = []
        documents = []
        metadatas = []
        for point in points:
            payload = point.payload
            ids.append(point.id)
            documents.append(payload.get("text"))
            metadatas.append(payload.get("metadata"))
        return GetResult(
            ids=[ids],
            documents=[documents],
            metadatas=[metadatas]
        )

    def _create_collection(self, collection_name: str, dimension: int):
        collection_name_with_prefix = f"{collection_name}"
        self.client.create_collection(
            collection_name=collection_name_with_prefix,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
        )

        log.info(f"collection {collection_name_with_prefix} successfully created!")

    def _create_collection_if_not_exists(self, collection_name, dimension):
        if not self.has_collection(collection_name=collection_name):
            self._create_collection(
                collection_name=collection_name, dimension=dimension
            )

    def _create_points(self, items: list[VectorItem]):
        return [
            PointStruct(
                id=item["id"],
                vector=item["vector"],
                payload={"text": item["text"], "metadata": item["metadata"]},
            )
            for item in items
        ]

    def has_collection(self, collection_name: str) -> bool:
        """
        Check if the given collection exists in Qdrant.
        """
        if not self.client:
            return False
        return self.client.collection_exists(collection_name)

    def delete_collection(self, collection_name: str) -> Any:
        """
        Delete an existing collection.
        """
        if not self.client:
            return None
        return self.client.delete_collection(collection_name=collection_name)

    def search(
        self, collection_name: str, vectors: list[list[float | int]], limit: int
    ) -> Optional[SearchResult]:
        # Search for the nearest neighbor items based on the vectors and return 'limit' number of results.
        if limit is None:
            limit = NO_LIMIT  # otherwise qdrant would set limit to 10!

        query_response = self.client.query_points(
            collection_name=f"{collection_name}",
            query=vectors[0],
            limit=limit,
        )
        get_result = self._result_to_get_result(query_response.points)
        return SearchResult(
            ids=get_result.ids,
            documents=get_result.documents,
            metadatas=get_result.metadatas,
            # qdrant distance is [-1, 1], normalize to [0, 1]
            distances=[[(point.score + 1.0) / 2.0 for point in query_response.points]],
        )

    def search_with_sparse_vector(self, collection_name: str, queries: List[str], limit: int = 10) -> GetResult:
        """
        Perform a sparse vector search using BM25.
        """
        if limit is None:
            limit = NO_LIMIT
        sparse_vector = next(self.sparse_text_embedding.query_embed(queries[0]))
        query_response = self.client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(**sparse_vector.as_object()),
            using="bm25",
            with_payload=True,
        )
        return self._result_to_get_result(query_response.points)

    def query(self, collection_name: str, filter: dict, limit: Optional[int] = None):
        # Construct the filter string for querying
        if not self.has_collection(collection_name):
            return None
        try:
            if limit is None:
                limit = NO_LIMIT  # otherwise qdrant would set limit to 10!

            field_conditions = []
            for key, value in filter.items():
                field_conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}", match=models.MatchValue(value=value)
                    )
                )

            points = self.client.query_points(
                collection_name=f"{collection_name}",
                query_filter=models.Filter(should=field_conditions),
                limit=limit,
            )
            return self._result_to_get_result(points.points)
        except Exception as e:
            log.exception(f"Error querying a collection '{collection_name}': {e}")
            return None
        
    def get_raw_data(self, collection_name: str) -> Optional[List[Any]]:
        """
        Retrieve the raw data from a collection.
        """
        is_exists = self.has_collection(collection_name)
        log.info(f"Collection {collection_name} exists: {is_exists}")
        if is_exists:
            points = self.client.query_points(
                collection_name=collection_name,
                limit=NO_LIMIT,
                with_vectors=True,
                with_payload=True,
            )
            return points.points
        return None

    def get(self, collection_name: str) -> Optional[GetResult]:
        # Get all the items in the collection.
        points = self.client.query_points(
            collection_name=f"{collection_name}",
            limit=NO_LIMIT,  # otherwise qdrant would set limit to 10!
        )
        return self._result_to_get_result(points.points)

    def insert(self, collection_name: str, items: list[VectorItem]):
        # Insert the items into the collection, if the collection does not exist, it will be created.
        self._create_collection_if_not_exists(collection_name, len(items[0]["vector"]))
        points = self._create_points(items)
        self.client.upload_points(f"{collection_name}", points)

    def upsert(self, collection_name: str, items: list[VectorItem]):
        # Update the items in the collection, if the items are not present, insert them. If the collection does not exist, it will be created.
        self._create_collection_if_not_exists(collection_name, len(items[0]["vector"]))
        points = self._create_points(items)
        return self.client.upsert(f"{collection_name}", points)

    def delete(
        self,
        collection_name: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ):
        # Delete the items from the collection based on the ids.
        field_conditions = []

        if ids:
            for id_value in ids:
                field_conditions.append(
                    models.FieldCondition(
                        key="metadata.id",
                        match=models.MatchValue(value=id_value),
                    ),
                ),
        elif filter:
            for key, value in filter.items():
                field_conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value),
                    ),
                ),

        return self.client.delete(
            collection_name=f"{collection_name}",
            points_selector=models.FilterSelector(
                filter=models.Filter(must=field_conditions)
            ),
        )

    def reset(self):
        # Resets the database. This will delete all collections and item entries.
        collection_names = self.client.get_collections().collections
        for collection_name in collection_names:
            self.client.delete_collection(collection_name=collection_name.name)