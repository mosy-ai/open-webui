from typing import Optional
import logging

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient as Qclient
from qdrant_client.http.models import PointStruct, ScoredPoint
from qdrant_client.models import models

from open_webui.retrieval.vector.main import VectorItem, SearchResult, GetResult
from open_webui.config import QDRANT_URI, QDRANT_API_KEY
from open_webui.env import SRC_LOG_LEVELS

NO_LIMIT = 999999999

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class QdrantClient:
    def __init__(self):
        self.QDRANT_URI = QDRANT_URI
        self.QDRANT_API_KEY = QDRANT_API_KEY
        self.client = (
            Qclient(url=self.QDRANT_URI, api_key=self.QDRANT_API_KEY)
            if self.QDRANT_URI
            else None
        )

        # TODO: Make this configurable
        # Note: The sparse text embedding in here only calculate the term frequency
        # The idf is calculated by qdrant engine when we define the modifier
        self.sparse_text_embedding = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        # Define threshold
        # TODO: Make this configurable through the config file and check for the best threshold 
        self.dense_search_threshold = 0.5
        self.sparse_search_threshold = 0.5
        self.fusion_threshold = 0.4

    def _result_to_get_result(self, points) -> GetResult:
        ids = []
        documents = []
        metadatas = []

        for point in points:
            payload = point.payload
            ids.append(point.id)
            documents.append(payload["text"])
            metadatas.append(payload["metadata"])

        return GetResult(
            **{
                "ids": [ids],
                "documents": [documents],
                "metadatas": [metadatas],
            }
        )

    def _create_collection(
        self, collection_name: str, dimension: int, enable_hybrid_search: bool = False
    ):
        if enable_hybrid_search:
            log.info(f"create collection {collection_name} with hybrid search")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense_embedding": models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    )
                },
                # Ref: https://qdrant.tech/documentation/concepts/indexing/#sparse-vector-index
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False),
                        # The modifier is the idf of the bm25
                        # It will automatically update when the collection is updated
                        # When a sparse vector is inserted, it will be modified by the idf -> idf * func(tf)
                        modifier=models.Modifier.IDF,
                    )
                },
            )
        else:
            log.info(f"create collection {collection_name} without hybrid search")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
            )

        log.info(f"collection {collection_name} successfully created!")

    def _create_collection_if_not_exists(
        self, collection_name, dimension, enable_hybrid_search: bool = False
    ):  
        is_collection_exists = self.has_collection(collection_name=collection_name)
        if not is_collection_exists:
            self._create_collection(
                collection_name=collection_name,
                dimension=dimension,
                enable_hybrid_search=enable_hybrid_search,
            )
        return is_collection_exists

    def _create_points(self, items: list[VectorItem]):
        points = []
        for item in items:
            if "sparse_vector" in item and item["sparse_vector"] is not None:
                points.append(
                    PointStruct(
                        id=item["id"],
                        vector={
                            "dense_embedding": item["vector"],
                            "bm25": item["sparse_vector"].as_object(),
                        },
                        payload={"text": item["text"], "metadata": item["metadata"]},
                    )
                )
            else:
                points.append(
                    PointStruct(
                        id=item["id"],
                        vector=item["vector"],
                        payload={"text": item["text"], "metadata": item["metadata"]},
                    )
                )
        return points

    def has_collection(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name)

    def delete_collection(self, collection_name: str):
        return self.client.delete_collection(collection_name=collection_name)

    def search(
        self,
        collection_name: str,
        vectors: list[list[float | int]],
        queries: list[str] = None,
        limit: int = 10,
        enable_hybrid_search: bool = False,
    ) -> Optional[SearchResult]:
        # Search for the nearest neighbor items based on the vectors and return 'limit' number of results.
        if limit is None:
            limit = NO_LIMIT  # otherwise qdrant would set limit to 10!

        if enable_hybrid_search:
            sparse_vector = next(self.sparse_text_embedding.query_embed(queries[0]))
            # Define the prefetch query for the sparse vector and the dense vector
            query_prefetch = [
                models.Prefetch(
                    query=models.SparseVector(**sparse_vector.as_object()),
                    using="bm25",
                    limit=limit,
                    # score_threshold=0.5,
                ),
                models.Prefetch(
                    query=vectors[0],
                    using="dense_embedding",
                    limit=limit,
                    # score_threshold=0.5,
                ),
            ]
            # Qdrant will prefetch the points with dense + sparse vector
            # and then apply the RRF fusion to the points
            query_response = self.client.query_points(
                collection_name=collection_name,
                prefetch=query_prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                # score_threshold=self.fusion_threshold,
            )
        else:
            query_response = self.client.query_points(
                collection_name=collection_name,
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

    def search_with_sparse_vector(
        self, collection_name: str, queries: list[str], limit: int = 10
    ):
        if limit is None:
            limit = NO_LIMIT  # otherwise qdrant would set limit to 10!

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
                collection_name=collection_name,
                query_filter=models.Filter(should=field_conditions),
                limit=limit,
            )
            return self._result_to_get_result(points.points)
        except Exception as e:
            log.exception(f"Error querying a collection '{collection_name}': {e}")
            return None

    def get(self, collection_name: str) -> Optional[GetResult]:
        # Get all the items in the collection.
        points = self.client.query_points(
            collection_name=collection_name,
            limit=NO_LIMIT,  # otherwise qdrant would set limit to 10!
        )
        return self._result_to_get_result(points.points)

    def insert(
        self,
        collection_name: str,
        items: list[VectorItem],
        enable_hybrid_search: bool = False,
        batch_size: int = 100,
    ):
        # Insert the items into the collection, if the collection does not exist, it will be created.
        self._create_collection_if_not_exists(
            collection_name, len(items[0]["vector"]), enable_hybrid_search
        )
        if enable_hybrid_search:
            for item in items:
                item["sparse_vector"] = next(
                    self.sparse_text_embedding.embed(item["text"])
                )
                
        # Disable the indexing when doing upload to avoid unnecessary indexing time
        # REF: https://qdrant.tech/documentation/database-tutorials/bulk-upload/
        self.client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )

        log.info(f"Inserting items: {len(items)}")
        for i in range(0, len(items), batch_size):
            points = self._create_points(items[i:i+batch_size])
            self.client.upsert(collection_name, points)
            
        # Re-enable the indexing after the upload for the collection to be searchable
        self.client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )

    def upsert(
        self,
        collection_name: str,
        items: list[VectorItem],
        enable_hybrid_search: bool = False,
    ):
        # Update the items in the collection, if the items are not present, insert them. If the collection does not exist, it will be created.
        self._create_collection_if_not_exists(
            collection_name, len(items[0]["vector"]), enable_hybrid_search
        )
        if enable_hybrid_search:
            for item in items:
                item["sparse_vector"] = next(
                    self.sparse_text_embedding.embed(item["text"])
                )

        points = self._create_points(items)
        return self.client.upsert(collection_name, points)

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
                (
                    field_conditions.append(
                        models.FieldCondition(
                            key="metadata.id",
                            match=models.MatchValue(value=id_value),
                        ),
                    ),
                )
        elif filter:
            for key, value in filter.items():
                (
                    field_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=value),
                        ),
                    ),
                )

        return self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=field_conditions)
            ),
        )

    def reset(self):
        # Resets the database. This will delete all collections and item entries.
        collection_names = self.client.get_collections().collections
        for collection_name in collection_names:
            self.client.delete_collection(collection_name=collection_name.name)

    def get_raw_data(self, collection_name: str):
        """This method is for getting the raw data from the collection
        In this case, we get the raw data from the file collection
        """
        # Get all the items in the collection.
        is_collection_exists = self.has_collection(collection_name=collection_name)
        log.info(f"collection {collection_name} exists: {is_collection_exists}")
        if is_collection_exists:
            # Get all the items in the collection.
            points = self.client.query_points(
                collection_name=collection_name,
                limit=NO_LIMIT,  # otherwise qdrant would set limit to 10!
                with_vectors=True,
                with_payload=True,
            )
            return points.points

        return None

    def insert_raw_data(
        self,
        collection_name: str,
        documents: list[ScoredPoint],
        enable_hybrid_search: bool = False,
        batch_size: int = 100,
    ):
        """This method is for migrating data from a collection to another collection
        In this case, we migrate from file collection to knowledge base collection
        """
        # Create points from the documents
        points = []
        dimension = None
        for item in documents:
            if enable_hybrid_search:
                point = PointStruct(
                    id=item.id,
                    vector={
                        "dense_embedding": item.vector["dense_embedding"],
                        "bm25": item.vector["bm25"],
                    },
                    payload=item.payload,
                )
                if dimension is None:
                    dimension = len(item.vector["dense_embedding"])
            else:
                point = PointStruct(
                    id=item.id,
                    vector=item.vector,
                    payload=item.payload,
                )
                if dimension is None:
                    dimension = len(item.vector)
                    
            points.append(point)

        log.info(f"Insert raw data: {len(points)}")
        if len(points) == 0:
            raise ValueError("No points to migrate from collection to file")
        
        # Create the collection if it doesn't exist
        is_collection_exists = self._create_collection_if_not_exists(
            collection_name=collection_name,
            dimension=dimension,
            enable_hybrid_search=enable_hybrid_search,
        )

        if not is_collection_exists:
            # Disable the indexing when doing upload to avoid unnecessary indexing time
            # We only do this for the first time migration
            # REF: https://qdrant.tech/documentation/database-tutorials/bulk-upload/
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
            )

        log.info(f"Migrating {len(points)} points to {collection_name} knowledge base collection")
        # upload the points to the collection
        for i in range(0, len(points), batch_size):
            log.info(f"Upserting points {i} to {i+batch_size} of {len(points)}")
            self.client.upsert(collection_name, points[i:i+batch_size])

        if not is_collection_exists:
            # Re-enable the indexing after the upload for the collection to be searchable
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
            )