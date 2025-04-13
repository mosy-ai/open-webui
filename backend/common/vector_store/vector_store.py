# common/vector_store.py

from abc import ABC, abstractmethod
from typing import List, Any

# Example Point structure for vector data (you might adjust this as needed)
class Point:
    def __init__(self, id: str, vector: List[float], payload: dict):
        self.id = id
        self.vector = vector
        self.payload = payload

class AbstractVectorStore(ABC):
    """Abstract interface for vector storage providers."""

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine") -> None:
        pass

    @abstractmethod
    def upsert_points(self, collection_name: str, points: List[Point]) -> None:
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Any]:
        pass

# --- Concrete Implementation for Qdrant ---
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

class QdrantProvider(AbstractVectorStore):
    def __init__(self, host: str, port: int):
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine") -> None:
        try:
            self.client.get_collection(collection_name=collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": vector_size, "distance": distance}
            )

    def upsert_points(self, collection_name: str, points: List[Point]) -> None:
        # Convert our generic Point to Qdrant's PointStruct
        q_points = []
        for p in points:
            q_points.append(PointStruct(id=p.id, vector=p.vector, payload=p.payload))
        self.client.upsert(collection_name=collection_name, points=q_points)

    def search(self, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Any]:
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )

# --- Stub/Placeholder for other providers ---
class ElasticProvider(AbstractVectorStore):
    def __init__(self, host: str, port: int):
        # Initialize your Elasticsearch client here
        pass

    def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine") -> None:
        # Create an index or similar structure
        raise NotImplementedError

    def upsert_points(self, collection_name: str, points: List[Point]) -> None:
        raise NotImplementedError

    def search(self, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Any]:
        raise NotImplementedError

# You can add additional providers (e.g., PGVectorProvider, MilvusProvider) following this pattern.
