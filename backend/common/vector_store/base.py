# common/vector_store/base.py

from abc import ABC, abstractmethod
from typing import List, Any

class Point:
    def __init__(self, id: str, vector: List[float], payload: dict):
        self.id = id
        self.vector = vector
        self.payload = payload

class AbstractVectorStore(ABC):
    """Abstract interface for vector storage providers."""

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine") -> None:
        """Ensure the collection exists; create if necessary."""
        pass

    @abstractmethod
    def upsert_points(self, collection_name: str, points: List[Point]) -> None:
        """Upsert a list of points into the collection."""
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Any]:
        """Search the collection with a query vector and return results."""
        pass

