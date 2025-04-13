from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional, List, Any


class VectorItem(BaseModel):
    id: str
    text: str
    vector: List[float | int]
    sparse_vector: Optional[Any] = None
    metadata: Any


class GetResult(BaseModel):
    ids: Optional[List[List[str]]]
    documents: Optional[List[List[str]]]
    metadatas: Optional[List[List[Any]]]


class SearchResult(GetResult):
    distances: Optional[List[List[float | int]]]


@dataclass
class ScoredPoint:
    id: str
    vector: dict       # e.g., {"dense_embedding": [...], "bm25": {...}}
    payload: dict
    score: float
