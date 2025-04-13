from common.vector_store.base import AbstractVectorStore
from common.vector_store.qdrant import QdrantProvider
# from common.vector_store.pgvector import PGVectorProvider
# from common.vector_store.milvus import MilvusProvider

# from common.config import VECTOR_DB
VECTOR_DB = "qdrant"

def get_vector_provider(provider_name: str, **kwargs) -> AbstractVectorStore:
    provider = provider_name.lower()
    if provider == "qdrant":
        return QdrantProvider()
    # elif provider == "elastic":
    #     return ElasticProvider()
    # elif provider == "pgvector":
    #     return PGVectorProvider(...)  # Initialize with proper kwargs
    # elif provider == "milvus":
    #     return MilvusProvider(...)
    else:
        raise RuntimeError(f"Unsupported vector provider: {provider}")
    
VECTOR_DB_CLIENT = get_vector_provider(VECTOR_DB)