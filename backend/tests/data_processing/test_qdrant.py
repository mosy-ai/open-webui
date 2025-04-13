import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import uuid
import os

def test_qdrant_insertion():
    # Get API key from environment variable
    api_key = os.getenv("QDRANT_API_KEY", "your-secret-key-here")
    
    # Initialize Qdrant client
    client = QdrantClient(
        host="localhost",
        port=6333,
        api_key=api_key
    )

    # Create a test collection
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,  # OpenAI embedding size
            distance=models.Distance.COSINE
        )
    )

    # Create test data
    test_vectors = np.random.rand(2, 1536).tolist()  # 2 vectors of size 1536
    test_payloads = [
        {"text": "Test document 1", "source": "test1.txt"},
        {"text": "Test document 2", "source": "test2.txt"}
    ]
    test_ids = [str(uuid.uuid4()) for _ in range(2)]

    # Insert vectors
    operation_info = client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=test_ids,
            vectors=test_vectors,
            payloads=test_payloads
        )
    )

    # Verify insertion
    assert operation_info.status == "completed"

    # Query the inserted vectors
    search_result = client.search(
        collection_name=collection_name,
        query_vector=test_vectors[0],
        limit=2
    )

    # Verify search results
    assert len(search_result) == 2
    assert search_result[0].id == test_ids[0]
    assert search_result[0].payload["text"] == "Test document 1"

    # Clean up
    client.delete_collection(collection_name=collection_name)

if __name__ == "__main__":
    pytest.main([__file__]) 