import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

# Import your consumer function and related models/utilities.
# Adjust the module name 'my_module' as needed.
from data_processing.retrieval import (
    process_file_consumer,
    ProcessFileForm,
    Files,
    VECTOR_DB_CLIENT,
    StatusEnum,
)

def test_process_file_consumer_with_collection(monkeypatch):
    # Initialize Qdrant client for verification
    api_key = os.getenv("QDRANT_API_KEY")
    if not api_key:
        pytest.skip("QDRANT_API_KEY environment variable is not set")
    
    try:
        qdrant_client = QdrantClient(
            host="localhost",
            port=6333,
            api_key=api_key
        )
        # Test connection
        qdrant_client.get_collections()
    except Exception as e:
        pytest.skip(f"Could not connect to Qdrant: {str(e)}")

    # Create a fake file model.
    fake_file = MagicMock()
    fake_file.id = "123"
    fake_file.filename = "test.pdf"
    fake_file.user_id = "08a5ea73-ea56-4d1b-b5a1-c74533d9b952"
    fake_file.path = "08a5ea73-ea56-4d1b-b5a1-c74533d9b952_Q&A.pdf"
    fake_file.data = {"content": "initial content"}
    fake_file.meta = {"existing": "meta", "content_type": "application/pdf"}

    # Create a fake form with a collection name.
    fake_form = MagicMock(spec=ProcessFileForm)
    fake_form.file_id = "123"
    fake_form.collection_name = None

    # Patch Files.get_file_by_id to return the fake file.
    monkeypatch.setattr(Files, "get_file_by_id", lambda file_id: fake_file)

    # Create a fake vector DB query result (non‑empty).
    fake_query_result = MagicMock()
    fake_query_result.ids = [["1"]]
    fake_query_result.documents = [["vector document content"]]
    fake_query_result.metadatas = [[{"doc_meta": "value"}]]
    monkeypatch.setattr(VECTOR_DB_CLIENT, "query", lambda **kwargs: fake_query_result)

    # Patch the hash calculation to return a dummy hash.
    monkeypatch.setattr("data_processing.retrieval.calculate_sha256_string", lambda text: "dummyhash")

    # Create flags to track whether update functions are called.
    data_updated = False
    def fake_update_file_data_by_id(file_id, data):
        nonlocal data_updated
        data_updated = True
        assert data == {"content": "vector document content"}
    monkeypatch.setattr(Files, "update_file_data_by_id", fake_update_file_data_by_id)

    hash_updated = False
    def fake_update_file_hash_by_id(file_id, hash_val):
        nonlocal hash_updated
        hash_updated = True
        assert hash_val == "dummyhash"
    monkeypatch.setattr(Files, "update_file_hash_by_id", fake_update_file_hash_by_id)

    meta_updated = False
    def fake_update_file_metadata_by_id(file_id, meta):
        nonlocal meta_updated
        meta_updated = True
        assert meta.get("collection_name") == f"file-{fake_file.id}"
        assert meta.get("status") == StatusEnum.COMPLETED
    monkeypatch.setattr(Files, "update_file_metadata_by_id", fake_update_file_metadata_by_id)

    # Mock the Storage.get_file method to return a local file path
    def mock_get_file(file_path):
        return file_path
    monkeypatch.setattr("data_processing.retrieval.Storage.get_file", mock_get_file)

    # Mock the Loader to return a simple document
    def mock_loader_load(filename, content_type, file_path):
        return [MagicMock(page_content="vector document content", metadata={})]
    monkeypatch.setattr("data_processing.retrieval.Loader", lambda **kwargs: MagicMock(load=mock_loader_load))

    # Ensure that embedding is enabled.
    monkeypatch.setattr("data_processing.retrieval.BYPASS_EMBEDDING_AND_RETRIEVAL", False)

    # Invoke the function under test.
    process_file_consumer(fake_form)

    # Verify that the update functions were called.
    assert data_updated
    assert hash_updated
    assert meta_updated

    # Verify Qdrant insertion
    collection_name = f"file-{fake_file.id}"
    
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        assert collection_name in collection_names, f"Collection {collection_name} not found in Qdrant"

        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        assert collection_info.points_count > 0, f"Collection {collection_name} is empty"

        # Query the collection
        search_result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1
        )
        assert len(search_result[0]) > 0, "No documents found in collection"
        assert "text" in search_result[0][0].payload, "Document payload missing 'text' field"
        assert "vector" in search_result[0][0].payload, "Document payload missing 'vector' field"

    finally:
        # Clean up
        try:
            qdrant_client.delete_collection(collection_name=collection_name)
        except Exception as e:
            print(f"Warning: Failed to clean up collection {collection_name}: {str(e)}")