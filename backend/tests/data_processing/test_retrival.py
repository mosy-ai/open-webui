import json

import pytest
from unittest.mock import patch, MagicMock, mock_open

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

    # Patch save_docs_to_vector_db to simulate a successful operation.
    monkeypatch.setattr("data_processing.retrieval.save_docs_to_vector_db", lambda **kwargs: True)

    # Create flags to track whether update functions are called.
    data_updated = False
    def fake_update_file_data_by_id(file_id, data):
        nonlocal data_updated
        data_updated = True
        # In this branch, text content comes from file.data["content"].
        assert data == {"content": "initial content"}
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
        # Expecting the provided collection name and a COMPLETED status.
        assert meta.get("collection_name") == "custom_collection"
        assert meta.get("status") == StatusEnum.COMPLETED
    monkeypatch.setattr(Files, "update_file_metadata_by_id", fake_update_file_metadata_by_id)

    # Ensure that embedding is enabled.
    monkeypatch.setattr("data_processing.retrieval.BYPASS_EMBEDDING_AND_RETRIEVAL", False)

    # Invoke the function under test.
    process_file_consumer(fake_form)

    # Verify that the update functions were called.
    assert data_updated
    assert hash_updated
    assert meta_updated