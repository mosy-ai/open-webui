import os
import time
import uuid
from datetime import datetime
from typing import List, Optional

import pytest

# Import the real QdrantProvider and shared models
from common.vector_store.qdrant import QdrantProvider
from common.models.vector import VectorItem

# Import the refactored worker function for saving documents into the vector DB.
# Adjust the import path as necessary if your worker module is organized differently.
from data_processing.retrieval import save_docs_to_vector_db

# Import your text splitter implementation.
# Replace 'your_text_splitter_module' with your actual module name.
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

# Set up test environment variables
os.environ["QDRANT_URI"] = "http://localhost:6333"
os.environ["QDRANT_API_KEY"] = "oNViSToRURpA"

# A minimal Document class for the test (adjust as needed to match your actual Document).
class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

# A dummy embedding function that returns a predictable vector.
def dummy_embedding_function(texts: List[str], user: Optional[str] = None) -> List[List[float]]:
    # For each text, return a fixed dummy 3-dimensional vector.
    return [[0.1, 0.2, 0.3] for _ in texts]

# Global vector database settings expected by save_docs_to_vector_db.
# Ensure that these are set before the function is invoked.
VECTOR_DB = "qdrant"
VECTOR_DB_CLIENT = QdrantProvider()

# Base configuration dictionary for processing.
config = {
    "ENABLE_RAG_HYBRID_SEARCH": False,
    "ENABLE_RAG_PARENT_RETRIEVER": False,
    # Use a character-based splitter for this test.
    "TEXT_SPLITTER": "character",
    # Set a small chunk size to force splitting.
    "CHUNK_SIZE": 20,
    "CHUNK_OVERLAP": 5,
    "RAG_EMBEDDING_ENGINE": "dummy_engine",
    "RAG_EMBEDDING_MODEL": "dummy_model",
    # Provide the dummy embedding function.
    "embedding_function": dummy_embedding_function,
}

# Name of the test collection for integration tests.
TEST_COLLECTION = "test_collection_save_docs"

# Fixture to clean up the test collection before and after each test.
@pytest.fixture(autouse=True)
def cleanup_collection():
    if VECTOR_DB_CLIENT.has_collection(TEST_COLLECTION):
        VECTOR_DB_CLIENT.delete_collection(TEST_COLLECTION)
    yield
    if VECTOR_DB_CLIENT.has_collection(TEST_COLLECTION):
        VECTOR_DB_CLIENT.delete_collection(TEST_COLLECTION)

def test_save_docs_to_vector_db_with_splitting():
    """
    Integration test to verify that saving a document with splitting enabled
    inserts multiple items into Qdrant.
    """
    # Create one Document whose content contains several paragraphs.
    content = (
        "This is the first paragraph. It is long enough to require splitting. \n\n"
        "Here is the second paragraph. It also needs to be split into multiple chunks. \n\n"
        "Finally, this is the third paragraph to test the splitter."
    )
    doc = Document(page_content=content, metadata={"name": "SplitDoc", "file_id": "file456"})

    # Update configuration to force splitting (using the character splitter).
    split_config = config.copy()
    # Ensure CHUNK_SIZE is small so the text is split into multiple chunks.
    split_config["CHUNK_SIZE"] = 20
    split_config["CHUNK_OVERLAP"] = 5

    # Call save_docs_to_vector_db with split=True.
    result = save_docs_to_vector_db(
        docs=[doc],
        collection_name=TEST_COLLECTION,
        config=split_config,
        metadata={"hash": "unique-hash-789", "file_id": "file456"},
        overwrite=True,
        split=True,
        add=False,
        user="test_user",
    )
    
    # Verify that the operation returns True.
    assert result is True, "save_docs_to_vector_db should return True on success"

    raw_data = VECTOR_DB_CLIENT.get_raw_data(TEST_COLLECTION)
    
    assert raw_data is not None, "Raw data should not be None after insertion"

    assert len(raw_data) > 1, "There should be multiple points inserted into the collection due to splitting"

    # Optional: Print out IDs and texts for visual verification.
    for point in raw_data:
        print(f"Inserted point ID: {point.id}, Text snippet: {point.payload.get('text')[:30]}")

