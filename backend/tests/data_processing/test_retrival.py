import uuid
import json
import pytest

from common.models.files import Files, FileForm
from common.vector_store.vector_factory import VECTOR_DB_CLIENT
from data_processing.worker import process_file_consumer


# Use a dedicated collection name for this test.
TEST_COLLECTION = "integration_collection"

@pytest.fixture(scope="module")
def cleanup_test_file():
    """
    This fixture should clean up the test file record and the associated vector DB collection
    after tests are run.
    """
    # Choose a unique collection name for this integration test.
    yield TEST_COLLECTION
    # Cleanup: delete the file record and remove the vector DB collection.
    # (Assumes Files.delete_file and VECTOR_DB_CLIENT.delete_collection are available.)
    # Delete the file record if your Files module supports it.
    # Files.delete_file(test_file.id)
    if VECTOR_DB_CLIENT.has_collection(TEST_COLLECTION):
        VECTOR_DB_CLIENT.delete_collection(collection_name=TEST_COLLECTION)

def test_process_file_consumer_integration(cleanup_test_file):
    """
    Integration test of process_file_consumer using a real file record
    and a real Qdrant vector database.
    """
    collection_name = cleanup_test_file

    # 1. Create a real test file record in your database.
    test_file_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    filename = "integration_test.pdf"
    file_path = "08a5ea73-ea56-4d1b-b5a1-c74533d9b952_Q&A.pdf"  # must be a valid path or a value that your loader uses
    file_content = (
        "This is the real integration test content. It will be processed by the worker and "
        "inserted into the vector database after extraction and embedding."
    )
    file_meta = {
        "name": filename,
        "content_type": "text/plain",
        "size": len(file_content),
        "data": {"sample": "metadata"},
    }

    file_form_data = FileForm(
        id=test_file_id,
        filename=filename,
        path=file_path,
        meta=file_meta,
    )

    # Insert the file record. This should return a FileModel with valid attributes.
    file_item = Files.insert_new_file(user_id, file_form_data)
    print(f"File inserted: {file_item}")

    # For the purpose of this integration test, we simulate a file that already has its content
    # stored (for example, after initial upload) by providing file_content in the "data" field.
    file_item = json.dumps(file_form_data.model_dump())
    
    # 2. Run the worker process (process_file_consumer) on the test file.
    # This call should:
    #   - Retrieve the file record,
    #   - Process its content (either from the stored content or by using the storage/loader),
    #   - Update the file record with new content and its SHA256 hash,
    #   - And update the vector DB (via save_docs_to_vector_db) inserting the document's vector representations.
    process_file_consumer(file_item)
    
    # 5. Verify that the vector database now contains entries in the designated collection.
    vector_result = VECTOR_DB_CLIENT.get_raw_data(TEST_COLLECTION)
    assert vector_result is not None, "Vector DB query should return a result"
    # Check that at least one point was inserted.
    if hasattr(vector_result, 'points'):
        inserted_points = vector_result.points
    else:
        inserted_points = vector_result  # or however your raw_data is structured
    assert len(inserted_points) > 0, "There should be at least one point inserted into the collection"
    
    # Optionally, print out some details for visual verification.
    for point in inserted_points:
        print(f"Inserted point ID: {point.id}, text snippet: {point.payload.get('text')[:30]}")