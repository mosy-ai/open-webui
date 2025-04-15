import json
import traceback
import logging

from open_webui.utils.misc import (
    calculate_sha256_string, measure_time
)
from common.models.files import Files, ProcessFileForm

from data_processing.retrieval import process_existing_collection, process_file_from_storage
from data_processing.retrieval import save_docs_to_vector_db


log = logging.getLogger(__name__)

@measure_time
def process_file_consumer(form_data) -> None:
    """
    Processes a file based on the provided form data.
    Intended to be invoked by a consumer that pulls messages from the queue.
    """
    try:
        # Retrieve file record from the database.
        form_data_dict = json.loads(form_data)
        # Use the value of "file_id" if present; otherwise, use the "id" key.
        file_id = form_data_dict.get("file_id") or form_data_dict.get("id")
        # Also, try to extract collection_name if present.
        collection_name = form_data_dict.get("collection_name")
        # Now create an instance of ProcessFileForm.
        form_data = ProcessFileForm(
            file_id=file_id,
            collection_name=collection_name
        )
        print(f"Form data: file_id={form_data.file_id}, collection_name={form_data.collection_name}")
        
        # Retrieve the file record.
        file = Files.get_file_by_id(form_data.file_id)
        if file is None:
            # Log an error and exit the processing if no record is found.
            log.error(f"File with id {form_data.file_id} not found in the database")
            return

        # Determine collection name: use provided one or fallback to default.
        collection_name = form_data.collection_name or f"file-{file.id}"

        # Prepare docs and text_content based on whether a collection name was provided.
        if form_data.collection_name:
            docs, text_content = process_existing_collection(file)
        else:
            docs, text_content = process_file_from_storage(file)
        
        print(f"Processed text_content: {text_content}")

        # Update file record with processed text and calculate hash.
        Files.update_file_data_by_id(file.id, {"content": text_content})
        hash_val = calculate_sha256_string(text_content)
        Files.update_file_hash_by_id(file.id, hash_val)

        # If embedding/retrieval is enabled, insert documents into the vector DB.
        try:
            result = save_docs_to_vector_db(
                docs=docs,
                collection_name=collection_name,
                metadata={
                    "file_id": file.id,
                    "name": file.filename,
                    "hash": hash_val,
                },
                add=True if form_data.collection_name else False,
                user=file.user_id,  # Use file's owner as the user
            )
            if result:
                Files.update_file_metadata_by_id(
                    file.id,
                    {
                        "collection_name": collection_name,
                    },
                )
            else:
                Files.update_file_metadata_by_id(
                    file.id,
                    {
                        "error_message": "Document insertion returned false.",
                    },
                )
        except Exception as e:
            log.exception(e)
            raise e

    except Exception as e:
        log.exception(traceback.format_exc())