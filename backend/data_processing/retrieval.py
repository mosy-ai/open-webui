import logging
import time
import json
import uuid
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from langchain_core.documents import Document
from datetime import datetime

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

from common.vector_store.vector_factory import VECTOR_DB_CLIENT
from common.models.files import Files, StatusEnum

# from open_webui.models.files import Files, StatusEnum
# from open_webui.utils.misc import (
#     calculate_sha256_string,
# )
# from open_webui.storage.provider import Storage
# from open_webui.retrieval.vector.connector import VECTOR_DB_CLIENT
# from open_webui.retrieval.loaders.main import Loader
# from open_webui.config import (
#     BYPASS_EMBEDDING_AND_RETRIEVAL,
#     CONTENT_EXTRACTION_ENGINE,
#     TIKA_SERVER_URL,
#     PDF_EXTRACT_IMAGES,
#     DOCUMENT_INTELLIGENCE_ENDPOINT,
#     DOCUMENT_INTELLIGENCE_KEY
# )

from data_processing.utils import get_embedding_function
from data_processing.errors import ERROR_MESSAGES

from open_webui.utils.misc import (
    calculate_sha256_string, measure_time
)
import traceback

log = logging.getLogger(__name__)

class ProcessFileForm(BaseModel):
    file_id: str
    content: Optional[str] = None
    collection_name: Optional[str] = None

@measure_time
def save_docs_to_vector_db(
    docs: List[Document],
    collection_name: str,
    config: dict,
    metadata: Optional[dict] = None,
    overwrite: bool = False,
    split: bool = True,
    add: bool = False,
    user=None
) -> bool:
    """
    Process a list of Document objects and insert them into the vector DB.
    
    Parameters:
      - docs: A list of Document instances that hold page_content and metadata.
      - collection_name: The target collection name in the vector DB.
      - config: A dict of configuration parameters. Expected keys include:
          * ENABLE_RAG_HYBRID_SEARCH (bool)
          * ENABLE_RAG_PARENT_RETRIEVER (bool)
          * TEXT_SPLITTER (str, e.g., "character" or "token")
          * CHUNK_SIZE, CHUNK_OVERLAP (integers)
          * (If token splitter is used) TIKTOKEN_ENCODING_NAME (str)
          * (Optionally) PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP (integers)
          * RAG_EMBEDDING_ENGINE, RAG_EMBEDDING_MODEL (str)
          * embedding_function: a callable that accepts a list of texts and an optional user
      - metadata: Optional additional metadata (e.g., including a content hash for duplicate checking).
      - overwrite: If True and the collection exists, delete it before inserting.
      - add: If True, migrate vectors from an existing file-based collection to the target collection.
      - user: Optional user context for the embedding function.
      
    Returns:
      - True on success.
      
    Raises:
      - ValueError in case of duplicate content (based on a provided hash) or configuration issues.
    """

    def _get_docs_info(docs: List[Document]) -> str:
        info = set()
        for doc in docs:
            mdata = getattr(doc, "metadata", {})
            name = mdata.get("name") or mdata.get("title") or mdata.get("source", "")
            if name:
                info.add(name)
        return ", ".join(info)

    log.info(f"Processing docs: {_get_docs_info(docs)} in collection {collection_name}")

    # Check for duplicate using a metadata hash if provided.
    if metadata and "hash" in metadata:
        result = VECTOR_DB_CLIENT.query(
            collection_name=collection_name,
            filter={"hash": metadata["hash"]}
        )
        if result is not None and result.ids[0]:
            log.info(f"Document with hash {metadata['hash']} already exists")
            raise ValueError("Duplicate content detected")

    # Migrate documents if add flag is set.
    if add:
        if VECTOR_DB_CLIENT in ["chroma", "qdrant"]:
            file_collection_name = f"file-{metadata['file_id']}"
        else:
            raise ValueError("Vector database not supported for file migration")
        log.info(f"Migrating {len(docs)} documents from {file_collection_name} to {collection_name}")
        all_documents = VECTOR_DB_CLIENT.get_raw_data(collection_name=file_collection_name)
        log.info(f"Inserting raw data from {file_collection_name} to {collection_name}")
        VECTOR_DB_CLIENT.insert_raw_data(
            collection_name=collection_name,
            documents=all_documents,
        )
        log.info(f"Migration from {file_collection_name} to {collection_name} successful")
        return True

    # Split documents if required.
    if split:
        splitter_type = config.get("TEXT_SPLITTER", "character")
        enable_parent = config.get("ENABLE_RAG_PARENT_RETRIEVER", False)
        if splitter_type in ["", "character"]:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.get("CHUNK_SIZE", 1000),
                chunk_overlap=config.get("CHUNK_OVERLAP", 100),
                add_start_index=True,
            )
            if enable_parent:
                parent_text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.get("PARENT_CHUNK_SIZE", 2000),
                    chunk_overlap=config.get("PARENT_CHUNK_OVERLAP", 200),
                    add_start_index=True,
                )
        elif splitter_type == "token":
            encoding_name = config.get("TIKTOKEN_ENCODING_NAME", "default")
            text_splitter = TokenTextSplitter(
                encoding_name=encoding_name,
                chunk_size=config.get("CHUNK_SIZE", 1000),
                chunk_overlap=config.get("CHUNK_OVERLAP", 100),
                add_start_index=True,
            )
            if enable_parent:
                parent_text_splitter = TokenTextSplitter(
                    encoding_name=encoding_name,
                    chunk_size=config.get("PARENT_CHUNK_SIZE", 2000),
                    chunk_overlap=config.get("PARENT_CHUNK_OVERLAP", 200),
                    add_start_index=True,
                )
        else:
            raise ValueError("Invalid text splitter configuration")

        if enable_parent:
            # First, create parent-level chunks.
            parent_docs = parent_text_splitter.split_documents(docs)
            parent_ids = [str(uuid.uuid4()) for _ in parent_docs]
            child_docs = []
            # (Optional) Prepare parent docs for saving in a separate DB if needed.
            for i, doc in enumerate(parent_docs):
                sub_docs = text_splitter.split_documents([doc])
                parent_id = parent_ids[i]
                for sub_doc in sub_docs:
                    sub_doc.metadata["parent_id"] = parent_id
                child_docs.extend(sub_docs)
            docs = child_docs
        else:
            docs = text_splitter.split_documents(docs)

    if len(docs) == 0:
        raise ValueError("No content available after splitting")

    # Prepare texts and metadata for embedding.
    texts = [doc.page_content for doc in docs]
    final_metadata = []
    context_contents = []
    for doc in docs:
        if "context_content" in doc.metadata:
            context_contents.append(doc.metadata["context_content"])
            del doc.metadata["context_content"]
        # Merge document metadata with additional metadata.
        merged = {**doc.metadata, **(metadata or {})}
        merged["embedding_config"] = json.dumps({
            "engine": config.get("RAG_EMBEDDING_ENGINE"),
            "model": config.get("RAG_EMBEDDING_MODEL")
        })
        final_metadata.append(merged)

    # Normalize metadata values (convert lists/dicts/datetime to string).
    for meta in final_metadata:
        for key, value in meta.items():
            if isinstance(value, (list, dict, datetime)):
                meta[key] = str(value)

    # If the collection exists, handle overwrite logic.
    if VECTOR_DB_CLIENT.has_collection(collection_name=collection_name):
        log.info(f"Collection {collection_name} exists")
        if overwrite:
            VECTOR_DB_CLIENT.delete_collection(collection_name=collection_name)
            log.info(f"Deleted existing collection {collection_name}")
        elif not add:
            log.info("Collection exists and overwrite is False; nothing to do")
            return True

    log.info(f"Inserting documents into collection {collection_name}")

    # Get the embedding function from the configuration.
    embedding_function = config.get("embedding_function")
    if not embedding_function:
        raise ValueError("Missing embedding function in configuration")

    start_time = time.time()
    # Calculate embeddings for all texts.
    embeddings = embedding_function([text.replace("\n", " ") for text in texts], user=user)
    end_time = time.time()
    log.info(f"Embedding completed in {end_time - start_time:.2f} seconds")

    if context_contents:
        if len(context_contents) != len(texts):
            raise AssertionError("Mismatch between context_contents and texts")
        texts = context_contents

    # Prepare items for upsert in the vector DB.
    items = []
    for idx, text in enumerate(texts):
        items.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "vector": embeddings[idx],
            "metadata": final_metadata[idx],
        })

    # Insert the items into the collection.
    VECTOR_DB_CLIENT.insert(
        collection_name=collection_name,
        items=items,
    )

    return True

def process_file_consumer(form_data: ProcessFileForm):
    """
    Processes a file based on the provided form data.
    This function is intended to be invoked by a consumer that pulls messages from the queue.
    """
    try:
        # Retrieve the file record from the database
        file = Files.get_file_by_id(form_data.file_id)

        # Determine the collection name: use the provided one or fallback to a default
        collection_name = form_data.collection_name or f"file-{file.id}"

        if form_data.collection_name:
            # If a collection name is provided without new content, query the vector DB first
            result = VECTOR_DB_CLIENT.query(
                collection_name=f"file-{file.id}", filter={"file_id": file.id}
            )
            if result is not None and len(result.ids[0]) > 0:
                docs = [
                    Document(
                        page_content=result.documents[0][idx],
                        metadata=result.metadatas[0][idx],
                    )
                    for idx, _ in enumerate(result.ids[0])
                ]
            else:
                docs = [
                    Document(
                        page_content=file.data.get("content", ""),
                        metadata={
                            **file.meta,
                            "name": file.filename,
                            "created_by": file.user_id,
                            "file_id": file.id,
                            "source": file.filename,
                        },
                    )
                ]
            text_content = file.data.get("content", "")

        else:
            print("!!! DEBUG !!!")
            file_path = file.path

            if file_path:
                # Retrieve the physical file using your storage helper
                file_path = Storage.get_file(file_path)
                loader = Loader(
                    engine=CONTENT_EXTRACTION_ENGINE,
                    TIKA_SERVER_URL=TIKA_SERVER_URL,
                    PDF_EXTRACT_IMAGES=PDF_EXTRACT_IMAGES,
                    DOCUMENT_INTELLIGENCE_ENDPOINT=DOCUMENT_INTELLIGENCE_ENDPOINT,
                    DOCUMENT_INTELLIGENCE_KEY=DOCUMENT_INTELLIGENCE_KEY,
                )
                docs = loader.load(
                    file.filename, 
                    file.meta.get("content_type"), 
                    file_path
                )
                docs = [
                    Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "name": file.filename,
                            "created_by": file.user_id,
                            "file_id": file.id,
                            "source": file.filename,
                        },
                    )
                    for doc in docs
                ]
            else:
                docs = [
                    Document(
                        page_content=file.data.get("content", ""),
                        metadata={
                            **file.meta,
                            "name": file.filename,
                            "created_by": file.user_id,
                            "file_id": file.id,
                            "source": file.filename,
                        },
                    )
                ]
            text_content = " ".join([doc.page_content for doc in docs])

        log.debug(f"text_content: {text_content}")

        # Update the file record with the processed text content and its hash
        Files.update_file_data_by_id(file.id, {"content": text_content})
        hash_val = calculate_sha256_string(text_content)
        Files.update_file_hash_by_id(file.id, hash_val)

        # If embedding and retrieval is enabled, update the vector database
        if not BYPASS_EMBEDDING_AND_RETRIEVAL:
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
                user=file.user_id,  # Using file's owner as the user
            )
                if result:
                    Files.update_file_metadata_by_id(
                        file.id,
                        {
                            "collection_name": collection_name,
                            "status": StatusEnum.COMPLETED,  # Update to completed status
                            "error_message": None,
                        },
                    )
                else:
                    Files.update_file_metadata_by_id(
                        file.id,
                        {
                            "status": StatusEnum.FAILED,  # Mark as failed if result is falsy
                            "error_message": "Document insertion returned false.",
                        },
                    )
            except Exception as e:
                log.exception(e)
                Files.update_file_metadata_by_id(
                    file.id,
                    {
                        "status": StatusEnum.FAILED,
                        "error_message": str(e),
                    },
                )
                raise e
        else:
            Files.update_file_metadata_by_id(
                file.id,
                {
                    "status": StatusEnum.COMPLETED,
                    "error_message": None,
                },
            )

    except Exception as e:
        log.exception(traceback.format_exc())
