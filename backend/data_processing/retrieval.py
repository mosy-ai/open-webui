import logging
import json
import uuid
from datetime import datetime
from typing import Optional, List
from fastapi import HTTPException, status
from pydantic import BaseModel
from langchain_core.documents import Document
from datetime import datetime

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

from open_webui.models.files import Files, StatusEnum
from open_webui.utils.misc import (
    calculate_sha256_string,
)
from open_webui.storage.provider import Storage
from open_webui.retrieval.vector.connector import VECTOR_DB_CLIENT
from open_webui.retrieval.loaders.main import Loader
from open_webui.config import (
    ENV,
    BYPASS_EMBEDDING_AND_RETRIEVAL,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    RAG_RERANKING_MODEL_AUTO_UPDATE,
    RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
    UPLOAD_DIR,
    DEFAULT_LOCALE,
    CONTENT_EXTRACTION_ENGINE,
    TIKA_SERVER_URL,
    PDF_EXTRACT_IMAGES,
    DOCUMENT_INTELLIGENCE_ENDPOINT,
    DOCUMENT_INTELLIGENCE_KEY

)

from data_processing.utils import get_embedding_function
from open_webui.constants import ERROR_MESSAGES
import traceback

log = logging.getLogger(__name__)

class ProcessFileForm(BaseModel):
    file_id: str
    content: Optional[str] = None
    collection_name: Optional[str] = None

def save_docs_to_vector_db(
    docs: List,  # assuming docs is a list of Document objects
    collection_name: str,
    config,         # configuration object containing attributes such as TEXT_SPLITTER, CHUNK_SIZE, etc.
    ef,             # embedding function helper dependency (i.e., an engine or similar)
    metadata: Optional[dict] = None,
    overwrite: bool = False,
    split: bool = True,
    add: bool = False,
    user=None,
) -> bool:
    def _get_docs_info(docs: List) -> str:
        docs_info = set()
        # Trying to select relevant metadata identifying the document.
        for doc in docs:
            doc_metadata = getattr(doc, "metadata", {})
            doc_name = doc_metadata.get("name", "")
            if not doc_name:
                doc_name = doc_metadata.get("title", "")
            if not doc_name:
                doc_name = doc_metadata.get("source", "")
            if doc_name:
                docs_info.add(doc_name)
        return ", ".join(docs_info)

    log.info(f"save_docs_to_vector_db: document {_get_docs_info(docs)} {collection_name}")

    # Check for duplicate content based on hash in metadata.
    if metadata and "hash" in metadata:
        result = VECTOR_DB_CLIENT.query(
            collection_name=collection_name,
            filter={"hash": metadata["hash"]},
        )
        if result is not None:
            existing_doc_ids = result.ids[0]
            if existing_doc_ids:
                log.info(f"Document with hash {metadata['hash']} already exists")
                raise ValueError(ERROR_MESSAGES.DUPLICATE_CONTENT)

    # If splitting is enabled, split the documents.
    if split:
        if config.TEXT_SPLITTER in ["", "character"]:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                add_start_index=True,
            )
        elif config.TEXT_SPLITTER == "token":
            log.info(f"Using token text splitter: {config.TIKTOKEN_ENCODING_NAME}")
            tiktoken.get_encoding(str(config.TIKTOKEN_ENCODING_NAME))
            text_splitter = TokenTextSplitter(
                encoding_name=str(config.TIKTOKEN_ENCODING_NAME),
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                add_start_index=True,
            )
        else:
            raise ValueError(ERROR_MESSAGES.DEFAULT("Invalid text splitter"))

        docs = text_splitter.split_documents(docs)

    if len(docs) == 0:
        raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)

    texts = [doc.page_content for doc in docs]
    metadatas = [
        {
            **doc.metadata,
            **(metadata if metadata else {}),
            "embedding_config": json.dumps(
                {
                    "engine": config.RAG_EMBEDDING_ENGINE,
                    "model": config.RAG_EMBEDDING_MODEL,
                }
            ),
        }
        for doc in docs
    ]

    # ChromaDB does not like datetime, list or dict formats for metadata so convert them to string.
    for meta in metadatas:
        for key, value in meta.items():
            if isinstance(value, (datetime, list, dict)):
                meta[key] = str(value)

    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=collection_name):
            log.info(f"collection {collection_name} already exists")
            if overwrite:
                VECTOR_DB_CLIENT.delete_collection(collection_name=collection_name)
                log.info(f"deleting existing collection {collection_name}")
            elif not add:
                log.info(f"collection {collection_name} already exists, overwrite is False and add is False")
                return True

        log.info(f"adding to collection {collection_name}")
        embedding_function = get_embedding_function(
            config.RAG_EMBEDDING_ENGINE,
            config.RAG_EMBEDDING_MODEL,
            ef,
            (config.RAG_OPENAI_API_BASE_URL if config.RAG_EMBEDDING_ENGINE == "openai" else config.RAG_OLLAMA_BASE_URL),
            (config.RAG_OPENAI_API_KEY if config.RAG_EMBEDDING_ENGINE == "openai" else config.RAG_OLLAMA_API_KEY),
            config.RAG_EMBEDDING_BATCH_SIZE,
        )

        # Clean newlines from texts and generate embeddings.
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        embeddings = embedding_function(cleaned_texts, user=user)

        items = [
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "vector": embeddings[idx],
                "metadata": metadatas[idx],
            }
            for idx, text in enumerate(texts)
        ]

        VECTOR_DB_CLIENT.insert(
            collection_name=collection_name,
            items=items,
        )

        return True
    except Exception as e:
        log.exception(e)
        raise e

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
