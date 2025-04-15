import logging
import time
import json
import uuid
import requests
from datetime import datetime
from typing import Optional, List, Union
from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

from common.vector_store.vector_factory import VECTOR_DB_CLIENT
from common.storage.provider import Storage

from open_webui.retrieval.loaders.main import Loader
from open_webui.config import (
    CONTENT_EXTRACTION_ENGINE,
    TIKA_SERVER_URL,
    PDF_EXTRACT_IMAGES,
    DOCUMENT_INTELLIGENCE_ENDPOINT,
    DOCUMENT_INTELLIGENCE_KEY
)
from open_webui.utils.misc import measure_time
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

CHUNK_SIZE = 20
CHUNK_OVERLAP = 5
PARENT_CHUNK_SIZE = 200
PARENT_CHUNK_OVERLAP = 20
TEXT_SPLITTER = "character"
ENABLE_RAG_PARENT_RETRIEVER = False
TIKTOKEN_ENCODING_NAME = "default"
RAG_EMBEDDING_ENGINE = "openai"
RAG_EMBEDDING_MODEL = "text-embedding-3-small"
RAG_EMBEDDING_BATCH_SIZE = 1
RAG_OPENAI_API_BASE_URL = "https://api.openai.com/v1"
RAG_OPENAI_API_KEY = ""
RAG_OLLAMA_API_KEY = ""
RAG_OLLAMA_BASE_URL = ""
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str = "https://api.openai.com/v1",
    key: str = "",
) -> Optional[list[list[float]]]:
    try:
        r = requests.post(
            f"{url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            json={"input": texts, "model": model},
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return [elem["embedding"] for elem in data["data"]]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        log.exception(f"Error generating openai batch embeddings: {e}")
        return None
    
def generate_embeddings(engine: str, model: str, text: Union[str, list[str]], **kwargs):
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")
    user = kwargs.get("user")

    if engine == "ollama":
        if isinstance(text, list):
            embeddings = generate_ollama_batch_embeddings(
                **{"model": model, "texts": text, "url": url, "key": key, "user": user}
            )
        else:
            embeddings = generate_ollama_batch_embeddings(
                **{
                    "model": model,
                    "texts": [text],
                    "url": url,
                    "key": key,
                    "user": user,
                }
            )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "openai":
        if isinstance(text, list):
            embeddings = generate_openai_batch_embeddings(model, text, url, key)
        else:
            embeddings = generate_openai_batch_embeddings(model, [text], url, key)

        return embeddings[0] if isinstance(text, str) else embeddings
    
def get_embedding_function(
    embedding_engine: str,
    embedding_model: str,
    embedding_function,  # This should be an instance of a model (for example, SentenceTransformer)
    url: str,
    key: str,
    embedding_batch_size: int,
):
    """
    Returns a callable that takes a query (a string or a list of strings) and an optional user,
    and produces embedding vectors.

    If embedding_engine is empty (or "sentence-transformers" or "default"), it assumes that
    embedding_function is a loaded SentenceTransformer (or similar) model and returns a function
    that supports batching.
    
    If embedding_engine is "ollama" or "openai", it wraps a call to generate_embeddings using batching.
    
    Parameters:
      embedding_engine (str): Engine name. e.g., "", "sentence-transformers", "ollama", "openai".
      embedding_model (str): The model name to use (useful for external engines).
      embedding_function: A callable (or a model instance) that provides an encode() method.
      url (str): URL for external API calls (if needed).
      key (str): API key for external engines.
      embedding_batch_size (int): Maximum number of texts to process per batch.

    Returns:
      Callable[[Union[str, List[str]], Optional[Any]], Union[List[List[float]], List[float]]]:
         A function that, given query (or queries) and an optional user parameter,
         returns embedding(s) as a list (or list of lists) of floats.
    """
    # Branch 1: Use a real embedding function (e.g. SentenceTransformer) if engine is empty or set to a default.
    if embedding_engine in ["", "sentence-transformers", "default"]:
        def batched_embedding(query, user=None):
            if isinstance(query, list):
                embeddings = []
                for i in range(0, len(query), embedding_batch_size):
                    batch = query[i:i+embedding_batch_size]
                    # Use the model's encode() method and convert the numpy array to lists.
                    batch_embeddings = embedding_function.encode(batch).tolist()
                    embeddings.extend(batch_embeddings)
                return embeddings
            else:
                # If a single string is provided, wrap it in a list and return the first result.
                return embedding_function.encode([query]).tolist()[0]
        return batched_embedding

    # Branch 2: Use external API embedding engines ("ollama", "openai")
    elif embedding_engine in ["ollama", "openai"]:
        func = lambda query, user=None: generate_embeddings(
            engine=embedding_engine,
            model=embedding_model,
            text=query,
            url=url,
            key=key,
            user=user,
        )
        def generate_multiple(query, user, func):
            if isinstance(query, list):
                embeddings = []
                for i in range(0, len(query), embedding_batch_size):
                    embeddings.extend(func(query[i:i+embedding_batch_size], user=user))
                return embeddings
            else:
                return func(query, user=user)
        return lambda query, user=None: generate_multiple(query, user, func)
    else:
        raise ValueError(f"Unknown embedding engine: {embedding_engine}")

@measure_time
def save_docs_to_vector_db(
    docs: List[Document],
    collection_name: str,
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
        splitter_type = TEXT_SPLITTER
        enable_parent = ENABLE_RAG_PARENT_RETRIEVER
        if splitter_type in ["", "character"]:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
            )
            if enable_parent:
                parent_text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=PARENT_CHUNK_SIZE,
                    chunk_overlap=PARENT_CHUNK_OVERLAP,
                    add_start_index=True,
                )
        elif splitter_type == "token":
            encoding_name = TIKTOKEN_ENCODING_NAME
            text_splitter = TokenTextSplitter(
                encoding_name=encoding_name,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
            )
            if enable_parent:
                parent_text_splitter = TokenTextSplitter(
                    encoding_name=encoding_name,
                    chunk_size=PARENT_CHUNK_SIZE,
                    chunk_overlap=PARENT_CHUNK_OVERLAP,
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
            "engine": RAG_EMBEDDING_ENGINE,
            "model": RAG_EMBEDDING_MODEL
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
    embedding_function = get_embedding_function(
        embedding_engine=RAG_EMBEDDING_ENGINE,
        embedding_model=RAG_EMBEDDING_MODEL,
        embedding_function=embedding_model,  # This should be the instance (e.g., a SentenceTransformer)
        url=(RAG_OPENAI_API_BASE_URL
            if RAG_EMBEDDING_ENGINE == "openai"
            else RAG_OLLAMA_API_BASE_URL),
        key=(RAG_OPENAI_API_KEY
            if RAG_EMBEDDING_ENGINE == "openai"
            else RAG_OLLAMA_API_KEY),
        embedding_batch_size=RAG_EMBEDDING_BATCH_SIZE
    )
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

def process_existing_collection(file):
    """
    Process a file record that already has an associated collection name.
    Queries the vector DB; if found, build docs from the stored vectors;
    otherwise, use the file content.
    """
    print(f"Processing existing collection for file {file}")
    result = VECTOR_DB_CLIENT.query(
        collection_name=f"file-{file.id}",
        filter={"file_id": file.id}
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
    return docs, text_content

def process_file_from_storage(file):
    """
    Process a file record by reading the physical file using storage helpers,
    extracting its content using a loader, and building Document objects.
    """
    if file.path:
        # Retrieve the physical file via your storage helper.
        real_path = Storage.get_file(file.path)
        loader = Loader(
            engine=CONTENT_EXTRACTION_ENGINE,
            TIKA_SERVER_URL=TIKA_SERVER_URL,
            PDF_EXTRACT_IMAGES=PDF_EXTRACT_IMAGES,
            DOCUMENT_INTELLIGENCE_ENDPOINT=DOCUMENT_INTELLIGENCE_ENDPOINT,
            DOCUMENT_INTELLIGENCE_KEY=DOCUMENT_INTELLIGENCE_KEY,
        )
        docs = loader.load(file.filename, file.meta.get("content_type"), real_path)
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
    text_content = " ".join(doc.page_content for doc in docs)
    return docs, text_content