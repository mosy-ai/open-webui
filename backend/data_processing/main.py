import os
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from data_processing.pdf_extractors.extraction_service import (
    MarkdownExtractionService,
)
from data_processing.pdf_extractors.factories import (
    DoclingFactory,
    MarkItDownFactory,
)
from data_processing.url_extractors.crawl4ai_adapter import (
    Crawl4AIAdapter,
)

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI()

class ExtractionRequest(BaseModel):
    file_path: str
    image_export_mode: str = "placeholder"
    table_mode: str = "accurate"
    
class UrlsRequest(BaseModel):
    urls: List[str]

# Dummy Document model for the response.
class Document(BaseModel):
    page_content: str
    metadata: dict

@app.post("/pdf-extractor")
async def pdf_extractor_endpoint(payload: ExtractionRequest):
    """
    Receives a JSON payload containing file_path and extraction options,
    processes the file, and returns extracted document content in JSON.
    """
    file_path = payload.file_path
    # log.debug(f"Received file path: {file_path}")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
    
    if os.path.getsize(file_path) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        # Insert your extraction logic here. For demonstration, we'll use dummy text.
        factories = [
            DoclingFactory(),
            MarkItDownFactory()
        ] 
        extraction_service = MarkdownExtractionService(factories)
        result = extraction_service.extract(file_path)
        document_data = {"md_content": result}
        response_json = {"document": document_data}
        return response_json
    except Exception as e:
        log.exception("An error occurred during extraction.")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the file.
        if os.path.exists(file_path):
            os.remove(file_path)
            
@app.post("/url-extractor")
async def url_extractor_batch(req: UrlsRequest):
    adapter = Crawl4AIAdapter()
    documents = []

    for url in req.urls:
        try:
            crawl_result = await adapter.crawl(url)
        except Exception as e:
            log.exception("Error crawling %s", url)

        documents.append({
            "source": url,
            "md_content": crawl_result.markdown
        })

    return {"documents": documents}