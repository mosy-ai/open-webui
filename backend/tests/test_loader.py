import pytest
import ftfy

from open_webui.env import DATA_DIR
from open_webui.storage.provider import Storage
from open_webui.retrieval.loaders.main import Loader, DoclingLoader

# --- Fake TikaLoader for Testing ---
class FakeTikaLoader:
    def __init__(self, url, file_path, mime_type):
        self.url = url
        self.file_path = file_path
        self.mime_type = mime_type

    def load(self):
        # Mimic the behavior of a Tika loader returning a document-like object.
        class FakeDoc:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata
        return [FakeDoc("Fake content from TikaLoader", {"source": "TikaFake"})]

# --- Dummy Request and Config Setup ---
class DummyConfig:
    CONTENT_EXTRACTION_ENGINE = "docling"
    DOCLING_SERVER_URL = "http://127.0.0.1:8000"
    TIKA_SERVER_URL = "http://dummy-tika-server"  # not used in docling branch
    PDF_EXTRACT_IMAGES = False
    DOCUMENT_INTELLIGENCE_ENDPOINT = "dummy_endpoint"
    DOCUMENT_INTELLIGENCE_KEY = "dummy_key"
    CRAWL4AI_SERVER_URL = "http://127.0.0.1:8000"
    
class DummyAppState:
    config = DummyConfig()

class DummyRequest:
    state = DummyAppState()
    
# --- Test Case ---
def test_loader_docling_branch(monkeypatch):
    """
    Test that Loader correctly selects the DoclingLoader branch when using the 'docling'
    engine with a non-text file.
    """
    # Create a dummy request to simulate configuration in the app state.
    dummy_request = DummyRequest()
    
    # Instantiate Loader with engine "docling". Note that we supply DOCLING_SERVER_URL
    # as an extra keyword argument.
    
    filename = "08a5ea73-ea56-4d1b-b5a1-c74533d9b952_Q&A.pdf"
    file_path = "08a5ea73-ea56-4d1b-b5a1-c74533d9b952_Q&A.pdf"
    file_content_type = "application/octet-stream"
    
    file_path = Storage.get_file(file_path)
    print(f"File path: {file_path}")
    
    loader_instance = Loader(
        engine=dummy_request.state.config.CONTENT_EXTRACTION_ENGINE,
        TIKA_SERVER_URL=dummy_request.state.config.TIKA_SERVER_URL,
        PDF_EXTRACT_IMAGES=dummy_request.state.config.PDF_EXTRACT_IMAGES,
        DOCUMENT_INTELLIGENCE_ENDPOINT=dummy_request.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
        DOCUMENT_INTELLIGENCE_KEY=dummy_request.state.config.DOCUMENT_INTELLIGENCE_KEY,
        DOCLING_SERVER_URL=dummy_request.state.config.DOCLING_SERVER_URL,  # necessary for docling branch
    )
    
    docs = loader_instance.load(
        filename, 
        file_content_type, 
        file_path
    )

    # Verify that the fake loader's document is correctly wrapped into a Document.
    print(f"Docs: {docs}")
    expected_content = ftfy.fix_text("Fake content from DoclingLoader")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.page_content == expected_content
    assert doc.metadata == {"source": "DoclingFake"}
    
def test_loader_excel_branch(monkeypatch):
    """
    Test that Loader correctly ingests Excel files into SQLite and returns
    a single Document containing the sheet as a markdown table.
    """
    # 1) Prepare a dummy Excel file path
    
    dummy_request = DummyRequest()
    
    filename = "Build Sheet.xlsx"
    file_path = "Build Sheet.xlsx"
    file_content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    file_path = Storage.get_file(file_path)
    
    loader = Loader(
        engine=dummy_request.state.config.CONTENT_EXTRACTION_ENGINE,
        TIKA_SERVER_URL=dummy_request.state.config.TIKA_SERVER_URL,
        PDF_EXTRACT_IMAGES=dummy_request.state.config.PDF_EXTRACT_IMAGES,
        DOCUMENT_INTELLIGENCE_ENDPOINT=dummy_request.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
        DOCUMENT_INTELLIGENCE_KEY=dummy_request.state.config.DOCUMENT_INTELLIGENCE_KEY,
        DOCLING_SERVER_URL=dummy_request.state.config.DOCLING_SERVER_URL,
    )
    
    docs = loader.load(
        filename,
        file_content_type,
        file_path
    )
    assert len(docs) == 1

def test_loader_crawl4ai_branch(monkeypatch):
    """
    Test that Loader correctly ingests URLs into SQLite and returns
    a single Document containing the URL's content.
    """
    dummy_request = DummyRequest()
    
    filename = "url.txt"
    file_path = "url.txt"
    file_content_type = "text/plain"
    
    file_path = Storage.get_file(file_path)
    
    loader = Loader(
        engine=dummy_request.state.config.CONTENT_EXTRACTION_ENGINE,
        TIKA_SERVER_URL=dummy_request.state.config.TIKA_SERVER_URL,
        PDF_EXTRACT_IMAGES=dummy_request.state.config.PDF_EXTRACT_IMAGES,
        DOCUMENT_INTELLIGENCE_ENDPOINT=dummy_request.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
        DOCUMENT_INTELLIGENCE_KEY=dummy_request.state.config.DOCUMENT_INTELLIGENCE_KEY,
        DOCLING_SERVER_URL=dummy_request.state.config.DOCLING_SERVER_URL,
        CRAWL4AI_SERVER_URL=dummy_request.state.config.CRAWL4AI_SERVER_URL,
    )
    
    docs = loader.load(
        filename,
        file_content_type,
        file_path)
    
    assert len(docs) == 1
    doc = docs[0]
    print(f"Doc: {doc}")
    assert doc.page_content == "Fake content from Crawl4aiLoader"
    assert doc.metadata == {"source": "Crawl4aiFake"}
