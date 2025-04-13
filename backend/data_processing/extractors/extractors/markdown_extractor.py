import os
import requests
import tempfile
from openai import OpenAI
from markitdown import MarkItDown
from docling.document_converter import DocumentConverter
from data_processing.extractors.abstract_extractor import AbstractMarkdownExtractor
from data_processing.extractors.validators import validate_output

def download_to_temp_file(url: str) -> str:
    """
    Downloads the file at the given URL and writes it to a temporary file.
    Returns the path to the temporary file.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors.
    
    # Extract a file extension from the URL (if available).
    suffix = os.path.splitext(url)[1]
    
    # Create a temporary file that will persist after closing (for use by the converters).
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(response.content)
        tmp_file.flush()
        return tmp_file.name

class MarkItDownExtractor(AbstractMarkdownExtractor):
    def __init__(self, client=None, model: str = "gpt-4o"):
        # If no client is provided, we'll use basic usage.
        self.client = client
        self.model = model

    @validate_output(lambda result: result is not None and result.strip() != "")
    def extract(self, source: str) -> str:
        """
        Downloads the document from the URL provided in `source` and converts it using MarkItDown.
        Uses basic usage (no LLM) if client is None; otherwise uses the provided LLM client.
        Returns the markdown content.
        """
        temp_file_path = download_to_temp_file(source)
        try:
            md = MarkItDown(llm_client=self.client, llm_model=self.model)
            result = md.convert(temp_file_path)
            return result.text_content
        finally:
            os.remove(temp_file_path)

class DoclingExtractor(AbstractMarkdownExtractor):
    @validate_output(lambda result: result is not None and result.strip() != "")
    def extract(self, source: str) -> str:
        """
        Downloads the document from the URL provided in `source`,
        converts it using DocumentConverter, and returns the markdown content.
        """
        # Download the document first.
        temp_file_path = download_to_temp_file(source)
        try:
            converter = DocumentConverter()
            result = converter.convert(temp_file_path)
            return result.document.export_to_markdown()
        finally:
            # Clean up the temporary file.
            os.remove(temp_file_path)
