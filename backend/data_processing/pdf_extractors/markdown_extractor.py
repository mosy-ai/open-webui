from docling.document_converter import DocumentConverter
from markitdown import MarkItDown

from data_processing.pdf_extractors.abstract_extraction import (
    AbstractMarkdownExtractor,
)
from data_processing.pdf_extractors.validator import validate_output


class MarkItDownExtractor(AbstractMarkdownExtractor):
    @validate_output(lambda result: result is not None and result.strip() != "")
    def extract(self, source: str) -> str:
        """
        Downloads the document from the URL provided in `source` and converts it using MarkItDown.
        Returns the markdown content.
        """
        try:
            md = MarkItDown()
            result = md.convert(source)
            return result.text_content
        except Exception as e:
            raise Exception(f"Error converting document to markdown: {e}")


class DoclingExtractor(AbstractMarkdownExtractor):
    @validate_output(lambda result: result is not None and result.strip() != "")
    def extract(self, source: str) -> str:
        """
        Downloads the document from the URL provided in `source`,
        converts it using DocumentConverter, and returns the markdown content.
        """
        try:
            converter = DocumentConverter()
            result = converter.convert(source)
            return result.document.export_to_markdown()
        except Exception as e:
            raise Exception(f"Error converting document to markdown: {e}")