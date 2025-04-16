from typing import List

from data_processing.pdf_extractors.factories import AbstractExtractorFactory


class MarkdownExtractionService:
    def __init__(self, factories: List[AbstractExtractorFactory]):
        self.factories = factories

    def extract(self, source: str) -> str:
        last_exception = None
        for factory in self.factories:
            extractor = factory.create_extractor()
            try:
                return extractor.extract(source)
            except Exception as ex:
                # Log or print the error; in production use a logger.
                print(f"Extractor {extractor.__class__.__name__} failed: {ex}")
                last_exception = ex
        raise Exception("All extraction services failed") from last_exception