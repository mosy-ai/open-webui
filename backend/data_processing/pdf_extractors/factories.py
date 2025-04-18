from data_processing.pdf_extractors.abstract_extraction import (
    AbstractMarkdownExtractor,
)
from data_processing.pdf_extractors.markdown_extractor import (
    DoclingExtractor,
    MarkItDownExtractor,
    GeminiExtractor,
)   


class AbstractExtractorFactory:
    def create_extractor(self) -> AbstractMarkdownExtractor:
        raise NotImplementedError


class MarkItDownFactory(AbstractExtractorFactory):
    def create_extractor(self) -> AbstractMarkdownExtractor:
        return MarkItDownExtractor()


class DoclingFactory(AbstractExtractorFactory):
    def create_extractor(self) -> AbstractMarkdownExtractor:
        return DoclingExtractor()
    
class GeminiFactory(AbstractExtractorFactory):
    def create_extractor(self) -> AbstractMarkdownExtractor:
        return GeminiExtractor()