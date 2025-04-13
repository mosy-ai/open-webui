from data_processing.extractors.markdown_extractor import MarkItDownExtractor, DoclingExtractor
from data_processing.extractors.abstract_extractor import AbstractMarkdownExtractor

class AbstractExtractorFactory:
    def create_extractor(self) -> AbstractMarkdownExtractor:
        raise NotImplementedError

class MarkItDownFactory(AbstractExtractorFactory):
    def create_extractor(self) -> AbstractMarkdownExtractor:
        return MarkItDownExtractor()

class DoclingFactory(AbstractExtractorFactory):
    def create_extractor(self) -> AbstractMarkdownExtractor:
        return DoclingExtractor()