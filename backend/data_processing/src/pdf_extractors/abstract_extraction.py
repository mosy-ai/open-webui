from abc import ABC, abstractmethod


class AbstractMarkdownExtractor(ABC):
    @abstractmethod
    def extract(self, source: str) -> str:
        """
        Extract markdown text from the given source.
        Should raise an exception if extraction fails or if the output is invalid.
        """
        pass