from typing import List, Dict
import requests
from langchain_core.documents import Document

from open_webui.retrieval.loaders.custom_loader.base import BaseCustomLoader

class Crawl4aiLoader(BaseCustomLoader):
    """
    Reads a text file of URLs, posts each URL to /url-extractor endpoint,
    and returns a Document per URL with valid string content.
    """
    endpoint_path = "/url-extractor"
    default_payload: Dict = {}

    def load(self) -> list[Document]:
        # 1) Read all URLs
        with open(self.file_path, "r") as f:
            urls = [line.strip() for line in f if line.strip()]

        if not urls:
            return []

        # 2) POST them in one go
        endpoint = f"{self.url}{self.endpoint_path}"
        response = requests.post(endpoint, json={"urls": urls})
        response.raise_for_status()

        # 3) Parse response
        docs = []
        for item in response.json().get("documents", []):
            md = item.get("md_content", "<No content>")
            src = item.get("source")
            docs.append(Document(page_content=md, metadata={"source": src}))
        return docs