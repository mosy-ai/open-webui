from typing import Dict, List
import requests
from langchain_core.documents import Document

from open_webui.retrieval.loaders.custom_loader.base import BaseCustomLoader

class DoclingLoader(BaseCustomLoader):
    """
    Loader for Docling: sends the file path in JSON payload only.
    """
    endpoint_path = "/pdf-extractor"
    default_payload: Dict = {"image_export_mode": "placeholder", "table_mode": "accurate"}

    def load(self) -> List[Document]:
        payload = {"file_path": self.file_path, **self.default_payload}
        endpoint = f"{self.url}{self.endpoint_path}"
        response = requests.post(endpoint, json=payload)
        if not response.ok:
            raise RuntimeError(f"Docling error: {response.status_code} {response.text}")
        doc = response.json().get("document", {})
        text = doc.get("md_content", "<No content>")
        meta = {"Content-Type": self.mime_type} if self.mime_type else {}
        return [Document(page_content=text, metadata=meta)]