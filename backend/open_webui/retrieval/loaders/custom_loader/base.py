import requests
from typing import Dict, List
from langchain_core.documents import Document

class BaseCustomLoader:
    """
    Base class for custom document extractors.
    Subclasses should define `endpoint_path` and `default_payload`.
    """
    endpoint_path: str
    default_payload: Dict = {}

    def __init__(
        self,
        url: str,
        file_path: str,
        mime_type: str = None,
        extra_payload: Dict = None,
    ):
        self.url = url.rstrip("/")
        self.file_path = file_path
        self.mime_type = mime_type
        # merge default_payload with any extras
        self.payload = {**self.default_payload, **(extra_payload or {})}