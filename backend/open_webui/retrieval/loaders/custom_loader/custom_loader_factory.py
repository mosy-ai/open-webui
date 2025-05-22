from typing import Dict, Type

from open_webui.retrieval.loaders.custom_loader.pdf2markdown import DoclingLoader
from open_webui.retrieval.loaders.custom_loader.url2markdown import Crawl4aiLoader

def get_api_loader(
    engine: str,
    url: str,
    file_path: str,
    mime_type: str = None,
    extra_payload: Dict = None,
) -> object:
    """
    Factory for API-based loaders: returns an extractor with .load().

    Supported engines: 'docling', 'crawl4ai'
    """
    mapping: Dict[str, Type] = {
        'docling': DoclingLoader,
        'crawl4ai': Crawl4aiLoader,
    }
    engine_key = engine.lower()
    if engine_key not in mapping:
        raise ValueError(f"Unknown API engine '{engine}'")
    cls = mapping[engine_key]
    return cls(
        url=url,
        file_path=file_path,
        mime_type=mime_type,
        extra_payload=extra_payload,
    )