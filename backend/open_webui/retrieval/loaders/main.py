import os
import requests
import logging
import ftfy
import sys
import pandas as pd
import sqlite3
from typing import List, Union

from langchain_community.document_loaders import (
    AzureAIDocumentIntelligenceLoader,
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredXMLLoader,
    YoutubeLoader,
)
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain


from open_webui.retrieval.loaders.mistral import MistralLoader
from open_webui.retrieval.loaders.custom_loader.custom_loader_factory import get_api_loader

from open_webui.env import SRC_LOG_LEVELS, GLOBAL_LOG_LEVEL, DATA_DIR

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

known_source_ext = [
    "go",
    "py",
    "java",
    "sh",
    "bat",
    "ps1",
    "cmd",
    "js",
    "ts",
    "css",
    "cpp",
    "hpp",
    "h",
    "c",
    "cs",
    "sql",
    "log",
    "ini",
    "pl",
    "pm",
    "r",
    "dart",
    "dockerfile",
    "env",
    "php",
    "hs",
    "hsc",
    "lua",
    "nginxconf",
    "conf",
    "m",
    "mm",
    "plsql",
    "perl",
    "rb",
    "rs",
    "db2",
    "scala",
    "bash",
    "swift",
    "vue",
    "svelte",
    "msg",
    "ex",
    "exs",
    "erl",
    "tsx",
    "jsx",
    "hs",
    "lhs",
    "json",
]


class TikaLoader:
    def __init__(self, url, file_path, mime_type=None):
        self.url = url
        self.file_path = file_path
        self.mime_type = mime_type

    def load(self) -> list[Document]:
        with open(self.file_path, "rb") as f:
            data = f.read()

        if self.mime_type is not None:
            headers = {"Content-Type": self.mime_type}
        else:
            headers = {}

        endpoint = self.url
        if not endpoint.endswith("/"):
            endpoint += "/"
        endpoint += "tika/text"

        r = requests.put(endpoint, data=data, headers=headers)

        if r.ok:
            raw_metadata = r.json()
            text = raw_metadata.get("X-TIKA:content", "<No text content found>").strip()
            text = raw_metadata.get("X-TIKA:content", "<No text content found>").strip()

            if "Content-Type" in raw_metadata:
                headers["Content-Type"] = raw_metadata["Content-Type"]

            log.debug("Tika extracted text: %s", text)

            return [Document(page_content=text, metadata=headers)]
        else:
            raise Exception(f"Error calling Tika: {r.reason}")


class DoclingLoader:
    def __init__(self, url, file_path=None, mime_type=None):
        self.url = url.rstrip("/")
        self.file_path = file_path
        self.mime_type = mime_type

    def load(self) -> list[Document]:
        with open(self.file_path, "rb") as f:
            files = {
                "files": (
                    self.file_path,
                    f,
                    self.mime_type or "application/octet-stream",
                )
            }

            payload = {
                "file_path": self.file_path,
                "image_export_mode": "placeholder",
                "table_mode": "accurate",
            }

            endpoint = f"{self.url}/pdf-extractor"
            r = requests.post(endpoint, json=payload)
            print(f"Docling response: {r.json()}")

        if r.ok:
            result = r.json()
            document_data = result.get("document", {})
            text = document_data.get("md_content", "<No text content found>")

            metadata = {"Content-Type": self.mime_type} if self.mime_type else {}

            log.debug("Docling extracted text: %s", text)

            return [Document(page_content=text, metadata=metadata)]
        else:
            error_msg = f"Error calling Docling API: {r.reason}"
            if r.text:
                try:
                    error_data = r.json()
                    if "detail" in error_data:
                        error_msg += f" - {error_data['detail']}"
                except Exception:
                    error_msg += f" - {r.text}"
            raise Exception(f"Error calling Docling: {error_msg}")
        
def _excel_to_sqlite(excel_path: str, db_path: str, table_name: str) -> list[Document]:
    """
    1) Read the Excel file into a pandas DataFrame
    2) Write that DataFrame into a new SQLite database at db_path, table=table_name
    3) Render the DataFrame as Markdown and return it as a single Document
    """
    # 1) Load
    df = pd.read_excel(excel_path)

    # 2) Ensure output dir exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # 3) Write to SQLite
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    # 4) Render to markdown
    md_table = df.to_markdown(index=False)

    # 5) Build metadata exactly as your test expects
    metadata = {
        "source": os.path.basename(excel_path),
        "Content-Type": "application/vnd-ms-excel"
    }

    # 6) Return a list of one Document
    print(f"Metadata: {metadata}")
    print(f"Markdown table: {md_table}")
    return [Document(page_content=md_table, metadata=metadata)]


FILE_LOADERS = {
    "pdf":      lambda fp, ct, kw: PyPDFLoader(fp, extract_images=kw.get("PDF_EXTRACT_IMAGES")),
    "csv":      lambda fp, ct, kw: CSVLoader(fp, autodetect_encoding=True),
    "rst":      lambda fp, ct, kw: UnstructuredRSTLoader(fp, mode="elements"),
    "xml":      lambda fp, ct, kw: UnstructuredXMLLoader(fp),
    "htm":      lambda fp, ct, kw: BSHTMLLoader(fp, open_encoding="unicode_escape"),
    "html":     lambda fp, ct, kw: BSHTMLLoader(fp, open_encoding="unicode_escape"),
    "md":       lambda fp, ct, kw: TextLoader(fp, autodetect_encoding=True),
    "epub":     lambda fp, ct, kw: UnstructuredEPubLoader(fp),
    "docx":     lambda fp, ct, kw: Docx2txtLoader(fp),
    "ppt":      lambda fp, ct, kw: UnstructuredPowerPointLoader(fp),
    "pptx":     lambda fp, ct, kw: UnstructuredPowerPointLoader(fp),
    "msg":      lambda fp, ct, kw: OutlookMessageLoader(fp),
    "txt":      lambda fp, ct, kw: TextLoader(fp, autodetect_encoding=True),
}

EXCEL_EXTS       = {"xls", "xlsx"}
EXCEL_MIME_TYPES = {
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class Loader:
    def __init__(self, engine: str = "", **kwargs):
        self.engine = engine
        self.kwargs = kwargs

    def load(
        self, filename: str, file_content_type: str, file_path: str
    ) -> List[Document]:
        # _get_loader may return either:
        #  - a loader instance (with .load()), or
        #  - a List[Document] directly (Excel branch)
        loader_or_docs: Union[object, List[Document]] = self._get_loader(
            filename, file_content_type, file_path
        )

        # If it's already a list of Documents, use it directly
        if isinstance(loader_or_docs, list):
            raw_docs = loader_or_docs
        else:
            # Otherwise assume it’s a loader and call .load()
            raw_docs = loader_or_docs.load()

        # Normalize/fix text for all docs
        return [
            Document(page_content=ftfy.fix_text(doc.page_content), metadata=doc.metadata)
            for doc in raw_docs
        ]

    def _is_text_file(self, file_ext: str, file_content_type: str) -> bool:
        return file_ext in known_source_ext or (
            file_content_type and file_content_type.find("text/") >= 0
        )

    def _get_loader(self, filename: str, file_content_type: str, file_path: str):
        ext = os.path.splitext(filename)[1].lower().lstrip('.')
        ct = file_content_type.lower()
        print(self.kwargs)
        print(f"Ext: {ext}, CT: {ct}")

        # 1) Excel ingestion
        if ext in EXCEL_EXTS or ct in EXCEL_MIME_TYPES:
            base = os.path.splitext(os.path.basename(file_path))[0]
            db_path = os.path.join(DATA_DIR, 'sqlite', f'{base}.db')
            return _excel_to_sqlite(self, file_path, db_path, table_name=base)

        # 2) Extension-specific API overrides
        if ext == 'txt' and (url := self.kwargs.get('CRAWL4AI_SERVER_URL')):
            print(f"Crawl4ai URL: {url}")
            return get_api_loader('crawl4ai', url, file_path, ct)
        if ext == 'pdf' and (url := self.kwargs.get('DOCLING_SERVER_URL')):
            return get_api_loader('docling', url, file_path, ct)

        # 3) Global API engines by engine name
        # if self.engine in ('docling', 'crawl4ai'):
        #     url_key = f"{self.engine.upper()}_SERVER_URL"
        #     if (url := self.kwargs.get(url_key)):
        #         return get_api_loader(self.engine, url, file_path, ct)

        # 4) Azure Document Intelligence
        if self.engine == 'document_intelligence':
            ep = self.kwargs.get('DOCUMENT_INTELLIGENCE_ENDPOINT')
            key = self.kwargs.get('DOCUMENT_INTELLIGENCE_KEY')
            if ep and key and (ext in {'pdf','docx','ppt','pptx'} or ct.startswith('application/')):
                return AzureAIDocumentIntelligenceLoader(
                    file_path=file_path, api_endpoint=ep, api_key=key)

        # 5) Mistral OCR
        if self.engine == 'mistral_ocr' and self.kwargs.get('MISTRAL_OCR_API_KEY') and ext == 'pdf':
            return MistralLoader(api_key=self.kwargs['MISTRAL_OCR_API_KEY'], file_path=file_path)

        # 6) Static file-based loaders
        if fn := FILE_LOADERS.get(ext):
            return fn(file_path, ct, self.kwargs)

        # 7) EPUB fallback
        if ct == 'application/epub+zip':
            return UnstructuredEPubLoader(file_path)

        # 8) Default
        return TextLoader(file_path, autodetect_encoding=True)
