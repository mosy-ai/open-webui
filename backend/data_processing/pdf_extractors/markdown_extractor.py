import os
import tempfile
import shutil

from docling.document_converter import DocumentConverter
from markitdown import MarkItDown
from google import genai
from pypdf import PdfReader, PdfWriter

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
        
class GeminiExtractor(AbstractMarkdownExtractor):
    @validate_output(lambda result: result is not None and result.strip() != "")
    def extract(self, source: str) -> str:
        """
        Downloads the PDF from the URL provided in `source` (or uses a local path),
        splits it into individual pages, sends each page to the Gemini API for OCR
        and markdown/chunk conversion, then merges all the page outputs into a single
        Markdown string which is returned.
        """
        base_dir = tempfile.mkdtemp(prefix="gemini_")
        split_dir = os.path.join(base_dir, "split")
        ocr_dir = os.path.join(base_dir, "ocr")
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(ocr_dir, exist_ok=True)

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        prompt = """
        You are an expert OCR and Markdown converter. Please process the incoming PDF page and output a single, well‑formed GitHub‑flavored Markdown document that:

        1. **Text → Markdown**  
        Convert all recognized text into Markdown paragraphs. Preserve headings and subheadings using `#`/`##` syntax where appropriate.

        2. **Tables → Markdown or HTML**  
        - If a table is simple (rows/columns), render it as a Markdown table.  
        - If a table is complex (merged cells, nested headers), render it using HTML `<table>...</table>`.

        3. **Images & Charts**  
        - For each image or chart, insert a Markdown image tag with descriptive alt text, e.g.  
            `![Figure 1: Sales over time](figure_1.png)`  
        - Do not embed base64—just use a placeholder filename (we’ll replace it later).

        4. **No Code‑Fence Wrapping**  
        Don’t surround your entire output in triple backticks.

        5. **Semantic Chunking**  
        Break the document into coherent sections of roughly **250–1000 words** each—these will become RAG embeddings.  

        6. **Chunk Tags**  
        Wrap each section in `<chunk>…</chunk>` HTML tags.

        Now process the PDF page below.
        """

        try:
            # 1. Split PDF into single-page files
            reader = PdfReader(source)
            for idx, page in enumerate(reader.pages, start=1):
                page_pdf = os.path.join(split_dir, f"page_{idx}.pdf")
                writer = PdfWriter()
                writer.add_page(page)
                with open(page_pdf, "wb") as f:
                    writer.write(f)

                # 2. Upload and process with Gemini
                file_ref = client.files.upload(
                    file=page_pdf,
                    config={"display_name": f"document_page_{idx}"}
                )
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt, file_ref]
                )

                # 3. Save markdown output per page
                txt_path = os.path.join(ocr_dir, f"page_{idx}.txt")
                with open(txt_path, "w", encoding="utf-8") as out:
                    out.write(response.text)

            # 4. Merge all page markdowns
            merged = []
            for fname in sorted(os.listdir(ocr_dir), key=lambda x: int(''.join(filter(str.isdigit, x)))):
                with open(os.path.join(ocr_dir, fname), "r", encoding="utf-8") as infile:
                    merged.append(infile.read())
            return "\n\n".join(merged)

        except Exception as e:
            raise Exception(f"Error converting document to markdown with Gemini: {e}")

        finally:
            shutil.rmtree(base_dir, ignore_errors=True)
