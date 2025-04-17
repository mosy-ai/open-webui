import pytest
from data_processing.pdf_extractors.factories import (
    DoclingFactory,
    MarkItDownFactory,
)
from data_processing.pdf_extractors.extraction_service import (
    MarkdownExtractionService,
)
from data_processing.url_extractors.crawl4ai_adapter import (
    Crawl4AIAdapter,
)

def test_pdf_extractor():
    file_path = "data/uploads/7234d30e-6134-458d-a977-e49cdefbfdff_real_internet_setup.pdf"
    factories = [
            DoclingFactory(),
            MarkItDownFactory()
        ] 
    extraction_service = MarkdownExtractionService(factories)
    result = extraction_service.extract(file_path)
    print(result)
    
    assert True
    
@pytest.mark.asyncio
async def test_url_extractor():
    url = "https://eximbank.com.vn/"
    crawler_adapter = Crawl4AIAdapter()
    result = await crawler_adapter.crawl(url)
    print(result)
    assert True