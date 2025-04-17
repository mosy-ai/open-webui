from typing import List
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
    
from data_processing.url_extractors.interface import CrawlerInterface, CrawlResult

class CrawlError(Exception):
    """Custom exception for crawl failures"""
    pass

class Crawl4AIAdapter(CrawlerInterface):
    def __init__(self):
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False
        )
    
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),     # Retry up to 3 times
        wait=wait_fixed(3),             # Wait 3 seconds between attempts
        retry=retry_if_exception_type(CrawlError)
    )
    async def crawl(
        self, 
        url: str, 
        excluded_tags: list = None,
        exclude_social_media_links: bool = None
    ) -> List[CrawlResult]:
        """Crawl the given URL and return a CrawlResult"""
        # Build the markdown generator with custom options
        markdown_options = {
            "ignore_links": False,
            "excluded_tags": excluded_tags,  # If None, your generator should handle defaults
            "exclude_social_media_links": exclude_social_media_links
        }
        
        # Build configuration parameters
        config_params = CrawlerRunConfig(
            cache_mode=CacheMode.DISABLED,
            markdown_generator=DefaultMarkdownGenerator(
                options=markdown_options,
                content_filter=PruningContentFilter(
                    threshold=0.48,
                    threshold_type="fixed",
                )
            )
        )
        
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            result = await crawler.arun(
                url=url, 
                config=config_params,
            )
            
            if not result.success:
                raise CrawlError(f"Unsuccessful crawl for {result.url}. Error: {result.error_message}")
            
            return CrawlResult(
                url=result.url,
                type="markdown",
                markdown=result.markdown,
                error_message=None,
            )