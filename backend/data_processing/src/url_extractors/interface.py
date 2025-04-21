from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Literal

# ---------- Data Models ----------
@dataclass
class CrawlResult:
    """Universal result container for crawler operations"""
    url: str
    type: Literal["markdown", "pdf"] 
    markdown: str 
    error_message: Optional[str] = None
    
    
class CrawlerInterface(ABC):
    """Abstract base class for asynchronous web crawlers"""
    
    @abstractmethod
    async def crawl(
        self,
        url: str,
    ) -> List[CrawlResult]:
        """
        Crawl a single URL
        Args:
            url: URL to crawl
        Returns:
            CrawlResult with unified structure
        """
        pass