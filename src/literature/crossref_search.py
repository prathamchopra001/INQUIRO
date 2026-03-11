# -*- coding: utf-8 -*-
"""
CrossRef API client for Inquiro.

CrossRef is the official DOI registration agency with metadata for 140M+ works.
Excellent for citation data, DOI resolution, and publication metadata.

API Docs: https://api.crossref.org/swagger-ui/index.html
Polite Pool: Include email in User-Agent for faster rate limits (50 req/sec vs 1 req/sec)
"""

import time
import logging
import threading
from typing import Optional, List

import httpx

from src.literature.models import Paper
from config.settings import settings

logger = logging.getLogger(__name__)


class CrossRefClient:
    """
    Client for the CrossRef REST API.
    
    CrossRef has the most comprehensive DOI metadata available.
    With a polite User-Agent (including email), you get 50 req/sec.
    
    Usage:
        client = CrossRefClient(email="you@example.com")
        papers = client.search_papers("diabetes readmission machine learning")
    """
    
    BASE_URL = "https://api.crossref.org"
    
    # Class-level lock for thread-safe rate limiting
    _global_lock = threading.Lock()
    _global_last_request_time: float = 0.0
    
    def __init__(
        self,
        email: Optional[str] = None,
        requests_per_minute: int = 45,  # Conservative for polite pool (max 50/sec with email)
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize CrossRef client.
        
        Args:
            email: Your email for the "polite pool" (50x faster rate limits).
                   Highly recommended - CrossRef uses this for abuse contact only.
            requests_per_minute: Rate limit (45 RPM is very conservative)
            max_retries: Retry count for transient errors
            timeout: Request timeout in seconds
        """
        self.email = email or "inquiro-research@example.com"
        self.max_retries = max_retries
        self.timeout = timeout
        
        self._min_interval = 60.0 / requests_per_minute
        
        # Polite User-Agent gets priority access
        headers = {
            "Accept": "application/json",
            "User-Agent": f"Inquiro/1.0 (https://github.com/inquiro; mailto:{self.email})",
        }
        
        self._http = httpx.Client(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=self.timeout,
        )
    
    def _wait_for_rate_limit(self):
        """Thread-safe rate limiting."""
        with CrossRefClient._global_lock:
            elapsed = time.time() - CrossRefClient._global_last_request_time
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                logger.debug(f"CrossRef rate limit: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            CrossRefClient._global_last_request_time = time.time()
    
    def _request(self, method: str, url: str, **kwargs) -> dict:
        """Make HTTP request with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()
            
            try:
                response = self._http.request(method, url, **kwargs)
                
                if response.status_code == 200:
                    return response.json()
                
                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"CrossRef rate limited (429). Waiting {wait}s... (attempt {attempt})")
                    time.sleep(wait)
                    continue
                
                if response.status_code >= 500:
                    logger.warning(f"CrossRef server error ({response.status_code}). Retrying...")
                    time.sleep(1)
                    continue
                
                logger.error(f"CrossRef API error {response.status_code}: {response.text[:200]}")
                return {}
                
            except (httpx.TimeoutException, httpx.RequestError) as e:
                logger.warning(f"CrossRef request error: {e}. Retrying... (attempt {attempt})")
                time.sleep(1)
                continue
        
        logger.error(f"CrossRef: all {self.max_retries} retries failed for {url}")
        return {}
    
    def _parse_work(self, data: dict) -> Optional[Paper]:
        """Convert CrossRef work to Paper model."""
        if not data:
            return None
        
        try:
            doi = data.get("DOI", "")
            
            # Extract title (CrossRef returns as list)
            titles = data.get("title", [])
            title = titles[0] if titles else "Unknown Title"
            
            # Extract authors
            authors = []
            for author in data.get("author", []):
                given = author.get("given", "")
                family = author.get("family", "")
                if given and family:
                    authors.append(f"{given} {family}")
                elif family:
                    authors.append(family)
            
            # Extract year from published-print or published-online
            year = None
            for date_field in ["published-print", "published-online", "created"]:
                date_parts = data.get(date_field, {}).get("date-parts", [[]])
                if date_parts and date_parts[0]:
                    year = date_parts[0][0]
                    break
            
            # Extract abstract (if available)
            abstract = data.get("abstract", "")
            # CrossRef abstracts often have JATS XML tags, strip them
            if abstract:
                import re
                abstract = re.sub(r'<[^>]+>', '', abstract)
            
            # Extract venue
            container = data.get("container-title", [])
            venue = container[0] if container else None
            
            # Citation count (CrossRef calls it "is-referenced-by-count")
            citation_count = data.get("is-referenced-by-count", 0)
            
            # URL - prefer DOI URL
            url = f"https://doi.org/{doi}" if doi else data.get("URL")
            
            # Check for open access via license
            is_open_access = False
            licenses = data.get("license", [])
            for lic in licenses:
                lic_url = lic.get("URL", "").lower()
                if "creativecommons" in lic_url or "open" in lic_url:
                    is_open_access = True
                    break
            
            # Try to find PDF link
            pdf_url = None
            for link in data.get("link", []):
                if link.get("content-type") == "application/pdf":
                    pdf_url = link.get("URL")
                    break
            
            return Paper(
                paper_id=f"crossref_{doi.replace('/', '_')}" if doi else f"crossref_{hash(title)}",
                doi=doi,
                title=title,
                abstract=abstract if abstract else None,
                year=year,
                venue=venue,
                authors=authors,
                citation_count=citation_count,
                influential_citation_count=0,
                url=url,
                pdf_url=pdf_url,
                is_open_access=is_open_access,
            )
            
        except Exception as e:
            logger.warning(f"CrossRef: failed to parse work: {e}")
            return None
    
    def search_papers(
        self,
        query: str,
        limit: int = None,
        year_range: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[Paper]:
        """
        Search CrossRef for papers matching query.
        
        Args:
            query: Search keywords
            limit: Max results (default from settings)
            year_range: Optional "2020-2024" or "2020-" filter
            fields_of_study: Not used (CrossRef doesn't support this filter)
        
        Returns:
            List of Paper objects
        """
        # Validate query length
        if len(query) > 100:
            logger.warning(f"CrossRef query too long ({len(query)} chars). Truncating.")
            query = query[:80]
        
        limit = limit or settings.literature.max_papers_per_search
        
        params = {
            "query": query,
            "rows": min(limit, 50),  # CrossRef max is 1000 but we don't need that many
            "select": "DOI,title,author,abstract,published-print,published-online,container-title,is-referenced-by-count,license,link,URL",
        }
        
        # Add year filter if specified
        if year_range:
            parts = year_range.split("-")
            if len(parts) == 2:
                if parts[0] and parts[1]:
                    params["filter"] = f"from-pub-date:{parts[0]},until-pub-date:{parts[1]}"
                elif parts[0]:
                    params["filter"] = f"from-pub-date:{parts[0]}"
        
        logger.info(f"CrossRef: searching for '{query[:50]}...' (limit: {limit})")
        data = self._request("GET", "/works", params=params)
        
        results = []
        items = data.get("message", {}).get("items", [])
        
        for item in items:
            paper = self._parse_work(item)
            if paper:
                results.append(paper)
        
        logger.info(f"CrossRef: found {len(results)} papers")
        return results[:limit]
    
    def get_paper_details(self, doi: str) -> Optional[Paper]:
        """Get paper details by DOI."""
        # Clean DOI
        doi = doi.replace("https://doi.org/", "").replace("DOI:", "")
        
        logger.info(f"CrossRef: fetching details for DOI {doi}")
        data = self._request("GET", f"/works/{doi}")
        
        message = data.get("message", {})
        if not message:
            return None
        
        return self._parse_work(message)
    
    def close(self):
        """Close HTTP client."""
        self._http.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
