# -*- coding: utf-8 -*-
"""
CORE API client for Inquiro.

CORE is the world's largest aggregator of open access research papers.
200M+ papers with full text available. Free API with registration.

API Docs: https://core.ac.uk/documentation/api
Get free API key: https://core.ac.uk/services/api
"""

import time
import logging
import threading
from typing import Optional, List

import httpx

from src.literature.models import Paper
from config.settings import settings

logger = logging.getLogger(__name__)


class COREClient:
    """
    Client for the CORE API (v3).
    
    CORE specializes in open access content with full text availability.
    Requires a free API key from https://core.ac.uk/services/api
    
    Usage:
        client = COREClient(api_key="your-api-key")
        papers = client.search_papers("diabetes machine learning")
    """
    
    BASE_URL = "https://api.core.ac.uk/v3"
    
    # Class-level rate limiting
    _global_lock = threading.Lock()
    _global_last_request_time: float = 0.0
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        requests_per_minute: int = 30,  # CORE free tier: ~10 req/sec, we use conservative 30 RPM
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize CORE client.
        
        Args:
            api_key: CORE API key (free at https://core.ac.uk/services/api).
                     Without it, most endpoints won't work.
            requests_per_minute: Rate limit
            max_retries: Retry count
            timeout: Request timeout
        """
        self.api_key = api_key or getattr(settings.literature, 'core_api_key', None)
        self.max_retries = max_retries
        self.timeout = timeout
        
        self._min_interval = 60.0 / requests_per_minute
        
        if not self.api_key:
            logger.warning(
                "CORE: No API key configured. Get a free key at https://core.ac.uk/services/api "
                "and set CORE_API_KEY in your .env file."
            )
        
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }
        
        self._http = httpx.Client(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=self.timeout,
        )
    
    def _wait_for_rate_limit(self):
        """Thread-safe rate limiting."""
        with COREClient._global_lock:
            elapsed = time.time() - COREClient._global_last_request_time
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                logger.debug(f"CORE rate limit: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            COREClient._global_last_request_time = time.time()
    
    def _request(self, method: str, url: str, **kwargs) -> dict:
        """Make HTTP request with retry logic."""
        if not self.api_key:
            logger.debug("CORE: skipping request (no API key)")
            return {}
        
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()
            
            try:
                response = self._http.request(method, url, **kwargs)
                
                if response.status_code == 200:
                    return response.json()
                
                if response.status_code == 401:
                    logger.error("CORE: Invalid API key (401 Unauthorized)")
                    return {}
                
                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"CORE rate limited (429). Waiting {wait}s... (attempt {attempt})")
                    time.sleep(wait)
                    continue
                
                if response.status_code >= 500:
                    logger.warning(f"CORE server error ({response.status_code}). Retrying...")
                    time.sleep(1)
                    continue
                
                logger.error(f"CORE API error {response.status_code}: {response.text[:200]}")
                return {}
                
            except (httpx.TimeoutException, httpx.RequestError) as e:
                logger.warning(f"CORE request error: {e}. Retrying... (attempt {attempt})")
                time.sleep(1)
                continue
        
        logger.error(f"CORE: all {self.max_retries} retries failed for {url}")
        return {}
    
    def _parse_work(self, data: dict) -> Optional[Paper]:
        """Convert CORE work to Paper model."""
        if not data:
            return None
        
        try:
            core_id = str(data.get("id", ""))
            
            # Extract identifiers
            doi = None
            identifiers = data.get("identifiers", [])
            if isinstance(identifiers, list):
                for ident in identifiers:
                    if isinstance(ident, str) and ident.startswith("10."):
                        doi = ident
                        break
            
            # Title
            title = data.get("title", "Unknown Title")
            
            # Authors
            authors = []
            for author in data.get("authors", []):
                if isinstance(author, dict):
                    name = author.get("name", "")
                    if name:
                        authors.append(name)
                elif isinstance(author, str):
                    authors.append(author)
            
            # Year
            year = data.get("yearPublished")
            
            # Abstract
            abstract = data.get("abstract", "")
            
            # Venue / Journal
            venue = None
            journals = data.get("journals", [])
            if journals and isinstance(journals[0], dict):
                venue = journals[0].get("title")
            
            # Citation count (CORE doesn't always have this)
            citation_count = data.get("citationCount", 0) or 0
            
            # URLs
            url = data.get("sourceFulltextUrls", [None])[0] if data.get("sourceFulltextUrls") else None
            if not url:
                url = data.get("downloadUrl") or f"https://core.ac.uk/works/{core_id}"
            
            # PDF URL - CORE specializes in open access
            pdf_url = data.get("downloadUrl")
            
            # Open access (CORE is all OA)
            is_open_access = True
            
            return Paper(
                paper_id=f"core_{core_id}",
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
            logger.warning(f"CORE: failed to parse work: {e}")
            return None
    
    def search_papers(
        self,
        query: str,
        limit: int = None,
        year_range: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[Paper]:
        """
        Search CORE for papers.
        
        Args:
            query: Search keywords
            limit: Max results
            year_range: Optional "2020-2024" filter
            fields_of_study: Not directly supported
        
        Returns:
            List of Paper objects
        """
        if not self.api_key:
            logger.debug("CORE: skipping search (no API key)")
            return []
        
        # Validate query length
        if len(query) > 100:
            logger.warning(f"CORE query too long ({len(query)} chars). Truncating.")
            query = query[:80]
        
        limit = limit or settings.literature.max_papers_per_search
        
        # CORE uses POST for search with JSON body
        search_body = {
            "q": query,
            "limit": min(limit, 100),
        }
        
        # Add year filter
        if year_range:
            parts = year_range.split("-")
            if len(parts) == 2 and parts[0]:
                search_body["yearFrom"] = int(parts[0])
                if parts[1]:
                    search_body["yearTo"] = int(parts[1])
        
        logger.info(f"CORE: searching for '{query[:50]}...' (limit: {limit})")
        data = self._request("POST", "/search/works", json=search_body)
        
        results = []
        items = data.get("results", [])
        
        for item in items:
            paper = self._parse_work(item)
            if paper:
                results.append(paper)
        
        logger.info(f"CORE: found {len(results)} papers")
        return results[:limit]
    
    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get paper details by CORE ID."""
        # Clean ID
        paper_id = paper_id.replace("core_", "")
        
        logger.info(f"CORE: fetching details for {paper_id}")
        data = self._request("GET", f"/works/{paper_id}")
        
        if not data:
            return None
        
        return self._parse_work(data)
    
    def close(self):
        """Close HTTP client."""
        self._http.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
