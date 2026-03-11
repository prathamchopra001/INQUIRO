"""
Semantic Scholar API client for Inquiro.

Handles paper search, details retrieval, and rate limiting.
Semantic Scholar is free (no API key required, but recommended).

API Docs: https://api.semanticscholar.org/api-docs/graph
"""

import time
import logging
import threading
from typing import Optional, List
from pathlib import Path

import httpx

from src.literature.models import Paper
from config.settings import settings

logger = logging.getLogger(__name__)


class SemanticScholarClient:
    """
    Client for the Semantic Scholar Academic Graph API.
    
    Features:
        - Paper search by keywords
        - Paper details by ID
        - Rate limiting (respects API limits)
        - Retry on transient errors
        - Converts API responses → Paper objects
    
    Usage:
        client = SemanticScholarClient()
        papers = client.search_papers("metabolomics neuroprotection")
        paper = client.get_paper_details("some-paper-id")
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    # Class-level lock shared across ALL instances and threads.
    # Ensures only one request fires at a time regardless of
    # how many SemanticScholarClient instances exist.
    _global_lock = threading.Lock()
    _global_last_request_time: float = 0.0
    
    # Fields we want from the API (saves bandwidth vs requesting everything)
    SEARCH_FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,"
    "authors,citationCount,influentialCitationCount,"
    "isOpenAccess,openAccessPdf,url"
)
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        requests_per_minute: int = 10,   # Very conservative: handles parallel threads
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize the Semantic Scholar client.
        
        Args:
            api_key: Optional API key (get one free at semanticscholar.org).
                     Without it, rate limits are lower (100 req/5min).
            requests_per_minute: Max requests per minute (for rate limiting).
            max_retries: How many times to retry on transient errors.
            timeout: HTTP request timeout in seconds.
        """
        self.api_key = api_key or settings.literature.semantic_scholar_api_key
        self.max_retries = max_retries
        self.timeout = timeout
        
        # --- Rate limiting ---
        self._min_interval = 60.0 / requests_per_minute  # seconds between requests
        
        # --- HTTP client ---
        headers = {
            "Accept": "application/json",
            "User-Agent": "Inquiro/1.0 (Academic Research Tool; mailto:chopra.pr@northeastern.edu)",
            }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        self._http = httpx.Client(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=self.timeout,
        )
    
    # =========================================================================
    # RATE LIMITING (boilerplate — done for you)
    # =========================================================================
    
    def _wait_for_rate_limit(self):
        """
        Sleep if needed to respect rate limits.

        Uses a class-level lock so ALL threads (across all instances)
        share the same rate limit counter. Prevents the burst-then-429
        pattern that happens when parallel threads each think they're alone.
        """
        with SemanticScholarClient._global_lock:
            elapsed = time.time() - SemanticScholarClient._global_last_request_time
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            SemanticScholarClient._global_last_request_time = time.time()
    
    # =========================================================================
    # HTTP REQUEST WITH RETRY (boilerplate — done for you)
    # =========================================================================
    
    def _request(self, method: str, url: str, **kwargs) -> dict:
        """
        Make an HTTP request with rate limiting and retry logic.
        
        Handles:
            - Rate limiting (waits between requests)
            - 429 Too Many Requests (backs off and retries)
            - 5xx Server Errors (retries)
            - Timeouts (retries)
        
        Returns the JSON response as a dict.
        Raises RuntimeError if all retries fail.
        """
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()
            
            try:
                response = self._http.request(method, url, **kwargs)
                
                if response.status_code == 200:
                    return response.json()
                
                if response.status_code == 429:
                    # Rate limited — back off exponentially
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited (429). Waiting {wait}s... (attempt {attempt})")
                    time.sleep(wait)
                    continue
                
                if response.status_code >= 500:
                    # Server error — retry
                    logger.warning(f"Server error ({response.status_code}). Retrying... (attempt {attempt})")
                    time.sleep(1)
                    continue
                
                # Client error (4xx, not 429) — don't retry
                logger.error(f"API error {response.status_code}: {response.text[:200]}")
                return {}
                
            except httpx.TimeoutException:
                logger.warning(f"Timeout. Retrying... (attempt {attempt})")
                time.sleep(1)
                continue
            except httpx.RequestError as e:
                logger.warning(f"Request error: {e}. Retrying... (attempt {attempt})")
                time.sleep(1)
                continue
        
        logger.error(f"All {self.max_retries} retries failed for {url}")
        return {}
    
    # =========================================================================
    # PARSING (boilerplate — done for you)
    # =========================================================================
    
    def _parse_paper(self, data: dict) -> Optional[Paper]:
        """
        Convert a single Semantic Scholar API result into a Paper object.
        """
        if not data or not data.get("paperId"):
            return None

        try:
            # Extract author names
            authors = []
            for author in data.get("authors", []):
                if isinstance(author, dict) and author.get("name"):
                    authors.append(author["name"])
                elif isinstance(author, str):
                    authors.append(author)

            # Extract DOI from externalIds
            external_ids = data.get("externalIds") or {}
            doi = external_ids.get("DOI")

            # Extract open access PDF URL
            pdf_url = None
            is_open_access = data.get("isOpenAccess", False)
            oa_info = data.get("openAccessPdf")
            if oa_info and isinstance(oa_info, dict):
                pdf_url = oa_info.get("url")

            return Paper(
                paper_id=data["paperId"],
                doi=doi,
                title=data.get("title", "Unknown Title"),
                abstract=data.get("abstract"),
                year=data.get("year"),
                venue=data.get("venue"),
                authors=authors,
                citation_count=data.get("citationCount", 0),
                influential_citation_count=data.get("influentialCitationCount", 0),
                url=data.get("url"),
                pdf_url=pdf_url,
                is_open_access=is_open_access,
            )
        except Exception as e:
            logger.warning(f"Failed to parse paper {data.get('paperId', '?')}: {e}")
            return None
    
    # =========================================================================
    # === YOUR CODE HERE === (the two methods you'll implement)
    # =========================================================================
    
    def search_papers(
        self,
        query: str,
        limit: int = None,
        year_range: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[Paper]:
        """
        Search for papers matching a query string.
        
        This is the main entry point for paper discovery.
        
        Args:
            query: Search keywords (e.g., "metabolomics neuroprotection hypothermia")
            limit: Max number of results. Defaults to settings.literature.max_papers_per_search
            year_range: Optional year filter (e.g., "2020-2024" or "2020-")
            fields_of_study: Optional list like ["Biology", "Medicine"]
        
        Returns:
            List of Paper objects, ordered by relevance.
        
        API endpoint: GET /paper/search
        Query params:
            - query: the search string
            - limit: max results (1-100)
            - fields: comma-separated field names (use self.SEARCH_FIELDS)
            - year: year range string (optional)
            - fieldsOfStudy: comma-separated (optional)
        
        Hints:
            1. Build a params dict with query, limit, and fields
            2. Add year and fieldsOfStudy only if provided
            3. Call self._request("GET", "/paper/search", params=params)
            4. The response has a "data" key containing a list of paper dicts
            5. Parse each one with self._parse_paper(item)
            6. Filter out None values (failed parses)
            7. Log how many papers were found
        """
        limit = limit or getattr(settings.literature, "max_papers_per_search", 10)
        
        # Validate query length - long task descriptions don't work well as search queries
        if len(query) > 100:
            logger.warning(
                f"Semantic Scholar query too long ({len(query)} chars). "
                f"Truncating. Query: '{query[:50]}...'"
            )
            query = query[:80]
        
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": self.SEARCH_FIELDS,
        }
        
        if year_range:
            # Format: "2020-2024" or "2020-"
            params["year"] = year_range
            
        if fields_of_study:
            # API expects comma-separated list: "Biology,Medicine"
            params["fieldsOfStudy"] = ",".join(fields_of_study)
            
        # 3. Execute request
        logger.info(f"Searching Semantic Scholar for: '{query}' (limit: {limit})")
        data = self._request("GET", "/paper/search", params=params)
        
        # 4. Parse results
        results = []
        paper_dicts = data.get("data", [])
        
        for item in paper_dicts:
            paper = self._parse_paper(item)
            if paper:
                results.append(paper)
                
        logger.info(f"Found {len(results)} valid papers for query: '{query}'")
        return results
    
    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """
        Get detailed information about a specific paper.
        
        Useful when you have a paper ID (from search results or citations)
        and need full details.
        
        Args:
            paper_id: Semantic Scholar paper ID, DOI, or other identifier.
                      Examples: "649def34f8be52c8b66281af98ae884c09aef38b"
                               "DOI:10.1038/s41586-023-06185-3"
                               "ARXIV:2511.02824"
        
        Returns:
            Paper object or None if not found.
        
        API endpoint: GET /paper/{paper_id}
        Query params:
            - fields: comma-separated field names (use self.SEARCH_FIELDS)
        
        Hints:
            1. Call self._request("GET", f"/paper/{paper_id}", params={"fields": self.SEARCH_FIELDS})
            2. Parse with self._parse_paper(data)
            3. Log success or failure
        """
        # TODO: Implement this method
        logger.info(f"Fetching details for paper ID: {paper_id}")
        params = {"fields": self.SEARCH_FIELDS}
        
        data = self._request("GET", f"/paper/{paper_id}", params=params)
        
        if not data:
            logger.warning(f"No data returned for paper ID: {paper_id}")
            return None
            
        # 2. Parse result
        paper = self._parse_paper(data)
        
        if paper:
            logger.info(f"Successfully retrieved details for: {paper.title}")
        else:
            logger.error(f"Failed to parse data for paper ID: {paper_id}")
            
        return paper
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def close(self):
        """Close the HTTP client."""
        self._http.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()