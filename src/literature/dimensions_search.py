# -*- coding: utf-8 -*-
"""
Dimensions API client for Inquiro.

Dimensions is a comprehensive research database with 130M+ publications,
plus grants, patents, clinical trials, and policy documents.

API Docs: https://docs.dimensions.ai/dsl/
Free API: https://www.dimensions.ai/scientometric-research/ (apply for free access)
"""

import time
import logging
import threading
from typing import Optional, List

import httpx

from src.literature.models import Paper
from config.settings import settings

logger = logging.getLogger(__name__)


class DimensionsClient:
    """
    Client for the Dimensions Analytics API.
    
    Dimensions requires authentication - either:
    1. API key (paid subscription)
    2. Username/password (free for scientometric research)
    
    Apply for free access: https://www.dimensions.ai/scientometric-research/
    
    Usage:
        client = DimensionsClient(api_key="your-key")
        # OR
        client = DimensionsClient(username="user", password="pass")
        papers = client.search_papers("diabetes readmission prediction")
    """
    
    BASE_URL = "https://app.dimensions.ai/api"
    AUTH_URL = "https://app.dimensions.ai/api/auth.json"  # Note: .json extension required
    
    # Class-level rate limiting
    _global_lock = threading.Lock()
    _global_last_request_time: float = 0.0
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        requests_per_minute: int = 30,  # Dimensions free tier limit
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize Dimensions client.
        
        Args:
            api_key: Dimensions API key (paid tier)
            username: Dimensions username (free tier)
            password: Dimensions password (free tier)
            requests_per_minute: Rate limit
            max_retries: Retry count
            timeout: Request timeout
        """
        self.api_key = api_key or getattr(settings.literature, 'dimensions_api_key', None)
        self.username = username or getattr(settings.literature, 'dimensions_username', None)
        self.password = password or getattr(settings.literature, 'dimensions_password', None)
        self.max_retries = max_retries
        self.timeout = timeout
        
        self._min_interval = 60.0 / requests_per_minute
        self._auth_token = None
        self._token_expiry = 0
        self._api_access_warned = False  # Track if we've warned about 403
        
        if not self.api_key and not (self.username and self.password):
            logger.warning(
                "Dimensions: No credentials configured. "
                "Apply for free access at https://www.dimensions.ai/scientometric-research/ "
                "and set DIMENSIONS_API_KEY or DIMENSIONS_USERNAME/DIMENSIONS_PASSWORD in .env"
            )
        
        self._http = httpx.Client(timeout=self.timeout)
    
    def _authenticate(self) -> bool:
        """Get authentication token (for username/password auth)."""
        if self.api_key:
            return True  # API key doesn't need token
        
        if not (self.username and self.password):
            return False
        
        # Check if existing token is still valid
        if self._auth_token and time.time() < self._token_expiry:
            return True
        
        try:
            response = self._http.post(
                self.AUTH_URL,
                json={"username": self.username, "password": self.password},
                headers={"Content-Type": "application/json"},
            )
            
            if response.status_code == 200:
                data = response.json()
                self._auth_token = data.get("token")
                # Token valid for ~2 hours, we refresh after 1.5 hours
                self._token_expiry = time.time() + 5400
                logger.info("Dimensions: authenticated successfully")
                return True
            else:
                logger.error(f"Dimensions auth failed: {response.status_code} {response.text[:200]}")
                return False
                
        except Exception as e:
            logger.error(f"Dimensions auth error: {e}")
            return False
    
    def _wait_for_rate_limit(self):
        """Thread-safe rate limiting."""
        with DimensionsClient._global_lock:
            elapsed = time.time() - DimensionsClient._global_last_request_time
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                logger.debug(f"Dimensions rate limit: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            DimensionsClient._global_last_request_time = time.time()
    
    def _get_headers(self) -> dict:
        """Get request headers with authentication."""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        elif self._auth_token:
            return {"Authorization": f"JWT {self._auth_token}"}
        return {}
    
    def _request(self, dsl_query: str) -> dict:
        """Execute a DSL query against Dimensions API."""
        # Ensure we're authenticated
        if not self.api_key and not self._authenticate():
            logger.debug("Dimensions: skipping request (not authenticated)")
            return {}
        
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()
            
            try:
                response = self._http.post(
                    f"{self.BASE_URL}/dsl.json",
                    data=dsl_query,
                    headers={
                        **self._get_headers(),
                        "Content-Type": "text/plain",
                    },
                )
                
                if response.status_code == 200:
                    return response.json()
                
                if response.status_code == 401:
                    # Token expired, try to re-authenticate
                    self._auth_token = None
                    if self._authenticate():
                        continue
                    logger.error("Dimensions: authentication failed")
                    return {}
                
                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Dimensions rate limited (429). Waiting {wait}s... (attempt {attempt})")
                    time.sleep(wait)
                    continue
                
                if response.status_code >= 500:
                    logger.warning(f"Dimensions server error ({response.status_code}). Retrying...")
                    time.sleep(1)
                    continue
                
                if response.status_code == 403:
                    if not self._api_access_warned:
                        logger.warning(
                            "Dimensions: Your account doesn't have API access. "
                            "Apply at https://www.dimensions.ai/scientometric-research/"
                        )
                        self._api_access_warned = True
                else:
                    logger.error(f"Dimensions API error {response.status_code}: {response.text[:200]}")
                return {}
                
            except (httpx.TimeoutException, httpx.RequestError) as e:
                logger.warning(f"Dimensions request error: {e}. Retrying... (attempt {attempt})")
                time.sleep(1)
                continue
        
        logger.error(f"Dimensions: all {self.max_retries} retries failed")
        return {}
    
    def _parse_publication(self, data: dict) -> Optional[Paper]:
        """Convert Dimensions publication to Paper model."""
        if not data:
            return None
        
        try:
            dim_id = data.get("id", "")
            
            # DOI
            doi = data.get("doi", "")
            
            # Title
            title = data.get("title", "Unknown Title")
            
            # Authors
            authors = []
            for author in data.get("authors", []):
                if isinstance(author, dict):
                    first = author.get("first_name", "")
                    last = author.get("last_name", "")
                    if first and last:
                        authors.append(f"{first} {last}")
                    elif last:
                        authors.append(last)
            
            # Year
            year = data.get("year")
            
            # Abstract
            abstract = data.get("abstract", "")
            
            # Venue / Journal
            venue = data.get("journal", {}).get("title") if isinstance(data.get("journal"), dict) else None
            
            # Citation count
            citation_count = data.get("times_cited", 0) or 0
            
            # URLs
            url = f"https://doi.org/{doi}" if doi else f"https://app.dimensions.ai/details/publication/{dim_id}"
            
            # Open access
            oa_status = data.get("open_access", {})
            is_open_access = bool(oa_status) if isinstance(oa_status, dict) else False
            
            # PDF URL (Dimensions doesn't always have this)
            pdf_url = None
            if is_open_access and isinstance(oa_status, dict):
                pdf_url = oa_status.get("url")
            
            return Paper(
                paper_id=f"dim_{dim_id}",
                doi=doi if doi else None,
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
            logger.warning(f"Dimensions: failed to parse publication: {e}")
            return None
    
    def search_papers(
        self,
        query: str,
        limit: int = None,
        year_range: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[Paper]:
        """
        Search Dimensions for papers using DSL.
        
        Args:
            query: Search keywords
            limit: Max results
            year_range: Optional "2020-2024" filter
            fields_of_study: Optional field filters (e.g., ["Medical and Health Sciences"])
        
        Returns:
            List of Paper objects
        """
        # Check if we have credentials
        if not self.api_key and not (self.username and self.password):
            logger.debug("Dimensions: skipping search (no credentials)")
            return []
        
        # Validate query length
        if len(query) > 100:
            logger.warning(f"Dimensions query too long ({len(query)} chars). Truncating.")
            query = query[:80]
        
        limit = limit or settings.literature.max_papers_per_search
        
        # Build DSL query
        # Escape quotes in query
        safe_query = query.replace('"', '\\"')
        
        # Start building the query
        dsl_parts = [f'search publications in full_data for "{safe_query}"']
        
        # Add filters
        filters = []
        if year_range:
            parts = year_range.split("-")
            if len(parts) == 2 and parts[0]:
                filters.append(f"year >= {parts[0]}")
                if parts[1]:
                    filters.append(f"year <= {parts[1]}")
        
        if fields_of_study:
            # Dimensions uses category_for for field filtering
            for_ids = '["' + '","'.join(fields_of_study) + '"]'
            filters.append(f"category_for.name in {for_ids}")
        
        if filters:
            dsl_parts.append("where " + " and ".join(filters))
        
        # Add return fields and limit
        dsl_parts.append(
            f"return publications[id, doi, title, abstract, authors, year, "
            f"journal, times_cited, open_access] limit {min(limit, 100)}"
        )
        
        dsl_query = " ".join(dsl_parts)
        logger.info(f"Dimensions: searching for '{query[:50]}...' (limit: {limit})")
        logger.debug(f"Dimensions DSL: {dsl_query}")
        
        data = self._request(dsl_query)
        
        results = []
        publications = data.get("publications", [])
        
        for item in publications:
            paper = self._parse_publication(item)
            if paper:
                results.append(paper)
        
        logger.info(f"Dimensions: found {len(results)} papers")
        return results[:limit]
    
    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get paper details by Dimensions ID or DOI."""
        # Clean ID
        paper_id = paper_id.replace("dim_", "")
        
        # Build DSL query
        if paper_id.startswith("10."):
            # It's a DOI
            dsl_query = f'search publications where doi = "{paper_id}" return publications[all]'
        else:
            # It's a Dimensions ID
            dsl_query = f'search publications where id = "{paper_id}" return publications[all]'
        
        logger.info(f"Dimensions: fetching details for {paper_id}")
        data = self._request(dsl_query)
        
        publications = data.get("publications", [])
        if not publications:
            return None
        
        return self._parse_publication(publications[0])
    
    def close(self):
        """Close HTTP client."""
        self._http.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
