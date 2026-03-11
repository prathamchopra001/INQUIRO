"""
OpenAlex API client for Inquiro.

OpenAlex is a free, open catalog of 240M+ scholarly works.
Better rate limits than Semantic Scholar, excellent OA detection.

API Docs: https://docs.openalex.org
"""

import time
import logging
from typing import Optional, List

import httpx

from src.literature.models import Paper
from config.settings import settings

logger = logging.getLogger(__name__)


class OpenAlexClient:
    """
    Client for the OpenAlex API.
    
    Drop-in alternative to SemanticScholarClient.
    Same interface: search_papers() and get_paper_details().
    
    Usage:
        client = OpenAlexClient(email="you@example.com")
        papers = client.search_papers("metabolomics neuroprotection")
    """
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(
        self,
        email: Optional[str] = None,
        requests_per_minute: int = 60,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self.email = email
        self.max_retries = max_retries
        self.timeout = timeout
        
        self._min_interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0
        
        headers = {"Accept": "application/json"}
        params = {}
        if self.email:
            params["mailto"] = self.email
        
        self._http = httpx.Client(
            base_url=self.BASE_URL,
            headers=headers,
            params=params,
            timeout=self.timeout,
        )
    
    def _wait_for_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _request(self, method: str, url: str, **kwargs) -> dict:
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self._http.request(method, url, **kwargs)
                
                if response.status_code == 200:
                    return response.json()
                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited (429). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if response.status_code >= 500:
                    logger.warning(f"Server error ({response.status_code}). Retrying...")
                    time.sleep(1)
                    continue
                    
                logger.error(f"API error {response.status_code}: {response.text[:200]}")
                return {}
                
            except (httpx.TimeoutException, httpx.RequestError) as e:
                logger.warning(f"Request error: {e}. Retrying... (attempt {attempt})")
                time.sleep(1)
                continue
        
        logger.error(f"All {self.max_retries} retries failed for {url}")
        return {}
    
    def _reconstruct_abstract(self, inverted_index: dict) -> Optional[str]:
        """
        OpenAlex stores abstracts as inverted indexes for licensing reasons.
        Reconstructs the original text.
        """
        if not inverted_index:
            return None
        
        words = []
        for word, positions in inverted_index.items():
            for pos in positions:
                words.append((pos, word))
        words.sort(key=lambda x: x[0])
        
        return " ".join(word for _, word in words)
    
    def _parse_work(self, data: dict) -> Optional[Paper]:
        """Convert an OpenAlex work object into our Paper model."""
        if not data or not data.get("id"):
            return None
        
        try:
            openalex_id = data["id"].replace("https://openalex.org/", "")
            
            doi = data.get("doi")
            if doi and doi.startswith("https://doi.org/"):
                doi = doi[len("https://doi.org/"):]
            
            authors = []
            for authorship in data.get("authorships", []):
                author = authorship.get("author", {})
                name = author.get("display_name")
                if name:
                    authors.append(name)
            
            pdf_url = None
            is_open_access = data.get("open_access", {}).get("is_oa", False)
            
            best_oa = data.get("best_oa_location") or {}
            pdf_url = best_oa.get("pdf_url")
            
            if not pdf_url:
                primary = data.get("primary_location") or {}
                pdf_url = primary.get("pdf_url")
            
            abstract = self._reconstruct_abstract(
                data.get("abstract_inverted_index")
            )
            
            primary_location = data.get("primary_location") or {}
            source = primary_location.get("source") or {}
            venue = source.get("display_name")
            
            return Paper(
                paper_id=openalex_id,
                doi=doi,
                title=data.get("display_name", data.get("title", "Unknown Title")),
                abstract=abstract,
                year=data.get("publication_year"),
                venue=venue,
                authors=authors,
                citation_count=data.get("cited_by_count", 0),
                influential_citation_count=0,
                url=f"https://openalex.org/{openalex_id}",
                pdf_url=pdf_url,
                is_open_access=is_open_access,
            )
        except Exception as e:
            logger.warning(f"Failed to parse work: {e}")
            return None
    
    def search_papers(
        self,
        query: str,
        limit: int = None,
        year_range: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[Paper]:
        """Search for papers. Same interface as SemanticScholarClient."""
        # Validate query length - long queries return poor results
        if len(query) > 100:
            logger.warning(
                f"OpenAlex query too long ({len(query)} chars). "
                f"Truncating for better results. Query: '{query[:50]}...'"
            )
            query = query[:80]
        
        limit = limit or settings.literature.max_papers_per_search
        
        params = {
            "search": query,
            "per_page": min(limit, 50),
        }
        
        if year_range:
            parts = year_range.split("-")
            if len(parts) == 2 and parts[0] and parts[1]:
                params["filter"] = f"publication_year:{parts[0]}-{parts[1]}"
            elif len(parts) == 2 and parts[0]:
                params["filter"] = f"publication_year:>{int(parts[0])-1}"
        
        logger.info(f"Searching OpenAlex for: '{query}' (limit: {limit})")
        data = self._request("GET", "/works", params=params)
        
        results = []
        for item in data.get("results", []):
            paper = self._parse_work(item)
            if paper:
                results.append(paper)
        
        logger.info(f"Found {len(results)} papers for query: '{query}'")
        return results[:limit]
    
    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get details about a specific paper by OpenAlex ID or DOI."""
        if paper_id.startswith("10.") or paper_id.startswith("DOI:"):
            doi = paper_id.replace("DOI:", "")
            url = f"/works/https://doi.org/{doi}"
        else:
            url = f"/works/{paper_id}"
        
        logger.info(f"Fetching details for: {paper_id}")
        data = self._request("GET", url)
        
        if not data:
            return None
        
        return self._parse_work(data)
    
    def close(self):
        self._http.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()