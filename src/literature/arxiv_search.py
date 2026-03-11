# -*- coding: utf-8 -*-
"""
ArXiv search client for INQUIRO literature search.

Uses the ArXiv public API (no key required).
Covers: physics, math, CS, biology, quantitative finance,
        statistics, electrical engineering, economics.

ArXiv API docs: https://arxiv.org/help/api
"""

import logging
import time
import re
from typing import Optional
import httpx

from src.literature.models import Paper

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
DEFAULT_MAX_RESULTS = 20
DEFAULT_TIMEOUT = 15


class ArXivClient:
    """
    Search ArXiv for scientific papers.

    Usage:
        client = ArXivClient()
        papers = client.search_papers("metabolomics group differences ANOVA")
    """

    def __init__(
        self,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.max_results = max_results
        self.timeout = timeout
        self._last_request_time: float = 0.0
        self._min_interval: float = 3.0  # ArXiv asks for 3s between requests

    def search_papers(self, query: str) -> list[Paper]:
        """
        Search ArXiv for papers matching the query.

        Args:
            query: Search string (plain text, not URL-encoded)

        Returns:
            List of Paper objects
        """
        # Validate query - warn if it looks like a task description rather than keywords
        if len(query) > 100:
            logger.warning(
                f"ArXiv query too long ({len(query)} chars). "
                f"Search APIs work best with short keyword queries. "
                f"Query preview: '{query[:60]}...'"
            )
            # Truncate to first 80 chars to avoid garbage results
            query = query[:80]
        
        # Check for signs this is a task description, not a search query
        bad_prefixes = ('conduct a', 'identify the', 'analyze the', 'investigate', 
                        'examine the', 'find papers', 'search for')
        if query.lower().startswith(bad_prefixes):
            logger.warning(
                f"ArXiv query looks like a task description, not search keywords: '{query[:50]}'"
            )
        
        self._rate_limit()

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(ARXIV_API_URL, params=params)
                response.raise_for_status()

            papers = self._parse_response(response.text)
            logger.info(
                f"ArXiv: found {len(papers)} papers for '{query[:50]}'"
            )
            return papers

        except httpx.HTTPStatusError as e:
            logger.warning(f"ArXiv HTTP error {e.response.status_code}: {query[:50]}")
            return []
        except httpx.TimeoutException:
            logger.warning(f"ArXiv timeout for query: {query[:50]}")
            return []
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return []

    def _parse_response(self, xml_text: str) -> list[Paper]:
        """Parse ArXiv Atom XML response into Paper objects."""
        papers = []

        # Extract entries using regex (avoids xml dependency issues)
        entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)

        for entry in entries:
            try:
                paper = self._parse_entry(entry)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.debug(f"ArXiv parse error for entry: {e}")
                continue

        return papers

    def _parse_entry(self, entry: str) -> Optional[Paper]:
        """Parse a single ArXiv entry into a Paper object."""

        def extract(tag: str, text: str) -> str:
            match = re.search(
                rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL
            )
            return match.group(1).strip() if match else ""

        arxiv_id_raw = extract("id", entry)
        title = extract("title", entry)
        abstract = extract("summary", entry)
        published = extract("published", entry)

        if not title or not arxiv_id_raw:
            return None

        # Extract clean ArXiv ID
        arxiv_id_match = re.search(r"arxiv\.org/abs/(.+?)$", arxiv_id_raw)
        arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else arxiv_id_raw

        # Build PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Extract year from published date
        year_match = re.search(r"(\d{4})", published)
        year = int(year_match.group(1)) if year_match else None

        # Extract authors
        author_names = re.findall(
            r"<name>(.*?)</name>", entry, re.DOTALL
        )
        # Paper model expects List[str] not List[dict]
        authors = [a.strip() for a in author_names[:5]]

        # Clean up title (remove newlines)
        title = re.sub(r"\s+", " ", title).strip()
        abstract = re.sub(r"\s+", " ", abstract).strip()

        return Paper(
            paper_id=f"arxiv_{arxiv_id.replace('/', '_')}",
            title=title,
            abstract=abstract[:1000],
            year=year,
            authors=authors,
            venue="arXiv",
            citation_count=0,
            influential_citation_count=0,
            is_open_access=True,
            open_access_pdf={"url": pdf_url},
            pdf_url=pdf_url,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            external_ids={"ArXiv": arxiv_id},
        )

    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()