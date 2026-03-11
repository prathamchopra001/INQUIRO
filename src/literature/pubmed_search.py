# -*- coding: utf-8 -*-
"""
PubMed search client for INQUIRO literature search.

Uses NCBI E-utilities API (no key required, but rate-limited).
Covers: biomedical literature, clinical trials, life sciences.

NCBI E-utilities docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import logging
import time
import re
from typing import Optional
import httpx

from src.literature.models import Paper

logger = logging.getLogger(__name__)

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

DEFAULT_MAX_RESULTS = 20
DEFAULT_TIMEOUT = 15


class PubMedClient:
    """
    Search PubMed for biomedical papers.

    Uses a two-step process:
      1. esearch — get PubMed IDs matching the query
      2. esummary — fetch metadata for those IDs

    Usage:
        client = PubMedClient()
        papers = client.search_papers("metabolomics biomarker discovery")
    """

    def __init__(
        self,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout: int = DEFAULT_TIMEOUT,
        api_key: Optional[str] = None,
    ):
        self.max_results = max_results
        self.timeout = timeout
        self.api_key = api_key
        self._last_request_time: float = 0.0
        # NCBI allows 3 req/s without key, 10 req/s with key
        self._min_interval: float = 0.4 if api_key else 0.35

    def search_papers(self, query: str) -> list[Paper]:
        """
        Search PubMed for papers matching the query.

        Args:
            query: Search string

        Returns:
            List of Paper objects
        """
        # Validate query length - PubMed works best with focused queries
        if len(query) > 100:
            logger.warning(
                f"PubMed query too long ({len(query)} chars). "
                f"Truncating. Query: '{query[:50]}...'"
            )
            query = query[:80]
        
        try:
            # Step 1: Get PubMed IDs
            pmids = self._search_ids(query)
            if not pmids:
                return []

            # Step 2: Fetch metadata
            papers = self._fetch_summaries(pmids)
            logger.info(
                f"PubMed: found {len(papers)} papers for '{query[:50]}'"
            )
            return papers

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    def _search_ids(self, query: str) -> list[str]:
        """Step 1: Get PubMed IDs for a query."""
        self._rate_limit()

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self.max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(PUBMED_SEARCH_URL, params=params)
                response.raise_for_status()
                data = response.json()

            pmids = data.get("esearchresult", {}).get("idlist", [])
            return pmids

        except httpx.HTTPStatusError as e:
            logger.warning(f"PubMed search HTTP {e.response.status_code}: {query[:50]}")
            return []
        except httpx.TimeoutException:
            logger.warning(f"PubMed search timeout: {query[:50]}")
            return []

    def _fetch_summaries(self, pmids: list[str]) -> list[Paper]:
        """Step 2: Fetch paper metadata for a list of PMIDs."""
        if not pmids:
            return []

        self._rate_limit()

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(PUBMED_SUMMARY_URL, params=params)
                response.raise_for_status()
                data = response.json()

            papers = []
            result = data.get("result", {})

            for pmid in pmids:
                if pmid not in result:
                    continue
                paper = self._parse_summary(pmid, result[pmid])
                if paper:
                    papers.append(paper)

            return papers

        except httpx.HTTPStatusError as e:
            logger.warning(f"PubMed fetch HTTP {e.response.status_code}")
            return []
        except httpx.TimeoutException:
            logger.warning("PubMed fetch timeout")
            return []

    def _parse_summary(self, pmid: str, summary: dict) -> Optional[Paper]:
        """Parse a PubMed summary dict into a Paper object."""
        try:
            title = summary.get("title", "").strip()
            if not title:
                return None

            # Authors — Paper model expects List[str] not List[dict]
            author_list = summary.get("authors", [])
            authors = [
                a.get("name", "") if isinstance(a, dict) else str(a)
                for a in author_list[:5]
            ]

            # Year from pubdate
            pubdate = summary.get("pubdate", "")
            year_match = re.search(r"(\d{4})", pubdate)
            year = int(year_match.group(1)) if year_match else None

            # Journal/venue
            venue = summary.get("fulljournalname", "") or summary.get("source", "")

            # DOI
            articleids = summary.get("articleids", [])
            doi = next(
                (aid["value"] for aid in articleids if aid.get("idtype") == "doi"),
                None
            )

            # PMC ID (for open access PDF)
            pmc_id = next(
                (aid["value"] for aid in articleids if aid.get("idtype") == "pmc"),
                None
            )
            pdf_url = None
            if pmc_id:
                pmc_num = pmc_id.replace("PMC", "")
                pdf_url = f"https://europepmc.org/articles/pmc{pmc_num}?pdf=render"

            # Abstract — not in esummary, would need efetch
            # Use title as fallback context
            abstract = summary.get("title", "")

            return Paper(
                paper_id=f"pubmed_{pmid}",
                title=title,
                abstract=abstract[:500],
                year=year,
                authors=authors,
                venue=venue,
                citation_count=0,
                influential_citation_count=0,
                is_open_access=bool(pmc_id),
                open_access_pdf={"url": pdf_url} if pdf_url else None,
                pdf_url=pdf_url,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                external_ids={"PubMed": pmid, "DOI": doi} if doi else {"PubMed": pmid},
            )

        except Exception as e:
            logger.warning(f"PubMed parse error for PMID {pmid}: {e}")
            return None

    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()