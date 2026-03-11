# -*- coding: utf-8 -*-
"""
Unpaywall API — finds free PDF URLs for papers using their DOI.

Unpaywall indexes 50M+ free scholarly articles. Given a DOI,
it returns the best open-access PDF URL if one exists.

API: https://unpaywall.org/products/api (free, just needs an email)
"""

import logging
import os
from typing import Optional, List

import dotenv
import httpx

from src.literature.models import Paper


dotenv.load_dotenv()
logger = logging.getLogger(__name__)

UNPAYWALL_API = "https://api.unpaywall.org/v2"
# Unpaywall requires an email — this identifies us as a research tool
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL")


def resolve_pdf_url(doi: str, timeout: float = 10.0) -> Optional[str]:
    """
    Given a DOI, query Unpaywall for a free PDF URL.

    Returns:
        PDF URL string if found, None otherwise
    """
    if not doi:
        return None

    try:
        url = f"{UNPAYWALL_API}/{doi}"
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, params={"email": CONTACT_EMAIL})

        if response.status_code != 200:
            return None

        data = response.json()

        # Try best_oa_location first
        best_oa = data.get("best_oa_location") or {}
        pdf_url = best_oa.get("url_for_pdf")
        if pdf_url:
            return pdf_url

        # Try all OA locations
        for location in data.get("oa_locations", []):
            pdf_url = location.get("url_for_pdf")
            if pdf_url:
                return pdf_url

        return None

    except Exception as e:
        logger.debug(f"Unpaywall lookup failed for DOI {doi}: {e}")
        return None


def resolve_pdf_urls_batch(papers: List[Paper]) -> int:
    """
    Try to find free PDF URLs for papers that don't have one.
    Modifies papers in-place.

    Returns:
        Number of papers that got a new PDF URL
    """
    resolved = 0
    for paper in papers:
        if paper.pdf_url:
            continue  # Already has a URL
        if not paper.doi:
            continue  # Can't look up without DOI

        pdf_url = resolve_pdf_url(paper.doi)
        if pdf_url:
            paper.pdf_url = pdf_url
            paper.is_open_access = True
            resolved += 1
            logger.info(
                f"  📎 Unpaywall: found PDF for '{paper.title[:50]}...'"
            )

    return resolved


def try_arxiv_url(paper: Paper) -> bool:
    """
    If a paper has an arXiv ID, construct the PDF URL directly.
    Many Semantic Scholar papers have arXiv versions but the
    pdf_url field points to a paywalled publisher instead.

    Modifies paper in-place. Returns True if URL was set.
    """
    # Check paper_id for arXiv pattern
    paper_id = paper.paper_id or ""

    # ArXiv IDs look like: "2102.09139", "arxiv_2102.09139v3"
    arxiv_id = None

    if paper_id.startswith("arxiv_"):
        arxiv_id = paper_id.replace("arxiv_", "")
    elif paper_id.startswith("ARXIV:"):
        arxiv_id = paper_id.replace("ARXIV:", "")

    # Also check the URL field
    if not arxiv_id and paper.url:
        if "arxiv.org" in paper.url:
            # Extract ID from URL like https://arxiv.org/abs/2102.09139
            parts = paper.url.rstrip("/").split("/")
            if len(parts) >= 2:
                arxiv_id = parts[-1]

    if arxiv_id:
        # Remove version suffix for cleaner URL if needed
        paper.pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        paper.is_open_access = True
        return True

    return False