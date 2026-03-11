# -*- coding: utf-8 -*-
"""
GROBID Client for structured PDF parsing.

GROBID (GeneRation Of BIbliographic Data) is a machine learning library
for extracting structured information from scholarly documents.

It extracts:
- Structured sections (abstract, intro, methods, results, discussion)
- References with citation positions
- Figures and tables
- Author/affiliation information

Requires GROBID service running (Docker or native):
    docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1

Falls back to PyMuPDF if GROBID is unavailable.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import httpx

logger = logging.getLogger(__name__)

# GROBID TEI namespace
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass
class StructuredSection:
    """A structured section extracted from a paper."""
    section_type: str  # abstract, introduction, methods, results, discussion, conclusion, references
    title: Optional[str] = None
    text: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    figures: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)


@dataclass
class GROBIDResult:
    """Result from GROBID parsing."""
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    sections: List[StructuredSection] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    raw_tei: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


class GROBIDClient:
    """
    Client for the GROBID PDF extraction service.
    
    Usage:
        client = GROBIDClient()
        if client.is_available():
            result = client.process_pdf("paper.pdf")
            for section in result.sections:
                print(f"{section.section_type}: {section.text[:100]}...")
    """
    
    DEFAULT_URL = "http://localhost:8070"
    
    def __init__(
        self,
        base_url: str = None,
        timeout: float = 120.0,
        consolidate_citations: bool = True,
    ):
        """
        Args:
            base_url: GROBID service URL (default: http://localhost:8070)
            timeout: Request timeout in seconds
            consolidate_citations: Whether to consolidate citations with CrossRef
        """
        self.base_url = (base_url or self.DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self.consolidate_citations = consolidate_citations
        self._available = None
    
    def is_available(self) -> bool:
        """Check if GROBID service is running."""
        if self._available is not None:
            return self._available
        
        try:
            response = httpx.get(
                f"{self.base_url}/api/isalive",
                timeout=5.0
            )
            self._available = response.status_code == 200
        except Exception:
            self._available = False
        
        return self._available
    
    def process_pdf(self, pdf_path: str) -> GROBIDResult:
        """
        Process a PDF and extract structured content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            GROBIDResult with extracted sections and metadata
        """
        if not self.is_available():
            return GROBIDResult(
                success=False,
                error="GROBID service not available"
            )
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return GROBIDResult(
                success=False,
                error=f"PDF not found: {pdf_path}"
            )
        
        try:
            # Call GROBID's fulltext endpoint
            with open(pdf_path, "rb") as f:
                response = httpx.post(
                    f"{self.base_url}/api/processFulltextDocument",
                    files={"input": (pdf_path.name, f, "application/pdf")},
                    data={
                        "consolidateCitations": "1" if self.consolidate_citations else "0",
                        "includeRawCitations": "1",
                        "segmentSentences": "1",
                    },
                    timeout=self.timeout,
                )
            
            if response.status_code != 200:
                return GROBIDResult(
                    success=False,
                    error=f"GROBID returned status {response.status_code}"
                )
            
            # Parse TEI XML response
            return self._parse_tei(response.text)
            
        except httpx.TimeoutException:
            return GROBIDResult(
                success=False,
                error="GROBID request timed out"
            )
        except Exception as e:
            logger.error(f"GROBID processing error: {e}")
            return GROBIDResult(
                success=False,
                error=str(e)
            )

    
    def _parse_tei(self, tei_xml: str) -> GROBIDResult:
        """Parse GROBID's TEI XML response into structured sections."""
        result = GROBIDResult(raw_tei=tei_xml, success=True)
        
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as e:
            result.success = False
            result.error = f"TEI XML parse error: {e}"
            return result
        
        # Extract title
        title_elem = root.find(".//tei:titleStmt/tei:title", TEI_NS)
        if title_elem is not None and title_elem.text:
            result.title = title_elem.text.strip()
        
        # Extract authors
        for author in root.findall(".//tei:author/tei:persName", TEI_NS):
            forename = author.find("tei:forename", TEI_NS)
            surname = author.find("tei:surname", TEI_NS)
            name_parts = []
            if forename is not None and forename.text:
                name_parts.append(forename.text)
            if surname is not None and surname.text:
                name_parts.append(surname.text)
            if name_parts:
                result.authors.append(" ".join(name_parts))
        
        # Extract abstract
        abstract_elem = root.find(".//tei:abstract", TEI_NS)
        if abstract_elem is not None:
            abstract_text = self._get_all_text(abstract_elem)
            if abstract_text:
                result.abstract = abstract_text
                result.sections.append(StructuredSection(
                    section_type="abstract",
                    title="Abstract",
                    text=abstract_text
                ))
        
        # Extract body sections
        body = root.find(".//tei:body", TEI_NS)
        if body is not None:
            self._parse_body_sections(body, result)
        
        # Extract references
        back = root.find(".//tei:back", TEI_NS)
        if back is not None:
            for ref in back.findall(".//tei:biblStruct", TEI_NS):
                ref_data = self._parse_reference(ref)
                if ref_data:
                    result.references.append(ref_data)
        
        return result
    
    def _parse_body_sections(self, body: ET.Element, result: GROBIDResult) -> None:
        """Parse body sections from TEI."""
        # Section type detection keywords
        section_keywords = {
            "introduction": ["introduction", "background", "motivation"],
            "methods": ["method", "methodology", "materials", "experimental", "approach", "procedure"],
            "results": ["result", "finding", "experiment", "evaluation", "analysis"],
            "discussion": ["discussion", "interpretation", "implication"],
            "conclusion": ["conclusion", "summary", "future work", "limitations"],
            "related_work": ["related work", "literature review", "previous work", "state of the art"],
        }
        
        for div in body.findall(".//tei:div", TEI_NS):
            # Get section head/title
            head = div.find("tei:head", TEI_NS)
            section_title = head.text.strip() if head is not None and head.text else None
            
            # Detect section type from title
            section_type = "body"  # default
            if section_title:
                title_lower = section_title.lower()
                for stype, keywords in section_keywords.items():
                    if any(kw in title_lower for kw in keywords):
                        section_type = stype
                        break
            
            # Extract section text
            section_text = self._get_all_text(div, skip_head=True)
            
            if section_text:
                result.sections.append(StructuredSection(
                    section_type=section_type,
                    title=section_title,
                    text=section_text
                ))

    
    def _get_all_text(self, element: ET.Element, skip_head: bool = False) -> str:
        """Extract all text from an element, handling nested elements."""
        texts = []
        
        for elem in element.iter():
            # Skip head element if requested
            if skip_head and elem.tag.endswith("}head"):
                continue
            
            if elem.text:
                texts.append(elem.text.strip())
            if elem.tail:
                texts.append(elem.tail.strip())
        
        return " ".join(t for t in texts if t)
    
    def _parse_reference(self, ref: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse a single reference from TEI."""
        try:
            ref_data = {}
            
            # Title
            title = ref.find(".//tei:title[@level='a']", TEI_NS)
            if title is None:
                title = ref.find(".//tei:title[@level='m']", TEI_NS)
            if title is not None and title.text:
                ref_data["title"] = title.text.strip()
            
            # Authors
            authors = []
            for author in ref.findall(".//tei:author/tei:persName", TEI_NS):
                forename = author.find("tei:forename", TEI_NS)
                surname = author.find("tei:surname", TEI_NS)
                name_parts = []
                if forename is not None and forename.text:
                    name_parts.append(forename.text)
                if surname is not None and surname.text:
                    name_parts.append(surname.text)
                if name_parts:
                    authors.append(" ".join(name_parts))
            if authors:
                ref_data["authors"] = authors
            
            # Year
            date = ref.find(".//tei:date[@type='published']", TEI_NS)
            if date is not None:
                when = date.get("when", "")
                if when:
                    ref_data["year"] = when[:4]
            
            # DOI
            doi = ref.find(".//tei:idno[@type='DOI']", TEI_NS)
            if doi is not None and doi.text:
                ref_data["doi"] = doi.text.strip()
            
            return ref_data if ref_data else None
            
        except Exception:
            return None


def get_grobid_client(
    base_url: str = None,
    timeout: float = 120.0,
) -> Optional[GROBIDClient]:
    """
    Get a GROBID client if the service is available.
    
    Returns:
        GROBIDClient instance if available, None otherwise
    """
    client = GROBIDClient(base_url=base_url, timeout=timeout)
    if client.is_available():
        logger.info("✅ GROBID service available for structured PDF parsing")
        return client
    else:
        logger.info("ℹ️  GROBID service not available, using PyMuPDF fallback")
        return None
