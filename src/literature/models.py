"""
Literature-specific models for Inquiro.

These represent papers found during literature search,
separate from the World Model's Finding/Source models.

Paper = full paper metadata (what we download)
Source = lightweight citation (what goes into a Finding)
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class Paper(BaseModel):
    """
    A scientific paper found via search.
    
    This holds everything we know about a paper from
    Semantic Scholar (or other sources in the future).
    
    Lifecycle:
        1. Created from search results (title, abstract, authors)
        2. Enriched with PDF URL if open access
        3. Text extracted if PDF downloaded
        4. Chunks stored in RAG system
    """
    # --- Identity ---
    paper_id: str = Field(..., description="Semantic Scholar paper ID")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    title: str = Field(..., description="Paper title")
    
    # --- Content ---
    abstract: Optional[str] = Field(None, description="Paper abstract")
    year: Optional[int] = Field(None, description="Publication year")
    venue: Optional[str] = Field(None, description="Journal/conference name")
    
    # --- Authors ---
    authors: List[str] = Field(default_factory=list, description="Author names")
    
    # --- Metrics ---
    citation_count: int = Field(0, description="Number of citations")
    influential_citation_count: int = Field(0, description="Influential citations")
    
    # --- Access ---
    url: Optional[str] = Field(None, description="URL to paper page")
    pdf_url: Optional[str] = Field(None, description="Direct PDF download URL")
    is_open_access: bool = Field(False, description="Whether PDF is freely available")
    
    # --- Processing state (updated as we work with the paper) ---
    is_downloaded: bool = Field(False, description="PDF has been downloaded")
    is_processed: bool = Field(False, description="Text has been extracted and chunked")
    local_pdf_path: Optional[str] = Field(None, description="Local path to downloaded PDF")
    relevance_score: Optional[float] = Field(None, description="LLM-assigned relevance 0-1")
    
    # --- Timestamps ---
    found_at: datetime = Field(default_factory=datetime.now)
    
    def to_source(self) -> dict:
        """
        Convert this Paper into a Source dict for use in Findings.
        
        This bridges the gap between the literature system and the
        world model. When the Literature Agent creates a Finding,
        it uses this to generate the source reference.
        
        Returns a dict compatible with Source(**paper.to_source())
        """
        return {
            "type": "paper",
            "doi": self.doi,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "year": self.year,
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"{authors_str} ({self.year}) - {self.title}"
    
    model_config = {"use_enum_values": True}


class TextChunk(BaseModel):
    """
    A chunk of text extracted from a paper.
    
    Papers get split into overlapping chunks for the RAG system.
    Each chunk remembers where it came from (which paper, which section).
    """
    text: str = Field(..., description="The chunk text content")
    paper_id: str = Field(..., description="Which paper this came from")
    paper_title: str = Field(..., description="Paper title (for display)")
    doi: Optional[str] = Field(None, description="Paper DOI")
    authors: List[str] = Field(default_factory=list, description="Paper authors")
    
    # --- Position info ---
    chunk_index: int = Field(..., description="Position in the paper (0-based)")
    section: Optional[str] = Field(None, description="Section name if detected")
    page_number: Optional[int] = Field(None, description="Page number if known")
    
    # --- Metadata for ChromaDB ---
    def to_metadata(self) -> dict:
        """Convert to a flat dict for ChromaDB storage."""
        # Format authors as a string (ChromaDB metadata must be scalar)
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "authors": authors_str,
            "doi": self.doi or "",
            "chunk_index": self.chunk_index,
            "section": self.section or "unknown",
            "page_number": self.page_number or -1,
        }
# ```

# A couple things to notice here:

# **`Paper.to_source()`** is the bridge between systems. When the Literature Agent finds something interesting in a paper, it creates a `Finding` with `source=Source(**paper.to_source())`. This keeps the world model clean — it doesn't need to know about the full `Paper` object, just the citation info.

# **`TextChunk.to_metadata()`** is there because ChromaDB stores metadata as flat key-value dicts (no nested objects). This method handles that conversion so the RAG system doesn't have to worry about it.

# ---

# ## Step 1B: Semantic Scholar Client

# Now the fun part — the API client. Here's what you need to know about Semantic Scholar's API:
# ```
# SEMANTIC SCHOLAR API
# ════════════════════

# Base URL: https://api.semanticscholar.org/graph/v1

# Endpoints we care about:
#   GET /paper/search?query=...     → Search by keywords
#   GET /paper/{paper_id}           → Get full paper details
#   GET /paper/{paper_id}/citations → Get papers that cite this one

# Rate limits:
#   Without API key: 100 requests per 5 minutes
#   With API key:    1000 requests per 5 minutes (free to get!)

# Key fields we request:
#   paperId, doi, title, abstract, year, venue,
#   authors, citationCount, influentialCitationCount,
#   isOpenAccess, openAccessPdf, url