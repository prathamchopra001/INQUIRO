"""
PDF download and text extraction for Inquiro.

Downloads open-access PDFs, extracts text using PyMuPDF,
and chunks the text for the RAG system.

Supports GROBID for structured section extraction when available.
Falls back to PyMuPDF if GROBID is unavailable.
"""

import logging
import hashlib
from pathlib import Path
from typing import Optional, List

import httpx
import fitz  # PyMuPDF — imported as "fitz" for historical reasons

from src.literature.models import Paper, TextChunk
from src.literature.grobid_client import GROBIDClient, get_grobid_client, GROBIDResult
from config.settings import settings

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Downloads and processes scientific PDFs.
    
    Features:
        - Downloads PDFs from URLs (with caching)
        - Extracts text page by page using PyMuPDF
        - Chunks text with configurable size and overlap
        - Produces TextChunk objects ready for the RAG system
    
    Usage:
        parser = PDFParser()
        chunks = parser.process_paper(paper)
        # chunks is a List[TextChunk] ready for ChromaDB
    """
    
    def __init__(
        self,
        cache_dir: str = "./data/pdf_cache",
        chunk_size: int = None,
        chunk_overlap: int = None,
        timeout: float = 60.0,
        use_grobid: bool = None,
    ):
        """
        Args:
            cache_dir: Where to save downloaded PDFs.
            chunk_size: Target chunk size in characters (~4 chars per token).
                        Defaults to settings.rag.chunk_size * 4.
            chunk_overlap: Overlap between chunks in characters.
                          Defaults to settings.rag.chunk_overlap * 4.
            timeout: HTTP timeout for PDF downloads.
            use_grobid: Whether to use GROBID for structured extraction.
                       None = auto-detect based on settings and availability.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert token counts to character counts (rough: 1 token ≈ 4 chars)
        self.chunk_size = chunk_size or (settings.rag.chunk_size * 4)
        self.chunk_overlap = chunk_overlap or (settings.rag.chunk_overlap * 4)
        self.timeout = timeout

        # Content-hash registry: md5 → local path
        # Prevents re-downloading/re-chunking the same PDF bytes
        # even when they arrive under different paper_ids
        self._content_hashes: dict[str, str] = {}
        
        # Initialize GROBID client if enabled
        self._grobid: Optional[GROBIDClient] = None
        if use_grobid is None:
            use_grobid = settings.grobid.enabled
        
        if use_grobid:
            self._grobid = get_grobid_client(
                base_url=settings.grobid.base_url,
                timeout=settings.grobid.timeout,
            )
    
    # =========================================================================
    # DOWNLOADING (boilerplate — done for you)
    # =========================================================================
    
    def _get_cache_path(self, paper: Paper) -> Path:
        """Generate a cache filename for a paper."""
        # Use paper_id hash to avoid filesystem issues with long IDs
        name_hash = hashlib.md5(paper.paper_id.encode()).hexdigest()[:12]
        return self.cache_dir / f"{name_hash}.pdf"
    
    def download_pdf(self, paper: Paper) -> Optional[str]:
        """
        Download a paper's PDF to the local cache.

        Returns the local file path if successful, None if failed.
        Skips download if already cached.
        """
        if not paper.pdf_url:
            logger.warning(f"No PDF URL for: {paper.title[:60]}")
            return None

        cache_path = self._get_cache_path(paper)

        # Check cache first
        if cache_path.exists() and cache_path.stat().st_size > 0:
            logger.debug(f"Cache hit: {cache_path}")
            return str(cache_path)

        # Normalize the URL (fix common patterns)
        pdf_url = self._normalize_pdf_url(paper.pdf_url)

        # Download
        try:
            logger.info(f"Downloading PDF: {paper.title[:60]}...")
            logger.debug(f"  URL: {pdf_url}")

            # Key fix: Add a realistic User-Agent header.
            # Many publishers reject requests from "python-httpx/0.27"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }

            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers=headers
            ) as client:
                response = client.get(pdf_url)
                response.raise_for_status()

                # Primary check: magic bytes (more reliable than content-type)
                is_pdf_bytes = response.content[:5] == b"%PDF-"
                content_type = response.headers.get("content-type", "")
                is_pdf_content_type = "pdf" in content_type

                if not is_pdf_bytes and not is_pdf_content_type:
                    logger.warning(
                        f"Not a PDF (content-type: {content_type}, "
                        f"magic bytes: {response.content[:5]})"
                    )
                    return None

                # Check content hash before saving
                content_md5 = hashlib.md5(response.content).hexdigest()
                if content_md5 in self._content_hashes:
                    existing = self._content_hashes[content_md5]
                    logger.info(
                        f"Duplicate PDF detected (same content as {Path(existing).name})"
                        f" — skipping: {paper.title[:50]}"
                    )
                    return None  # Signal caller to skip chunking

                cache_path.write_bytes(response.content)
                self._content_hashes[content_md5] = str(cache_path)
                logger.info(f"Downloaded: {cache_path} ({len(response.content)} bytes)")
                return str(cache_path)

        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error downloading PDF: {e.response.status_code}")
            return None
        except Exception as e:
            logger.warning(f"Failed to download PDF: {e}")
            return None

    def _normalize_pdf_url(self, url: str) -> str:
        """
        Fix common URL patterns that don't point directly to PDFs.

        Many sources give us abstract/landing page URLs instead of
        direct PDF links. This method converts known patterns.
        """
        # ArXiv: /abs/XXXX → /pdf/XXXX.pdf
        if "arxiv.org/abs/" in url:
            url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/")
            if not url.endswith(".pdf"):
                url += ".pdf"

        # ACM: add &format=pdf if missing
        if "dl.acm.org" in url and "format=pdf" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}format=pdf"

        return url
    
    # =========================================================================
    # === YOUR CODE HERE === (two methods to implement)
    # =========================================================================
    
    def extract_text(self, pdf_path: str) -> List[dict]:
        """
        Extract text from a PDF file, page by page using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file on disk.
        
        Returns:
            A list of dicts, one per page:
            [
                {"page": 1, "text": "Introduction\nThis paper explores..."},
                {"page": 2, "text": "Methods\nWe used a dataset of..."},
                ...
            ]
        """
        pages = []
        doc = None
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()
                
                # Skip pages with very little text (cover pages, etc.)
                if len(text) < 50:
                    continue
                    
                pages.append({
                    "page": page_num + 1, 
                    "text": text
                })
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []
        finally:
            if doc:
                doc.close()
                
        return pages
    
    def extract_structured(self, pdf_path: str) -> Optional[GROBIDResult]:
        """
        Extract structured sections from a PDF using GROBID.
        
        Returns GROBIDResult with sections, or None if GROBID unavailable/failed.
        """
        if self._grobid is None:
            return None
        
        result = self._grobid.process_pdf(pdf_path)
        if not result.success:
            logger.debug(f"GROBID extraction failed: {result.error}")
            return None
        
        return result
    
    def chunk_text(self, pages: List[dict], paper: Paper) -> List[TextChunk]:
        """
        Split extracted text into overlapping chunks for the RAG system.
        
        This is where text goes from "full pages" to "bite-sized pieces"
        that fit in an LLM's context window and can be embedded for search.
        
        Args:
            pages: Output from extract_text() — list of {"page": int, "text": str}
            paper: The Paper object (for metadata in each chunk)
        
        Returns:
            List of TextChunk objects ready for ChromaDB.
        
        Algorithm (sliding window):
            1. Concatenate all page texts into one big string
               (keep track of which page each character came from — 
                or simplify by just using the page of the chunk start)
            2. Slide a window of size chunk_size across the text
            3. Each window step moves forward by (chunk_size - chunk_overlap)
            4. For each window, create a TextChunk with:
               - text: the chunk content
               - paper_id, paper_title, doi from the Paper
               - chunk_index: 0, 1, 2, ...
               - page_number: which page this chunk starts on
               
        Why overlap?
            If a key sentence spans two chunks, the overlap ensures it
            appears fully in at least one chunk. Think of it like shingles
            on a roof — each one overlaps the previous.
        
        Hints:
            1. First combine all pages into one string, but build a helper 
               structure to track page boundaries:
               
               full_text = ""
               page_breaks = []  # list of (char_position, page_number)
               for p in pages:
                   page_breaks.append((len(full_text), p["page"]))
                   full_text += p["text"] + "\n"
            
            2. Calculate step size: step = self.chunk_size - self.chunk_overlap
            
            3. Slide the window:
               chunks = []
               for i in range(0, len(full_text), step):
                   chunk_text = full_text[i : i + self.chunk_size].strip()
                   if len(chunk_text) < 50:  # skip tiny trailing chunks
                       continue
                   
                   # Figure out which page this chunk starts on
                   page_num = 1  # default
                   for pos, pn in page_breaks:
                       if pos <= i:
                           page_num = pn
                   
                   chunks.append(TextChunk(...))
               
            4. Return the list of TextChunks
        """
        # TODO: Implement this method
        # 1. Combine all pages and track boundaries
        full_text = ""
        page_breaks = []  # list of (char_start_pos, page_number)
        
        for p in pages:
            page_breaks.append((len(full_text), p["page"]))
            full_text += p["text"] + "\n\n" # Double newline helps keep sections distinct

        # 2. Slide a window across the text
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        
        # Ensure we don't get an infinite loop if overlap >= size
        if step <= 0:
            step = self.chunk_size // 2
            logger.warning(f"chunk_overlap >= chunk_size. Adjusting step to {step}")

        for i in range(0, len(full_text), step):
            chunk_content = full_text[i : i + self.chunk_size].strip()
            
            # Skip tiny trailing chunks
            if len(chunk_content) < 50:
                continue
            
            # 3. Determine the page number for this chunk
            # We find the last page break that starts before or at the current index
            current_page = 1
            for pos, pn in page_breaks:
                if pos <= i:
                    current_page = pn
                else:
                    break # page_breaks is ordered, so we can stop once we pass 'i'

            # 4. Create the TextChunk object
            chunk = TextChunk(
                text=chunk_content,
                paper_id=paper.paper_id,
                paper_title=paper.title,
                doi=paper.doi,
                authors=paper.authors,  # Include authors!
                chunk_index=len(chunks),
                page_number=current_page,
                section=None # Section detection would require a more complex regex parser
            )
            chunks.append(chunk)
            
        return chunks
    
    def chunk_structured(self, grobid_result: GROBIDResult, paper: Paper) -> List[TextChunk]:
        """
        Create section-aware chunks from GROBID structured extraction.
        
        This produces MUCH better chunks than the sliding window approach:
        - Each chunk belongs to a semantic section (intro, methods, results, etc.)
        - Chunks don't cut across section boundaries
        - Section metadata is preserved for better retrieval
        
        Args:
            grobid_result: Structured extraction from GROBID
            paper: The Paper object for metadata
            
        Returns:
            List of TextChunk objects with section information
        """
        chunks = []
        chunk_index = 0
        
        for section in grobid_result.sections:
            section_text = section.text.strip()
            if not section_text or len(section_text) < 50:
                continue
            
            # If section is small enough, keep it as one chunk
            if len(section_text) <= self.chunk_size:
                chunk = TextChunk(
                    text=section_text,
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    doi=paper.doi,
                    authors=paper.authors,
                    chunk_index=chunk_index,
                    page_number=section.page_start or 1,
                    section=section.section_type,
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Large section: split with overlap but stay within section
                step = self.chunk_size - self.chunk_overlap
                if step <= 0:
                    step = self.chunk_size // 2
                
                for i in range(0, len(section_text), step):
                    chunk_content = section_text[i : i + self.chunk_size].strip()
                    if len(chunk_content) < 50:
                        continue
                    
                    chunk = TextChunk(
                        text=chunk_content,
                        paper_id=paper.paper_id,
                        paper_title=paper.title,
                        doi=paper.doi,
                        authors=paper.authors,
                        chunk_index=chunk_index,
                        page_number=section.page_start or 1,
                        section=section.section_type,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        
        # Log section breakdown
        section_counts = {}
        for c in chunks:
            section_counts[c.section] = section_counts.get(c.section, 0) + 1
        logger.debug(f"Structured chunks by section: {section_counts}")
        
        return chunks
    
    # =========================================================================
    # SEED PAPER SUPPORT
    # =========================================================================

    def process_local_pdf(self, pdf_path: str, paper: Paper) -> List[TextChunk]:
        """
        Process a PDF that's already on disk (seed papers pipeline).

        Uses GROBID for structured extraction when available,
        falls back to PyMuPDF otherwise.

        Args:
            pdf_path: Absolute path to the local PDF file.
            paper:    A Paper object with metadata (can be minimal —
                      paper_id + title are enough).

        Returns:
            List[TextChunk] ready for RAGSystem.add_chunks().
            Empty list if extraction fails.
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            logger.warning(f"Seed PDF not found: {pdf_path}")
            return []

        # Mark as "downloaded" since it's already local
        paper.is_downloaded = True
        paper.local_pdf_path = str(pdf_file)

        # Try GROBID for structured extraction
        if self._grobid is not None:
            grobid_result = self.extract_structured(str(pdf_file))
            if grobid_result and grobid_result.sections:
                logger.info(f"📑 GROBID extracted {len(grobid_result.sections)} sections from seed: {paper.title[:50]}...")
                chunks = self.chunk_structured(grobid_result, paper)
                if chunks:
                    paper.is_processed = True
                    logger.info(f"Created {len(chunks)} structured chunks from seed: {paper.title[:50]}...")
                    return chunks

        # Fallback to PyMuPDF extraction
        pages = self.extract_text(str(pdf_file))
        if not pages:
            logger.warning(f"No text extracted from seed paper: {paper.title[:60]}")
            return []

        logger.info(f"Extracted {len(pages)} pages from seed paper: {paper.title[:60]}")

        # Chunk with sliding window
        chunks = self.chunk_text(pages, paper)

        paper.is_processed = True
        logger.info(f"Created {len(chunks)} chunks from seed paper: {paper.title[:60]}")

        return chunks

    # =========================================================================
    # ORCHESTRATOR (boilerplate — done for you)
    # =========================================================================
    
    def process_paper(self, paper: Paper) -> List[TextChunk]:
        """
        Full pipeline: download → extract → chunk.
        
        Uses GROBID for structured section extraction when available,
        falls back to PyMuPDF otherwise.
        
        This is the main entry point. Give it a Paper, get back TextChunks.
        Updates the Paper object's state flags as it goes.
        
        Returns empty list if any step fails.
        """
        # Step 1: Download
        pdf_path = self.download_pdf(paper)
        if not pdf_path:
            return []
        
        paper.is_downloaded = True
        paper.local_pdf_path = pdf_path
        
        # Step 2: Try GROBID for structured extraction
        if self._grobid is not None:
            grobid_result = self.extract_structured(pdf_path)
            if grobid_result and grobid_result.sections:
                logger.info(f"📑 GROBID extracted {len(grobid_result.sections)} sections from: {paper.title[:50]}...")
                chunks = self.chunk_structured(grobid_result, paper)
                if chunks:
                    paper.is_processed = True
                    logger.info(f"Created {len(chunks)} structured chunks from: {paper.title[:50]}...")
                    return chunks
                else:
                    logger.debug(f"GROBID produced no chunks, falling back to PyMuPDF")
        
        # Step 3: Fallback to PyMuPDF extraction
        pages = self.extract_text(pdf_path)
        if not pages:
            logger.warning(f"No text extracted from: {paper.title[:60]}")
            return []
        
        logger.info(f"Extracted {len(pages)} pages from: {paper.title[:60]}")
        
        # Step 4: Chunk with sliding window
        chunks = self.chunk_text(pages, paper)
        
        paper.is_processed = True
        logger.info(f"Created {len(chunks)} chunks from: {paper.title[:60]}")
        
        return chunks