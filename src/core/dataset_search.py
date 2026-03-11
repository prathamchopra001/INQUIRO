# -*- coding: utf-8 -*-
"""
Public Dataset Searcher — finds relevant datasets from multiple sources.

Tier 1 of the no-dataset strategy:
  1. Detect research domain (economics, genomics, climate, etc.)
  2. Route to domain-specific sources (FRED, World Bank, NCBI GEO, etc.)
  3. Also search general sources (HuggingFace, Kaggle, OpenML)
  4. LLM scores each candidate for relevance
  5. Download the best match if above threshold

Falls back to synthetic generation (Tier 2) if no suitable dataset is found.
"""

import json
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import httpx
import pandas as pd

from src.utils.llm_client import LLMClient
from src.core.domain_routing import (
    DomainDetector, DomainDatasetSearcher, ResearchDomain, get_domain_sources
)
from config.prompts.dataset_search import (
    DATASET_QUERY_PROMPT,
    DATASET_RELEVANCE_PROMPT,
)

logger = logging.getLogger(__name__)

HUGGINGFACE_API = "https://huggingface.co/api/datasets"
MIN_RELEVANCE_SCORE = 0.6  # Dataset must score above this to be used


@dataclass
class DatasetCandidate:
    """A dataset found during search."""
    name: str
    description: str
    tags: List[str]
    downloads: int
    size: str
    relevance_score: float = 0.0
    relevance_reason: str = ""
    download_url: str = ""


@dataclass
class DatasetSearchResult:
    """Result of the dataset search process."""
    success: bool
    path: str                          # Path to downloaded CSV
    source: str                        # "huggingface", "kaggle", etc.
    dataset_name: str                  # Name of the dataset used
    description: str                   # What the dataset contains
    relevance_score: float             # How relevant the LLM judged it
    candidates_found: int              # Total candidates evaluated
    queries_used: List[str] = field(default_factory=list)
    
    
class DatasetSearcher:
    """
    Searches public dataset repositories for data relevant to a research objective.
    
    Domain-Aware Routing:
    - Detects research domain from objective (economics, genomics, climate, etc.)
    - Routes to specialized sources (FRED, World Bank, NCBI GEO, etc.)
    - Falls back to general sources (HuggingFace, Kaggle, OpenML)
    
    General Sources: HuggingFace, Kaggle, OpenML, Harvard Dataverse
    Domain Sources: FRED, World Bank, Eurostat, NCBI GEO, UniProt, NOAA, WHO
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self._http = httpx.Client(timeout=30.0)
        
        # Domain-aware routing
        self.domain_detector = DomainDetector(llm_client)
        self.domain_searcher = DomainDatasetSearcher(self._http)

    def search(
        self,
        objective: str,
        output_dir: str,
        num_queries: int = 4,
        max_candidates: int = 50,
        seed_context=""
    ) -> DatasetSearchResult:
        """
        Search for a public dataset matching the research objective.

        Args:
            objective:      Research question to find data for
            output_dir:     Where to save the downloaded dataset
            num_queries:    How many search queries to generate
            max_candidates: Max datasets to evaluate

        Returns:
            DatasetSearchResult with path to CSV if found
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("🔍 Searching for public datasets...")
        
        # Step 0: Detect research domain
        domain, domain_confidence = self.domain_detector.detect(objective)
        domain_sources = get_domain_sources(domain)
        if domain != ResearchDomain.GENERAL:
            logger.info(f"  📊 Domain: {domain.value} ({domain_confidence:.0%} confidence)")
            logger.info(f"     Available specialized sources: {', '.join(domain_sources) if domain_sources else 'none'}")

        # Step 1: Generate search queries
        queries = self._generate_queries(objective, num_queries)
        if not queries:
            logger.warning("  Failed to generate search queries")
            return self._empty_result(queries)

        logger.info(f"  Generated {len(queries)} search queries: {queries}")
        
        # Step 1.5: Search domain-specific sources FIRST (higher priority)
        domain_candidates = []
        if domain != ResearchDomain.GENERAL:
            logger.info(f"  🎯 Searching domain-specific sources for {domain.value}...")
            domain_results = self.domain_searcher.search(domain, queries, max_results=20)
            
            for dr in domain_results:
                domain_candidates.append(DatasetCandidate(
                    name=dr["name"],
                    description=dr["description"],
                    tags=dr.get("tags", []),
                    downloads=0,  # Domain sources don't always have download counts
                    size="varies",
                    download_url=json.dumps(dr.get("download_info", {})),  # Store as JSON
                ))
            
            if domain_candidates:
                logger.info(f"  ✅ Found {len(domain_candidates)} domain-specific datasets")

        # Step 2: Search all GENERAL sources in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        candidates = []
        sources = [
            ("HuggingFace", self._search_huggingface),
            ("OpenML", self._search_openml),
            ("Harvard Dataverse", self._search_dataverse),
            ("Kaggle", self._search_kaggle),
            ("Awesome Public Datasets", self._search_awesome_datasets),
            ("Web (DuckDuckGo)", self._search_web),
        ]

        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            future_to_source = {
                executor.submit(search_fn, queries, max_candidates): source_name
                for source_name, search_fn in sources
            }

            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    results = future.result()
                    if results:
                        logger.info(f"  {source_name}: found {len(results)} candidates")
                        candidates.extend(results)
                    else:
                        logger.info(f"  {source_name}: no results")
                except Exception as e:
                    logger.warning(f"  {source_name}: search failed — {e}")

        # Deduplicate by name
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.name not in seen:
                seen.add(c.name)
                unique_candidates.append(c)
        candidates = unique_candidates[:max_candidates]

        if not candidates:
            logger.info("  No dataset candidates found on any source")
            return self._empty_result(queries)

        # Step 3: Score candidates for relevance
        scored = self._score_candidates(candidates, objective)
        scored.sort(key=lambda c: c.relevance_score, reverse=True)

        # Log top candidates
        for i, c in enumerate(scored[:5]):
            logger.info(
                f"  #{i+1} {c.name} — score={c.relevance_score:.2f} "
                f"({c.relevance_reason[:60]})"
            )

        # Step 4: Check if best candidate is good enough
        if not scored:
            logger.warning("  No candidates were scored — this shouldn't happen")
            return self._empty_result(queries, len(candidates))
            
        best = scored[0]
        logger.info(f"  Best candidate: {best.name[:80]} (score={best.relevance_score:.2f})")
        
        if best.relevance_score < MIN_RELEVANCE_SCORE:
            logger.info(
                f"  Best dataset '{best.name[:50]}' scored {best.relevance_score:.2f} "
                f"(below threshold {MIN_RELEVANCE_SCORE})"
            )
            return self._empty_result(queries, len(candidates))

        # Step 5: Download the best dataset
        logger.info(f"  ⬇️ Downloading '{best.name}' (score={best.relevance_score:.2f})...")
        logger.info(f"     Download URL/ref: {best.download_url}")
        
        try:
            csv_path = self._download_dataset(best, output_path)
        except Exception as e:
            logger.error(f"  ❌ Download exception: {e}")
            csv_path = None

        if csv_path and csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                logger.info(
                    f"  ✅ Dataset ready: {csv_path} "
                    f"({len(df)} rows × {len(df.columns)} cols)"
                )
            except Exception:
                pass

            return DatasetSearchResult(
                success=True,
                path=str(csv_path),
                source="huggingface",
                dataset_name=best.name,
                description=best.description[:200],
                relevance_score=best.relevance_score,
                candidates_found=len(candidates),
                queries_used=queries,
            )

        logger.warning(f"  Download failed for '{best.name}'")
        return self._empty_result(queries, len(candidates))
    def _generate_queries(self, objective: str, num_queries: int,seed_context="") -> List[str]:
        """Use the LLM to generate dataset search queries."""
        prompt = DATASET_QUERY_PROMPT.format(
            objective=objective,
            num_queries=num_queries,
            seed_context=seed_context,
        )
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="dataset_search",
                system="You are a data scientist searching for datasets.",
            )
            text = response.content.strip()

            # Parse JSON array
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                queries = json.loads(text[start:end])
                if isinstance(queries, list):
                    return [str(q) for q in queries[:num_queries]]

            logger.warning("Could not parse query JSON from LLM")
            return []
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return []
    def _search_huggingface(
        self, queries: List[str], max_total: int
    ) -> List[DatasetCandidate]:
        """Search HuggingFace Datasets API."""
        seen_names = set()
        candidates = []

        for query in queries:
            try:
                response = self._http.get(
                    HUGGINGFACE_API,
                    params={
                        "search": query,
                        "limit": 10,
                        "sort": "downloads",
                        "direction": "-1",
                    },
                )
                if response.status_code != 200:
                    logger.warning(
                        f"  HuggingFace API returned {response.status_code} "
                        f"for query '{query}'"
                    )
                    continue

                results = response.json()
                for item in results:
                    name = item.get("id", "")
                    if name in seen_names:
                        continue
                    seen_names.add(name)

                    candidates.append(DatasetCandidate(
                        name=name,
                        description=item.get("description", "")[:500] or item.get("id", ""),
                        tags=item.get("tags", []),
                        downloads=item.get("downloads", 0),
                        size=str(item.get("cardData", {}).get("dataset_size", "unknown")),
                        download_url="",  # Will be constructed at download time
                    ))

            except Exception as e:
                logger.warning(f"  HuggingFace search failed for '{query}': {e}")
                continue

            if len(candidates) >= max_total:
                break

        return candidates[:max_total]
    
    def _search_openml(
        self, queries: List[str], max_total: int
    ) -> List[DatasetCandidate]:
        """Search OpenML — free, no auth, clean ML datasets."""
        seen_names = set()
        candidates = []

        for query in queries:
            try:
                response = self._http.get(
                    "https://www.openml.org/api/v1/json/data/list",
                    params={"status": "active", "limit": 10, "output_format": "json"},
                    headers={"Accept": "application/json"},
                )
                if response.status_code != 200:
                    continue

                data = response.json()
                datasets = data.get("data", {}).get("dataset", [])
                if not isinstance(datasets, list):
                    continue

                query_lower = query.lower()
                for item in datasets:
                    name = item.get("name", "")
                    if name in seen_names:
                        continue

                    # Basic keyword filter since OpenML list endpoint 
                    # doesn't support full-text search well
                    desc = (name + " " + item.get("description", "")).lower()
                    if not any(w in desc for w in query_lower.split()):
                        continue

                    seen_names.add(name)
                    candidates.append(DatasetCandidate(
                        name=f"openml/{name}",
                        description=item.get("description", "")[:500] or name,
                        tags=[item.get("format", ""), f"rows:{item.get('NumberOfInstances', '?')}"],
                        downloads=int(item.get("NumberOfDownloads", 0)),
                        size=str(item.get("NumberOfInstances", "unknown")),
                        download_url=f"https://www.openml.org/data/download/{item.get('file_id', '')}/{name}.csv",
                    ))

            except Exception as e:
                logger.debug(f"  OpenML search error for '{query}': {e}")
                continue

            if len(candidates) >= max_total:
                break

        return candidates[:max_total]

    def _search_dataverse(
        self, queries: List[str], max_total: int
    ) -> List[DatasetCandidate]:
        """Search Harvard Dataverse — free, no auth, academic quality."""
        seen_names = set()
        candidates = []

        for query in queries:
            try:
                response = self._http.get(
                    "https://dataverse.harvard.edu/api/search",
                    params={
                        "q": query,
                        "type": "dataset",
                        "per_page": 10,
                        "sort": "score",
                        "order": "desc",
                    },
                )
                if response.status_code != 200:
                    continue

                data = response.json()
                items = data.get("data", {}).get("items", [])

                for item in items:
                    name = item.get("name", "")
                    if name in seen_names:
                        continue
                    seen_names.add(name)

                    # Build download URL from the global_id (DOI)
                    global_id = item.get("global_id", "")

                    candidates.append(DatasetCandidate(
                        name=f"dataverse/{name[:60]}",
                        description=item.get("description", "")[:500] or name,
                        tags=[s.strip() for s in item.get("subjects", [])],
                        downloads=int(item.get("download_count", 0)),
                        size=str(item.get("file_count", "unknown")) + " files",
                        download_url=global_id,
                    ))

            except Exception as e:
                logger.debug(f"  Dataverse search error for '{query}': {e}")
                continue

            if len(candidates) >= max_total:
                break

        return candidates[:max_total]

    def _search_kaggle(
        self, queries: List[str], max_total: int
    ) -> List[DatasetCandidate]:
        """Search Kaggle — needs kaggle.json API key in ~/.kaggle/."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            logger.debug("  Kaggle: `kaggle` library not installed — skipping")
            return []

        try:
            api = KaggleApi()
            api.authenticate()
        except Exception as e:
            logger.debug(f"  Kaggle: authentication failed — {e}")
            return []

        seen_names = set()
        candidates = []

        for query in queries:
            try:
                results = api.dataset_list(search=query, sort_by="hottest", page=1)
                for item in results:
                    # Extract the proper reference (owner/dataset-name format)
                    # str(item) returns JSON which breaks downloads
                    ref = getattr(item, "ref", None) or str(item)
                    if ref in seen_names:
                        continue
                    seen_names.add(ref)

                    candidates.append(DatasetCandidate(
                        name=f"kaggle/{ref}",
                        description=getattr(item, "title", ref),
                        tags=[str(t) for t in getattr(item, "tags", [])],
                        downloads=getattr(item, "downloadCount", 0),
                        size=str(getattr(item, "totalBytes", "unknown")),
                        download_url=ref,  # Must be owner/dataset-name format for API
                    ))

            except Exception as e:
                logger.debug(f"  Kaggle search error for '{query}': {e}")
                continue

            if len(candidates) >= max_total:
                break

        return candidates[:max_total]

    def _search_awesome_datasets(
        self, queries: List[str], max_total: int
    ) -> List[DatasetCandidate]:
        """
        Search the Awesome Public Datasets GitHub repo.
        Fetches the curated markdown list, parses categories and links,
        then matches against our queries.
        """
        # Cache the parsed list to avoid re-fetching
        if not hasattr(self, "_awesome_cache"):
            self._awesome_cache = self._fetch_awesome_list()

        if not self._awesome_cache:
            return []

        candidates = []
        seen = set()

        for query in queries:
            query_words = set(query.lower().split())
            for entry in self._awesome_cache:
                if entry["name"] in seen:
                    continue

                # Match: any query word in name, category, or description
                text = (entry["name"] + " " + entry["category"] + " " + entry.get("description", "")).lower()
                if any(w in text for w in query_words):
                    seen.add(entry["name"])
                    candidates.append(DatasetCandidate(
                        name=f"awesome/{entry['name'][:60]}",
                        description=f"[{entry['category']}] {entry.get('description', entry['name'])}",
                        tags=[entry["category"]],
                        downloads=0,
                        size="unknown",
                        download_url=entry.get("url", ""),
                    ))

            if len(candidates) >= max_total:
                break

        return candidates[:max_total]

    def _fetch_awesome_list(self) -> List[dict]:
        """Fetch and parse the Awesome Public Datasets README from GitHub."""
        url = "https://raw.githubusercontent.com/awesomedata/apd-core/master/README.md"
        try:
            response = self._http.get(url)
            if response.status_code != 200:
                logger.debug(f"  Awesome Datasets: HTTP {response.status_code}")
                return []

            entries = []
            current_category = "General"

            for line in response.text.split("\n"):
                line = line.strip()

                # Category headers: ## Economics, ## Climate, etc.
                if line.startswith("## "):
                    current_category = line[3:].strip()
                    continue

                # Dataset entries: - [Name](url) - description
                if line.startswith("- [") and "](" in line:
                    try:
                        name_start = line.index("[") + 1
                        name_end = line.index("]")
                        url_start = line.index("(") + 1
                        url_end = line.index(")")
                        name = line[name_start:name_end]
                        link = line[url_start:url_end]
                        desc = line[url_end + 1:].strip(" -–—")

                        entries.append({
                            "name": name,
                            "url": link,
                            "category": current_category,
                            "description": desc or name,
                        })
                    except (ValueError, IndexError):
                        continue

            logger.info(f"  Awesome Datasets: parsed {len(entries)} entries")
            return entries

        except Exception as e:
            logger.debug(f"  Awesome Datasets fetch failed: {e}")
            return []

    def _search_web(
        self, queries: List[str], max_total: int
    ) -> List[DatasetCandidate]:
        """
        Search the web for datasets using DuckDuckGo.
        Appends 'dataset CSV download' to each query for better results.
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.debug("  DuckDuckGo: `duckduckgo-search` library not installed — skipping")
            return []

        candidates = []
        seen_urls = set()

        for query in queries[:2]:  # Limit to 2 queries to avoid rate limits
            try:
                search_query = f"{query} dataset CSV download"
                with DDGS() as ddgs:
                    results = list(ddgs.text(search_query, max_results=5))

                for item in results:
                    url = item.get("href", "")
                    if url in seen_urls or not url:
                        continue
                    seen_urls.add(url)

                    # Only keep results that look like dataset pages
                    title = item.get("title", "")
                    body = item.get("body", "")
                    text_lower = (title + " " + body + " " + url).lower()

                    if any(w in text_lower for w in ["dataset", "data set", ".csv", ".parquet", "download"]):
                        candidates.append(DatasetCandidate(
                            name=f"web/{title[:50]}",
                            description=body[:300],
                            tags=["web-search"],
                            downloads=0,
                            size="unknown",
                            download_url=url,
                        ))

            except Exception as e:
                logger.debug(f"  DuckDuckGo search error for '{query}': {e}")
                continue

            if len(candidates) >= max_total:
                break

        return candidates[:max_total]
    
    def _score_candidates(
        self, candidates: List[DatasetCandidate], objective: str
    ) -> List[DatasetCandidate]:
        """Use the LLM to score each candidate's relevance."""
        for candidate in candidates:
            prompt = DATASET_RELEVANCE_PROMPT.format(
                objective=objective[:500],
                dataset_name=candidate.name,
                dataset_description=candidate.description[:300],
                dataset_tags=", ".join(candidate.tags[:10]),
                dataset_downloads=candidate.downloads,
                dataset_size=candidate.size,
            )
            try:
                response = self.llm.complete_for_role(
                    prompt=prompt,
                    role="scoring",
                    system="You are a data scientist. Respond only with JSON.",
                )
                text = response.content.strip()

                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(text[start:end])
                    candidate.relevance_score = float(result.get("score", 0.0))
                    candidate.relevance_reason = result.get("reason", "")
                else:
                    candidate.relevance_score = 0.0

            except Exception as e:
                logger.debug(f"  Scoring failed for {candidate.name}: {e}")
                candidate.relevance_score = 0.0

        return candidates
    
    def _download_dataset(
        self, candidate: DatasetCandidate, output_path: Path
    ) -> Optional[Path]:
        """
        Download a dataset. Routes to the appropriate method based on source.
        """
        csv_path = output_path / "public_dataset.csv"

        # Domain-specific sources (higher priority)
        if candidate.name.startswith("worldbank/"):
            return self._download_worldbank(candidate, csv_path)
        elif candidate.name.startswith("fred/"):
            return self._download_fred(candidate, csv_path)
        elif candidate.name.startswith("eurostat/"):
            return self._download_eurostat(candidate, csv_path)
        elif candidate.name.startswith("geo/"):
            return self._download_ncbi_geo(candidate, csv_path)
        elif candidate.name.startswith("who/"):
            return self._download_who(candidate, csv_path)
        
        # General sources
        if candidate.name.startswith("kaggle/"):
            return self._download_kaggle(candidate, csv_path)
        elif candidate.name.startswith("openml/"):
            return self._download_direct_csv(candidate.download_url, csv_path)
        elif candidate.name.startswith("dataverse/"):
            return self._download_dataverse(candidate, csv_path)
        elif candidate.name.startswith("awesome/") or candidate.name.startswith("web/"):
            return self._download_direct_csv(candidate.download_url, csv_path)
        else:
            # HuggingFace — use existing logic
            return self._download_huggingface(candidate, csv_path)

    def _download_huggingface(
        self, candidate: DatasetCandidate, csv_path: Path
    ) -> Optional[Path]:
        """Download from HuggingFace using datasets library or direct HTTP."""
        output_path = csv_path.parent

        # Method 1: datasets library
        try:
            from datasets import load_dataset
            ds = load_dataset(candidate.name, split="train", trust_remote_code=False)
            df = ds.to_pandas()
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
                logger.info(f"  Sampled 10,000 rows from {len(ds)} total")
            df.to_csv(csv_path, index=False)
            return csv_path
        except ImportError:
            logger.info("  `datasets` library not installed — trying direct download")
        except Exception as e:
            logger.warning(f"  datasets library failed: {e} — trying direct download")

        # Method 2: Direct HTTP
        try:
            base_url = f"https://huggingface.co/datasets/{candidate.name}/resolve/main"
            for filename in ["data.csv", "train.csv", "dataset.csv", "data/train.csv",
                             "data.parquet", "train.parquet", "data/train-00000-of-00001.parquet"]:
                url = f"{base_url}/{filename}"
                try:
                    response = self._http.get(url, follow_redirects=True)
                    if response.status_code == 200 and len(response.content) > 100:
                        if filename.endswith(".parquet"):
                            parquet_path = output_path / "temp.parquet"
                            parquet_path.write_bytes(response.content)
                            df = pd.read_parquet(parquet_path)
                            if len(df) > 10000:
                                df = df.sample(n=10000, random_state=42)
                            df.to_csv(csv_path, index=False)
                            parquet_path.unlink(missing_ok=True)
                        else:
                            csv_path.write_bytes(response.content)
                            df = pd.read_csv(csv_path)
                        if len(df) > 0:
                            logger.info(f"  Downloaded {filename} ({len(df)} rows)")
                            return csv_path
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"  HuggingFace direct download failed: {e}")

        return None

    def _download_kaggle(
        self, candidate: DatasetCandidate, csv_path: Path
    ) -> Optional[Path]:
        """Download from Kaggle using the API."""
        import shutil
        download_dir = csv_path.parent / "kaggle_tmp"
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            dataset_ref = candidate.download_url  # e.g., "owner/dataset-name"
            download_dir.mkdir(exist_ok=True)

            # Download files (Kaggle returns ZIP archives)
            api.dataset_download_files(dataset_ref, path=str(download_dir), unzip=True)

            # Find CSV files in the downloaded content
            csv_files = list(download_dir.rglob("*.csv"))
            if not csv_files:
                logger.warning("  Kaggle: no CSV found in download")
                return None
            
            # Try each CSV file with encoding fallback
            for f in csv_files:
                df = self._read_csv_with_encoding_fallback(f)
                if df is not None and len(df) > 0:
                    if len(df) > 10000:
                        df = df.sample(n=10000, random_state=42)
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    logger.info(f"  Kaggle: downloaded {f.name} ({len(df)} rows)")
                    return csv_path

            logger.warning("  Kaggle: no readable CSV found")
            return None

        except ImportError:
            logger.warning("  Kaggle: `kaggle` library not installed")
            return None
        except Exception as e:
            logger.warning(f"  Kaggle download failed: {e}")
            return None
        finally:
            # Always cleanup temp directory
            if download_dir.exists():
                shutil.rmtree(download_dir, ignore_errors=True)
    
    def _read_csv_with_encoding_fallback(
        self, filepath: Path, sep: str = ","
    ) -> Optional[pd.DataFrame]:
        """
        Read a CSV file, trying multiple encodings if UTF-8 fails.
        This handles datasets with non-ASCII characters (degree symbols, etc.).
        
        Args:
            filepath: Path to the CSV file
            sep: Field separator (default ',' for CSV, '\t' for TSV/TAB files)
        """
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, sep=sep, on_bad_lines='skip')
                if len(df) > 0:
                    logger.debug(f"  Successfully read {filepath.name} with {encoding} encoding")
                    return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # Other errors (like malformed CSV) — try next encoding
                logger.debug(f"  {encoding} failed for {filepath.name}: {e}")
                continue
        
        # Last resort: read as binary and decode with errors='replace'
        try:
            with open(filepath, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')
            from io import StringIO
            df = pd.read_csv(StringIO(content), sep=sep, on_bad_lines='skip')
            if len(df) > 0:
                logger.debug(f"  Read {filepath.name} with binary fallback")
                return df
        except Exception as e:
            logger.debug(f"  Binary fallback failed for {filepath.name}: {e}")
        
        return None

    def _download_dataverse(
        self, candidate: DatasetCandidate, csv_path: Path
    ) -> Optional[Path]:
        """Download from Harvard Dataverse using the API."""
        try:
            # Get dataset files list using the DOI
            global_id = candidate.download_url
            if not global_id:
                return None

            response = self._http.get(
                "https://dataverse.harvard.edu/api/datasets/:persistentId",
                params={"persistentId": global_id},
            )
            if response.status_code != 200:
                return None

            data = response.json().get("data", {})
            files = data.get("latestVersion", {}).get("files", [])

            # Find a CSV or tabular file
            for f in files:
                file_info = f.get("dataFile", {})
                filename = file_info.get("filename", "")
                file_id = file_info.get("id", "")

                if filename.endswith((".csv", ".tab", ".tsv")):
                    download_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
                    resp = self._http.get(download_url, follow_redirects=True)
                    if resp.status_code == 200 and len(resp.content) > 100:
                        csv_path.write_bytes(resp.content)
                        try:
                            sep = "\t" if filename.endswith((".tab", ".tsv")) else ","
                            # Use encoding fallback for non-UTF-8 files
                            df = self._read_csv_with_encoding_fallback(csv_path, sep=sep)
                            if df is not None and len(df) > 0:
                                if len(df) > 10000:
                                    df = df.sample(n=10000, random_state=42)
                                df.to_csv(csv_path, index=False, encoding='utf-8')
                                logger.info(f"  Dataverse: downloaded {filename} ({len(df)} rows)")
                                return csv_path
                        except Exception:
                            continue

            logger.warning("  Dataverse: no CSV/tabular file found")
            return None

        except Exception as e:
            logger.warning(f"  Dataverse download failed: {e}")
            return None

    def _download_direct_csv(
        self, url: str, csv_path: Path
    ) -> Optional[Path]:
        """Download a CSV directly from a URL (OpenML, Awesome, web search)."""
        if not url:
            return None
        try:
            response = self._http.get(url, follow_redirects=True)
            if response.status_code == 200 and len(response.content) > 100:
                csv_path.write_bytes(response.content)
                # Use encoding fallback for non-UTF-8 files
                df = self._read_csv_with_encoding_fallback(csv_path)
                if df is not None and len(df) > 0:
                    if len(df) > 10000:
                        df = df.sample(n=10000, random_state=42)
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    logger.info(f"  Direct download: {len(df)} rows")
                    return csv_path
        except Exception as e:
            logger.debug(f"  Direct CSV download failed for {url}: {e}")
        return None

    # =========================================================================
    # DOMAIN-SPECIFIC DOWNLOAD METHODS
    # =========================================================================
    
    def _download_worldbank(
        self, candidate: DatasetCandidate, csv_path: Path
    ) -> Optional[Path]:
        """Download data from World Bank API."""
        try:
            download_info = json.loads(candidate.download_url) if candidate.download_url else {}
            indicator_id = download_info.get("indicator_id")
            
            if not indicator_id:
                logger.warning(f"  World Bank: no indicator_id in download_info")
                return None
            
            # Fetch data for all countries, last 20 years
            api_url = (
                f"https://api.worldbank.org/v2/country/all/indicator/{indicator_id}"
                f"?format=json&per_page=5000&date=2004:2024"
            )
            
            response = self._http.get(api_url, timeout=60.0)
            if response.status_code != 200:
                logger.warning(f"  World Bank API returned {response.status_code}")
                return None
            
            data = response.json()
            if not isinstance(data, list) or len(data) < 2:
                logger.warning(f"  World Bank: unexpected response format")
                return None
            
            records = data[1] if len(data) > 1 else []
            if not records:
                logger.warning(f"  World Bank: no data records")
                return None
            
            # Convert to DataFrame
            rows = []
            for record in records:
                if record.get("value") is not None:
                    rows.append({
                        "country": record.get("country", {}).get("value", ""),
                        "country_code": record.get("countryiso3code", ""),
                        "year": record.get("date", ""),
                        "indicator": indicator_id,
                        "value": record.get("value"),
                    })
            
            if not rows:
                logger.warning(f"  World Bank: no non-null values")
                return None
            
            df = pd.DataFrame(rows)
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
            df.to_csv(csv_path, index=False)
            logger.info(f"  World Bank: downloaded {len(df)} rows for {indicator_id}")
            return csv_path
            
        except Exception as e:
            logger.error(f"  World Bank download failed: {e}")
            return None
    
    def _download_fred(
        self, candidate: DatasetCandidate, csv_path: Path
    ) -> Optional[Path]:
        """Download data from FRED API."""
        try:
            download_info = json.loads(candidate.download_url) if candidate.download_url else {}
            api_url = download_info.get("api_url")
            
            if not api_url:
                logger.warning(f"  FRED: no api_url in download_info")
                return None
            
            response = self._http.get(api_url, timeout=60.0)
            if response.status_code != 200:
                logger.warning(f"  FRED API returned {response.status_code}")
                return None
            
            data = response.json()
            observations = data.get("observations", [])
            
            if not observations:
                logger.warning(f"  FRED: no observations")
                return None
            
            # Convert to DataFrame
            rows = []
            for obs in observations:
                value = obs.get("value", ".")
                if value != ".":
                    try:
                        rows.append({
                            "date": obs.get("date", ""),
                            "value": float(value),
                        })
                    except (ValueError, TypeError):
                        continue
            
            if not rows:
                logger.warning(f"  FRED: no valid values")
                return None
            
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            logger.info(f"  FRED: downloaded {len(df)} observations")
            return csv_path
            
        except Exception as e:
            logger.error(f"  FRED download failed: {e}")
            return None
    
    def _download_eurostat(
        self, candidate: DatasetCandidate, csv_path: Path
    ) -> Optional[Path]:
        """Download data from Eurostat API."""
        try:
            download_info = json.loads(candidate.download_url) if candidate.download_url else {}
            api_url = download_info.get("api_url")
            
            if not api_url:
                dataset_code = download_info.get("dataset_code", "")
                if dataset_code:
                    api_url = f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{dataset_code}?format=SDMX-CSV"
                else:
                    logger.warning(f"  Eurostat: no api_url or dataset_code")
                    return None
            
            response = self._http.get(api_url, timeout=120.0)
            if response.status_code != 200:
                logger.warning(f"  Eurostat API returned {response.status_code}")
                return None
            
            # SDMX-CSV format
            csv_path.write_bytes(response.content)
            df = pd.read_csv(csv_path)
            
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
                df.to_csv(csv_path, index=False)
            
            logger.info(f"  Eurostat: downloaded {len(df)} rows")
            return csv_path
            
        except Exception as e:
            logger.error(f"  Eurostat download failed: {e}")
            return None
    
    def _download_ncbi_geo(
        self, candidate: DatasetCandidate, csv_path: Path
    ) -> Optional[Path]:
        """Download metadata from NCBI GEO (full data requires manual processing)."""
        try:
            # GEO datasets are complex (often require R/Bioconductor)
            # For now, download the metadata/annotation as a summary
            download_info = json.loads(candidate.download_url) if candidate.download_url else {}
            accession = download_info.get("accession", "")
            
            if not accession:
                logger.warning(f"  NCBI GEO: no accession")
                return None
            
            # Get series matrix (simplified data)
            # Note: Full GEO data requires specialized tools
            soft_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}&targ=self&form=text&view=brief"
            
            response = self._http.get(soft_url, timeout=60.0)
            if response.status_code != 200:
                logger.warning(f"  NCBI GEO: failed to fetch metadata")
                return None
            
            # Parse SOFT format to extract sample info
            lines = response.text.split("\n")
            samples = []
            current_sample = {}
            
            for line in lines:
                if line.startswith("^SAMPLE"):
                    if current_sample:
                        samples.append(current_sample)
                    current_sample = {"accession": line.split("=")[-1].strip() if "=" in line else ""}
                elif line.startswith("!Sample_"):
                    key = line.split("=")[0].replace("!Sample_", "").strip()
                    value = line.split("=")[-1].strip() if "=" in line else ""
                    current_sample[key] = value
            
            if current_sample:
                samples.append(current_sample)
            
            if not samples:
                logger.warning(f"  NCBI GEO: no samples found in metadata")
                return None
            
            df = pd.DataFrame(samples)
            df.to_csv(csv_path, index=False)
            logger.info(f"  NCBI GEO: downloaded metadata for {len(df)} samples")
            return csv_path
            
        except Exception as e:
            logger.error(f"  NCBI GEO download failed: {e}")
            return None
    
    def _download_who(
        self, candidate: DatasetCandidate, csv_path: Path
    ) -> Optional[Path]:
        """Download data from WHO Global Health Observatory API."""
        try:
            download_info = json.loads(candidate.download_url) if candidate.download_url else {}
            api_url = download_info.get("api_url")
            indicator_code = download_info.get("indicator_code", "")
            
            if not api_url and indicator_code:
                api_url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
            
            if not api_url:
                logger.warning(f"  WHO: no api_url or indicator_code")
                return None
            
            response = self._http.get(api_url, timeout=60.0)
            if response.status_code != 200:
                logger.warning(f"  WHO API returned {response.status_code}")
                return None
            
            data = response.json()
            records = data.get("value", [])
            
            if not records:
                logger.warning(f"  WHO: no data records")
                return None
            
            # Convert to DataFrame
            rows = []
            for record in records:
                rows.append({
                    "country": record.get("SpatialDim", ""),
                    "year": record.get("TimeDim", ""),
                    "value": record.get("NumericValue"),
                    "indicator": indicator_code,
                })
            
            df = pd.DataFrame(rows)
            # Filter out rows with null values
            df = df.dropna(subset=["value"])
            
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
            
            df.to_csv(csv_path, index=False)
            logger.info(f"  WHO: downloaded {len(df)} rows for {indicator_code}")
            return csv_path
            
        except Exception as e:
            logger.error(f"  WHO download failed: {e}")
            return None

    def _empty_result(
        self, queries: List[str] = None, candidates_found: int = 0
    ) -> DatasetSearchResult:
        """Return an empty/failed result."""
        logger.info(f"  ❌ Dataset search returning empty result (candidates_found={candidates_found})")
        return DatasetSearchResult(
            success=False,
            path="",
            source="",
            dataset_name="",
            description="",
            relevance_score=0.0,
            candidates_found=candidates_found,
            queries_used=queries or [],
        )
    
    def close(self) -> None:
        """Close the HTTP client to prevent resource leaks."""
        try:
            self._http.close()
        except Exception:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()