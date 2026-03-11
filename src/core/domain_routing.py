# -*- coding: utf-8 -*-
"""
Domain-Aware Dataset Search Routing.

Detects research domain from objectives and routes to specialized data sources:
- Economics/Finance → FRED, World Bank, Eurostat, IMF
- Genomics/Biology → NCBI GEO, UniProt, GenBank
- Climate/Environment → NOAA, NASA Earthdata
- Social Science → Census, IPUMS, Pew
- Health/Medical → CDC, WHO, ClinicalTrials
- General → Kaggle, HuggingFace, OpenML (fallback)
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class ResearchDomain(Enum):
    """Research domain classifications."""
    ECONOMICS = "economics"
    GENOMICS = "genomics"
    CLIMATE = "climate"
    SOCIAL = "social"
    HEALTH = "health"
    GENERAL = "general"


@dataclass
class DomainSource:
    """A domain-specific data source."""
    name: str
    domain: ResearchDomain
    base_url: str
    requires_auth: bool = False
    description: str = ""


# Domain-specific data sources
DOMAIN_SOURCES: Dict[ResearchDomain, List[DomainSource]] = {
    ResearchDomain.ECONOMICS: [
        DomainSource(
            name="FRED",
            domain=ResearchDomain.ECONOMICS,
            base_url="https://api.stlouisfed.org/fred",
            requires_auth=True,  # Requires API key
            description="Federal Reserve Economic Data - US economic indicators",
        ),
        DomainSource(
            name="World Bank",
            domain=ResearchDomain.ECONOMICS,
            base_url="https://api.worldbank.org/v2",
            requires_auth=False,
            description="World Bank Open Data - global development indicators",
        ),
        DomainSource(
            name="Eurostat",
            domain=ResearchDomain.ECONOMICS,
            base_url="https://ec.europa.eu/eurostat/api/dissemination",
            requires_auth=False,
            description="European Union statistics",
        ),
    ],
    ResearchDomain.GENOMICS: [
        DomainSource(
            name="NCBI GEO",
            domain=ResearchDomain.GENOMICS,
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            requires_auth=False,
            description="Gene Expression Omnibus - genomics datasets",
        ),
        DomainSource(
            name="UniProt",
            domain=ResearchDomain.GENOMICS,
            base_url="https://rest.uniprot.org",
            requires_auth=False,
            description="Universal Protein Resource - protein sequences and functions",
        ),
    ],
    ResearchDomain.CLIMATE: [
        DomainSource(
            name="NOAA",
            domain=ResearchDomain.CLIMATE,
            base_url="https://www.ncei.noaa.gov/cdo-web/api/v2",
            requires_auth=True,  # Requires free API token
            description="National Oceanic and Atmospheric Administration climate data",
        ),
    ],
    ResearchDomain.HEALTH: [
        DomainSource(
            name="CDC WONDER",
            domain=ResearchDomain.HEALTH,
            base_url="https://wonder.cdc.gov",
            requires_auth=False,
            description="CDC public health statistics",
        ),
    ],
    ResearchDomain.SOCIAL: [
        DomainSource(
            name="US Census",
            domain=ResearchDomain.SOCIAL,
            base_url="https://api.census.gov/data",
            requires_auth=True,  # Requires free API key
            description="US Census Bureau demographic data",
        ),
    ],
}

# Keywords for domain detection (used as backup to LLM)
DOMAIN_KEYWORDS: Dict[ResearchDomain, List[str]] = {
    ResearchDomain.ECONOMICS: [
        "gdp", "inflation", "unemployment", "interest rate", "monetary policy",
        "fiscal", "trade", "export", "import", "currency", "exchange rate",
        "stock", "bond", "investment", "market", "price", "cost", "wage",
        "income", "poverty", "inequality", "economic growth", "recession",
        "federal reserve", "central bank", "treasury", "finance", "banking",
    ],
    ResearchDomain.GENOMICS: [
        "gene", "genome", "dna", "rna", "protein", "expression", "sequencing",
        "mutation", "variant", "chromosome", "allele", "transcriptome",
        "proteome", "metabolome", "pathway", "cell", "tissue", "organism",
        "species", "evolution", "phylogenetic", "blast", "alignment",
        "bioinformatics", "molecular", "genetic", "hereditary",
    ],
    ResearchDomain.CLIMATE: [
        "climate", "weather", "temperature", "precipitation", "rainfall",
        "drought", "flood", "hurricane", "tornado", "storm", "wind",
        "carbon", "emission", "greenhouse", "sea level", "ice", "glacier",
        "ocean", "atmosphere", "ozone", "pollution", "air quality",
        "renewable", "solar", "wind energy", "fossil fuel",
    ],
    ResearchDomain.HEALTH: [
        "disease", "mortality", "morbidity", "health", "medical", "clinical",
        "patient", "treatment", "drug", "pharmaceutical", "vaccine",
        "epidemic", "pandemic", "infection", "virus", "bacteria",
        "cancer", "diabetes", "heart", "stroke", "mental health",
        "hospital", "healthcare", "insurance", "medicare", "medicaid",
    ],
    ResearchDomain.SOCIAL: [
        "population", "demographic", "census", "survey", "poll",
        "education", "school", "college", "university", "literacy",
        "crime", "justice", "police", "prison", "court",
        "election", "voting", "political", "government", "policy",
        "immigration", "migration", "refugee", "ethnic", "racial",
        "gender", "age", "family", "household", "community",
    ],
}


class DomainDetector:
    """
    Detects research domain from objective text.
    
    Uses a combination of:
    1. Keyword matching (fast, reliable)
    2. LLM classification (accurate, slower)
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    def detect(self, objective: str, use_llm: bool = True) -> Tuple[ResearchDomain, float]:
        """
        Detect the research domain from an objective.
        
        Args:
            objective: Research objective text
            use_llm: Whether to use LLM for classification
            
        Returns:
            Tuple of (domain, confidence)
        """
        # First try keyword matching (fast)
        keyword_domain, keyword_confidence = self._detect_by_keywords(objective)
        
        if keyword_confidence >= 0.7:
            logger.info(f"🎯 Domain detected by keywords: {keyword_domain.value} ({keyword_confidence:.0%})")
            return keyword_domain, keyword_confidence
        
        # Use LLM for uncertain cases
        if use_llm and self.llm:
            llm_domain, llm_confidence = self._detect_by_llm(objective)
            if llm_confidence >= 0.5:
                logger.info(f"🎯 Domain detected by LLM: {llm_domain.value} ({llm_confidence:.0%})")
                return llm_domain, llm_confidence
        
        # Fall back to keyword result or general
        if keyword_confidence >= 0.3:
            return keyword_domain, keyword_confidence
        
        logger.info(f"🎯 Domain: general (no strong domain signal)")
        return ResearchDomain.GENERAL, 0.5
    
    def _detect_by_keywords(self, objective: str) -> Tuple[ResearchDomain, float]:
        """Detect domain using keyword matching."""
        text_lower = objective.lower()
        
        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            # Normalize by number of keywords (some domains have more)
            scores[domain] = matches / len(keywords) if keywords else 0
        
        if not scores:
            return ResearchDomain.GENERAL, 0.0
        
        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]
        
        # Convert to confidence (0-1)
        # 0.1 normalized score = ~0.5 confidence, 0.2 = ~0.8
        confidence = min(1.0, best_score * 5)
        
        return best_domain, confidence
    
    def _detect_by_llm(self, objective: str) -> Tuple[ResearchDomain, float]:
        """Detect domain using LLM classification."""
        prompt = f"""Classify this research objective into ONE domain:

Research Objective: {objective}

Domains:
- economics: Economic indicators, finance, markets, trade, monetary policy
- genomics: Genes, proteins, DNA/RNA, molecular biology, bioinformatics
- climate: Weather, climate change, environmental science, emissions
- health: Disease, medical research, public health, clinical studies
- social: Demographics, surveys, education, crime, politics
- general: Does not fit clearly into any specific domain

Respond with JSON only:
{{"domain": "<domain>", "confidence": <0.0-1.0>, "reason": "<brief reason>"}}"""

        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="query_classification",
                temperature=0.1,
            )
            
            text = response.content.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                domain_str = result.get("domain", "general").lower()
                confidence = float(result.get("confidence", 0.5))
                
                # Map string to enum
                domain_map = {
                    "economics": ResearchDomain.ECONOMICS,
                    "genomics": ResearchDomain.GENOMICS,
                    "climate": ResearchDomain.CLIMATE,
                    "health": ResearchDomain.HEALTH,
                    "social": ResearchDomain.SOCIAL,
                    "general": ResearchDomain.GENERAL,
                }
                
                domain = domain_map.get(domain_str, ResearchDomain.GENERAL)
                return domain, confidence
                
        except Exception as e:
            logger.debug(f"LLM domain detection failed: {e}")
        
        return ResearchDomain.GENERAL, 0.0




class DomainDatasetSearcher:
    """
    Searches domain-specific data repositories.
    
    Routes to specialized APIs based on detected research domain:
    - Economics → FRED, World Bank, Eurostat
    - Genomics → NCBI GEO, UniProt
    - Climate → NOAA
    - Health → CDC
    - Social → Census
    """
    
    def __init__(self, http_client: httpx.Client = None):
        self._http = http_client or httpx.Client(timeout=30.0)
        
        # API keys from environment (optional)
        import os
        self.fred_api_key = os.getenv("FRED_API_KEY")
        self.census_api_key = os.getenv("CENSUS_API_KEY")
        self.noaa_token = os.getenv("NOAA_TOKEN")
    
    def search(
        self,
        domain: ResearchDomain,
        queries: List[str],
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search domain-specific sources for datasets.
        
        Returns list of dataset candidates with standard format:
        {
            "name": str,
            "description": str,
            "source": str,
            "tags": List[str],
            "download_info": Any,  # Source-specific download info
        }
        """
        results = []
        
        if domain == ResearchDomain.ECONOMICS:
            results.extend(self._search_world_bank(queries, max_results))
            if self.fred_api_key:
                results.extend(self._search_fred(queries, max_results))
            results.extend(self._search_eurostat(queries, max_results))
            
        elif domain == ResearchDomain.GENOMICS:
            results.extend(self._search_ncbi_geo(queries, max_results))
            results.extend(self._search_uniprot(queries, max_results))
            
        elif domain == ResearchDomain.CLIMATE:
            if self.noaa_token:
                results.extend(self._search_noaa(queries, max_results))
                
        elif domain == ResearchDomain.HEALTH:
            results.extend(self._search_who(queries, max_results))
            
        elif domain == ResearchDomain.SOCIAL:
            if self.census_api_key:
                results.extend(self._search_census(queries, max_results))
        
        return results[:max_results]
    
    # =========================================================================
    # ECONOMICS SOURCES
    # =========================================================================
    
    def _search_world_bank(
        self, queries: List[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """Search World Bank Open Data API (no auth required)."""
        results = []
        seen = set()
        
        for query in queries[:2]:
            try:
                # Search indicators
                response = self._http.get(
                    "https://api.worldbank.org/v2/indicator",
                    params={
                        "format": "json",
                        "per_page": 20,
                        "source": "all",
                    },
                )
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                if not isinstance(data, list) or len(data) < 2:
                    continue
                
                indicators = data[1] if len(data) > 1 else []
                query_lower = query.lower()
                
                for ind in indicators:
                    name = ind.get("name", "")
                    ind_id = ind.get("id", "")
                    
                    if ind_id in seen:
                        continue
                    
                    # Filter by query relevance
                    text = (name + " " + ind.get("sourceNote", "")).lower()
                    if not any(w in text for w in query_lower.split()):
                        continue
                    
                    seen.add(ind_id)
                    results.append({
                        "name": f"worldbank/{ind_id}",
                        "description": name + ": " + ind.get("sourceNote", "")[:200],
                        "source": "World Bank",
                        "tags": ["economics", "global", ind.get("source", {}).get("value", "")],
                        "download_info": {
                            "indicator_id": ind_id,
                            "api_url": f"https://api.worldbank.org/v2/country/all/indicator/{ind_id}?format=json&per_page=1000",
                        },
                    })
                    
            except Exception as e:
                logger.debug(f"World Bank search error: {e}")
                continue
        
        return results[:max_results]
    
    def _search_fred(
        self, queries: List[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """Search FRED (Federal Reserve Economic Data) - requires API key."""
        if not self.fred_api_key:
            return []
        
        results = []
        seen = set()
        
        for query in queries[:2]:
            try:
                response = self._http.get(
                    "https://api.stlouisfed.org/fred/series/search",
                    params={
                        "api_key": self.fred_api_key,
                        "search_text": query,
                        "file_type": "json",
                        "limit": 10,
                        "order_by": "popularity",
                    },
                )
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                series_list = data.get("seriess", [])
                
                for series in series_list:
                    series_id = series.get("id", "")
                    if series_id in seen:
                        continue
                    seen.add(series_id)
                    
                    results.append({
                        "name": f"fred/{series_id}",
                        "description": series.get("title", "") + ": " + series.get("notes", "")[:200],
                        "source": "FRED",
                        "tags": ["economics", "us", series.get("frequency", "")],
                        "download_info": {
                            "series_id": series_id,
                            "api_url": f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.fred_api_key}&file_type=json",
                        },
                    })
                    
            except Exception as e:
                logger.debug(f"FRED search error: {e}")
                continue
        
        return results[:max_results]
    
    def _search_eurostat(
        self, queries: List[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """Search Eurostat (no auth required)."""
        results = []
        seen = set()
        
        for query in queries[:2]:
            try:
                # Eurostat table of contents search
                response = self._http.get(
                    "https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc",
                    params={"lang": "EN"},
                    headers={"Accept": "application/json"},
                )
                
                if response.status_code != 200:
                    continue
                
                # The response is large, so we filter locally
                data = response.json()
                items = data.get("items", []) if isinstance(data, dict) else []
                query_lower = query.lower()
                
                for item in items[:500]:  # Limit scan
                    code = item.get("code", "")
                    title = item.get("title", "")
                    
                    if code in seen:
                        continue
                    
                    if not any(w in title.lower() for w in query_lower.split()):
                        continue
                    
                    seen.add(code)
                    results.append({
                        "name": f"eurostat/{code}",
                        "description": title,
                        "source": "Eurostat",
                        "tags": ["economics", "europe", "eu"],
                        "download_info": {
                            "dataset_code": code,
                            "api_url": f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{code}?format=SDMX-CSV",
                        },
                    })
                    
                    if len(results) >= max_results:
                        break
                    
            except Exception as e:
                logger.debug(f"Eurostat search error: {e}")
                continue
        
        return results[:max_results]
    
    # =========================================================================
    # GENOMICS SOURCES
    # =========================================================================
    
    def _search_ncbi_geo(
        self, queries: List[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """Search NCBI GEO (Gene Expression Omnibus) - no auth required."""
        results = []
        seen = set()
        
        for query in queries[:2]:
            try:
                # Use E-utilities to search GEO
                response = self._http.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={
                        "db": "gds",  # GEO DataSets
                        "term": query,
                        "retmax": 10,
                        "retmode": "json",
                    },
                )
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                
                if not id_list:
                    continue
                
                # Fetch details for each ID
                detail_response = self._http.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                    params={
                        "db": "gds",
                        "id": ",".join(id_list),
                        "retmode": "json",
                    },
                )
                
                if detail_response.status_code != 200:
                    continue
                
                details = detail_response.json().get("result", {})
                
                for gds_id in id_list:
                    if gds_id in seen or gds_id == "uids":
                        continue
                    seen.add(gds_id)
                    
                    info = details.get(gds_id, {})
                    title = info.get("title", f"GEO Dataset {gds_id}")
                    summary = info.get("summary", "")
                    accession = info.get("accession", gds_id)
                    
                    results.append({
                        "name": f"geo/{accession}",
                        "description": title + ": " + summary[:200],
                        "source": "NCBI GEO",
                        "tags": ["genomics", info.get("gdstype", ""), info.get("organism", "")],
                        "download_info": {
                            "accession": accession,
                            "ftp_link": f"https://ftp.ncbi.nlm.nih.gov/geo/datasets/{accession[:5]}nnn/{accession}/",
                        },
                    })
                    
            except Exception as e:
                logger.debug(f"NCBI GEO search error: {e}")
                continue
        
        return results[:max_results]
    
    def _search_uniprot(
        self, queries: List[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """Search UniProt (no auth required)."""
        results = []
        
        for query in queries[:2]:
            try:
                response = self._http.get(
                    "https://rest.uniprot.org/uniprotkb/search",
                    params={
                        "query": query,
                        "format": "json",
                        "size": 10,
                    },
                )
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                entries = data.get("results", [])
                
                for entry in entries:
                    accession = entry.get("primaryAccession", "")
                    protein_name = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
                    organism = entry.get("organism", {}).get("scientificName", "")
                    
                    results.append({
                        "name": f"uniprot/{accession}",
                        "description": f"{protein_name} ({organism})",
                        "source": "UniProt",
                        "tags": ["genomics", "protein", organism],
                        "download_info": {
                            "accession": accession,
                            "url": f"https://rest.uniprot.org/uniprotkb/{accession}.fasta",
                        },
                    })
                    
            except Exception as e:
                logger.debug(f"UniProt search error: {e}")
                continue
        
        return results[:max_results]
    
    # =========================================================================
    # OTHER DOMAIN SOURCES
    # =========================================================================
    
    def _search_noaa(
        self, queries: List[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """Search NOAA Climate Data Online - requires token."""
        if not self.noaa_token:
            return []
        
        results = []
        
        for query in queries[:2]:
            try:
                response = self._http.get(
                    "https://www.ncei.noaa.gov/cdo-web/api/v2/datasets",
                    headers={"token": self.noaa_token},
                )
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                datasets = data.get("results", [])
                query_lower = query.lower()
                
                for ds in datasets:
                    name = ds.get("name", "")
                    if not any(w in name.lower() for w in query_lower.split()):
                        continue
                    
                    results.append({
                        "name": f"noaa/{ds.get('id', '')}",
                        "description": name,
                        "source": "NOAA",
                        "tags": ["climate", "weather", "environment"],
                        "download_info": {
                            "dataset_id": ds.get("id", ""),
                            "min_date": ds.get("mindate", ""),
                            "max_date": ds.get("maxdate", ""),
                        },
                    })
                    
            except Exception as e:
                logger.debug(f"NOAA search error: {e}")
                continue
        
        return results[:max_results]
    
    def _search_who(
        self, queries: List[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """Search WHO Global Health Observatory (no auth required)."""
        results = []
        
        for query in queries[:2]:
            try:
                response = self._http.get(
                    "https://ghoapi.azureedge.net/api/Indicator",
                )
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                indicators = data.get("value", [])
                query_lower = query.lower()
                
                for ind in indicators[:200]:
                    name = ind.get("IndicatorName", "")
                    code = ind.get("IndicatorCode", "")
                    
                    if not any(w in name.lower() for w in query_lower.split()):
                        continue
                    
                    results.append({
                        "name": f"who/{code}",
                        "description": name,
                        "source": "WHO GHO",
                        "tags": ["health", "global"],
                        "download_info": {
                            "indicator_code": code,
                            "api_url": f"https://ghoapi.azureedge.net/api/{code}",
                        },
                    })
                    
                    if len(results) >= max_results:
                        break
                    
            except Exception as e:
                logger.debug(f"WHO search error: {e}")
                continue
        
        return results[:max_results]
    
    def _search_census(
        self, queries: List[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """Search US Census API (requires API key)."""
        if not self.census_api_key:
            return []
        
        # Census API is complex - return a few known useful datasets
        results = [
            {
                "name": "census/acs5",
                "description": "American Community Survey 5-Year Estimates - demographic, economic, housing data",
                "source": "US Census",
                "tags": ["social", "demographics", "us"],
                "download_info": {
                    "dataset": "acs/acs5",
                    "api_key": self.census_api_key,
                },
            },
            {
                "name": "census/pep",
                "description": "Population Estimates Program - annual population estimates",
                "source": "US Census",
                "tags": ["social", "population", "us"],
                "download_info": {
                    "dataset": "pep/population",
                    "api_key": self.census_api_key,
                },
            },
        ]
        
        return results


def get_domain_sources(domain: ResearchDomain) -> List[str]:
    """Get list of available source names for a domain."""
    sources = DOMAIN_SOURCES.get(domain, [])
    return [s.name for s in sources]
