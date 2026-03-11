# -*- coding: utf-8 -*-
"""
Inquiro Configuration Management.

Loads settings from environment variables (via .env file).
Provides typed, validated configuration for all components.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field
from dotenv import load_dotenv


# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_env(key: str, default: str = None, required: bool = False) -> Optional[str]:
    """Get environment variable with optional default."""
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int) -> int:
    """Get integer environment variable."""
    return int(os.getenv(key, str(default)))


def get_env_float(key: str, default: float) -> float:
    """Get float environment variable."""
    return float(os.getenv(key, str(default)))


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = field(default_factory=lambda: get_env("LLM_PROVIDER", "ollama"))
    model: str = field(default_factory=lambda: get_env("LLM_MODEL", "llama3.1"))
    temperature: float = field(default_factory=lambda: get_env_float("LLM_TEMPERATURE", 0.7))
    max_tokens: int = field(default_factory=lambda: get_env_int("LLM_MAX_TOKENS", 4096))
    
    # API Keys
    openai_api_key: Optional[str] = field(default_factory=lambda: get_env("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: get_env("ANTHROPIC_API_KEY"))
    google_api_key: Optional[str] = field(default_factory=lambda: get_env("GOOGLE_API_KEY"))
    groq_api_key: Optional[str] = field(default_factory=lambda: get_env("GROQ_API_KEY"))
    
    # Local LLM URLs
    ollama_base_url: str = field(default_factory=lambda: get_env("OLLAMA_BASE_URL", "http://localhost:11434"))
    lm_studio_base_url: str = field(default_factory=lambda: get_env("LM_STUDIO_BASE_URL", "http://localhost:1234"))


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = field(default_factory=lambda: get_env("DATABASE_PATH", "./data/world_model.db"))


@dataclass
class OutputConfig:
    """Output settings."""
    output_dir: str = field(default_factory=lambda: get_env("OUTPUT_DIR", "./outputs"))
    save_notebooks: bool = field(default_factory=lambda: get_env_bool("SAVE_NOTEBOOKS", True))


@dataclass
class ExecutionConfig:
    """Execution settings."""
    max_cycles: int = field(default_factory=lambda: get_env_int("MAX_CYCLES", 20))
    tasks_per_cycle: int = field(default_factory=lambda: get_env_int("TASKS_PER_CYCLE", 10))
    max_retries: int = field(default_factory=lambda: get_env_int("MAX_RETRIES", 3))
    timeout_seconds: int = field(default_factory=lambda: get_env_int("EXECUTION_TIMEOUT_SECONDS", 300))
    run_mode: str = field(default_factory=lambda: get_env("RUN_MODE", "auto"))


@dataclass
class DockerConfig:
    """Docker sandbox settings."""
    image: str = field(default_factory=lambda: get_env("DOCKER_IMAGE", "inquiro-sandbox:latest"))
    memory_limit: str = field(default_factory=lambda: get_env("DOCKER_MEMORY_LIMIT", "4g"))
    cpu_limit: float = field(default_factory=lambda: get_env_float("DOCKER_CPU_LIMIT", 2.0))


@dataclass
class LiteratureConfig:
    """Literature search settings."""
    max_papers_per_search: int = field(default_factory=lambda: get_env_int("MAX_PAPERS_PER_SEARCH", 20))
    max_papers_to_read: int = field(default_factory=lambda: get_env_int("MAX_PAPERS_TO_READ", 15))
    seed_papers_dir: Optional[str] = field(default_factory=lambda: get_env("SEED_PAPERS_DIR"))
    
    # API Keys for literature sources
    semantic_scholar_api_key: Optional[str] = field(default_factory=lambda: get_env("SEMANTIC_SCHOLAR_API_KEY"))
    core_api_key: Optional[str] = field(default_factory=lambda: get_env("CORE_API_KEY"))
    dimensions_api_key: Optional[str] = field(default_factory=lambda: get_env("DIMENSIONS_API_KEY"))
    dimensions_username: Optional[str] = field(default_factory=lambda: get_env("DIMENSIONS_USERNAME"))
    dimensions_password: Optional[str] = field(default_factory=lambda: get_env("DIMENSIONS_PASSWORD"))


@dataclass
class GROBIDConfig:
    """GROBID PDF parsing settings."""
    enabled: bool = field(default_factory=lambda: get_env_bool("GROBID_ENABLED", True))
    base_url: str = field(default_factory=lambda: get_env("GROBID_URL", "http://localhost:8070"))
    timeout: float = field(default_factory=lambda: get_env_float("GROBID_TIMEOUT", 120.0))
    consolidate_citations: bool = field(default_factory=lambda: get_env_bool("GROBID_CONSOLIDATE_CITATIONS", True))


@dataclass
class RAGConfig:
    """RAG (Retrieval Augmented Generation) settings."""
    chunk_size: int = field(default_factory=lambda: get_env_int("RAG_CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: get_env_int("RAG_CHUNK_OVERLAP", 200))
    top_k: int = field(default_factory=lambda: get_env_int("RAG_TOP_K", 10))


@dataclass
class ModelRouterConfig:
    """
    Per-role model routing configuration.
    
    Tiers:
        - strong: Critical reasoning (orchestrator, reports, findings)
        - fast: Simple extraction/formatting (queries, ranking)
        - code: Code generation (data analysis scripts)
        - local: Fallback for cheap/offline tasks
    
    Usage in .env:
        ROUTER_ENABLED=true
        ROUTER_CODE_PROVIDER=ollama
        ROUTER_CODE_MODEL=qwen2.5-coder:7b
    """
    # Strong model — for critical reasoning tasks
    strong_provider: str = field(default_factory=lambda: get_env("ROUTER_STRONG_PROVIDER", "gemini"))
    strong_model: str = field(default_factory=lambda: get_env("ROUTER_STRONG_MODEL", "gemini-2.0-flash"))
    
    # Fast model — for simple extraction/formatting tasks  
    fast_provider: str = field(default_factory=lambda: get_env("ROUTER_FAST_PROVIDER", "groq"))
    fast_model: str = field(default_factory=lambda: get_env("ROUTER_FAST_MODEL", "llama-3.3-70b-versatile"))
    
    # Code model — specialized for code generation (Data Agent)
    code_provider: str = field(default_factory=lambda: get_env("ROUTER_CODE_PROVIDER", "ollama"))
    code_model: str = field(default_factory=lambda: get_env("ROUTER_CODE_MODEL", "qwen2.5-coder:7b"))
    
    # Local model — fallback / cheap tasks
    local_provider: str = field(default_factory=lambda: get_env("ROUTER_LOCAL_PROVIDER", "ollama"))
    local_model: str = field(default_factory=lambda: get_env("ROUTER_LOCAL_MODEL", "qwen3:8b"))
    
    # Enable/disable routing (when False, all calls use the default LLM_PROVIDER/LLM_MODEL)
    enabled: bool = field(default_factory=lambda: get_env_bool("ROUTER_ENABLED", False))


@dataclass
class SkillsConfig:
    """
    Skill system configuration.
    
    Skills are markdown files that inject domain knowledge into LLM prompts,
    enabling smaller models to perform better on specialized tasks.
    
    Usage in .env:
        SKILLS_ENABLED=true
        SKILLS_AUTO_GENERATE=true
        SKILLS_DIR=./skills
    """
    # Enable/disable the skill system
    enabled: bool = field(default_factory=lambda: get_env_bool("SKILLS_ENABLED", True))
    
    # Auto-generate missing skills using teacher model
    auto_generate: bool = field(default_factory=lambda: get_env_bool("SKILLS_AUTO_GENERATE", True))
    
    # Directory containing skill files
    skills_dir: str = field(default_factory=lambda: get_env("SKILLS_DIR", "./skills"))


@dataclass
class PeerReviewConfig:
    """
    Automated peer review configuration.
    
    Peer review evaluates generated reports using academic-style criteria:
    - Soundness: Are claims well-supported by evidence?
    - Significance: Is the research impactful?
    - Novelty: Are findings new or non-obvious?
    - Clarity: Is the report well-organized?
    
    Usage in .env:
        PEER_REVIEW_ENABLED=true
        PEER_REVIEW_MIN_LENGTH=500
    """
    # Enable/disable peer review
    enabled: bool = field(default_factory=lambda: get_env_bool("PEER_REVIEW_ENABLED", True))
    
    # Minimum report length (chars) to trigger review
    min_length: int = field(default_factory=lambda: int(get_env("PEER_REVIEW_MIN_LENGTH", "500")))


@dataclass
class FigureVerificationConfig:
    """
    VLM Figure Verification configuration.
    
    Uses vision language models to verify generated figures match
    their captions and support claimed findings.
    
    Usage in .env:
        FIGURE_VERIFICATION_ENABLED=true
        FIGURE_VERIFICATION_MAX_FIGURES=10
        FIGURE_VERIFICATION_MIN_SCORE=2.0
    """
    # Enable/disable figure verification
    enabled: bool = field(default_factory=lambda: get_env_bool("FIGURE_VERIFICATION_ENABLED", True))
    
    # Maximum figures to verify per run
    max_figures: int = field(default_factory=lambda: int(get_env("FIGURE_VERIFICATION_MAX_FIGURES", "10")))
    
    # Minimum score (1-4) to pass verification
    min_score: float = field(default_factory=lambda: float(get_env("FIGURE_VERIFICATION_MIN_SCORE", "2.0")))


@dataclass
class ReproducibilityConfig:
    """
    Reproducibility package configuration.
    
    Generates everything needed to reproduce a INQUIRO run:
    - Environment specs (Python version, packages)
    - Configuration snapshot
    - Data manifests with checksums
    - Random seeds
    - Reproduction script
    
    Usage in .env:
        REPRODUCIBILITY_ENABLED=true
    """
    # Enable/disable reproducibility package generation
    enabled: bool = field(default_factory=lambda: get_env_bool("REPRODUCIBILITY_ENABLED", True))


@dataclass
class DomainSkillsConfig:
    """
    Domain-specific skill prompts configuration.
    
    Injects specialized knowledge for different research domains:
    - Economics, Biology, Medicine, Physics, Psychology
    - Computer Science, Environmental, Social Science
    
    Usage in .env:
        DOMAIN_SKILLS_ENABLED=true
    """
    # Enable/disable domain skill injection
    enabled: bool = field(default_factory=lambda: get_env_bool("DOMAIN_SKILLS_ENABLED", True))


@dataclass
class LatexConfig:
    """
    LaTeX compilation configuration.
    
    Enables conversion of INQUIRO reports to LaTeX and PDF format.
    Supports multiple academic templates: NeurIPS, IEEE, arXiv, plain.
    
    Usage in .env:
        LATEX_ENABLED=true
        LATEX_TEMPLATE=arxiv
        LATEX_COMPILE_PDF=true
        LATEX_AUTHOR=Your Name
    """
    # Enable/disable LaTeX compilation
    enabled: bool = field(default_factory=lambda: get_env_bool("LATEX_ENABLED", False))
    # Template: plain, arxiv, neurips, ieee
    template: str = field(default_factory=lambda: os.getenv("LATEX_TEMPLATE", "plain"))
    # Whether to compile to PDF (requires LaTeX installed)
    compile_pdf: bool = field(default_factory=lambda: get_env_bool("LATEX_COMPILE_PDF", True))
    # LaTeX compiler: pdflatex, xelatex, lualatex
    compiler: str = field(default_factory=lambda: os.getenv("LATEX_COMPILER", "pdflatex"))
    # Author name for documents
    author: str = field(default_factory=lambda: os.getenv("LATEX_AUTHOR", "INQUIRO Autonomous Research System"))
    # Institution (optional)
    institution: str = field(default_factory=lambda: os.getenv("LATEX_INSTITUTION", ""))
    # Clean up auxiliary files after compilation
    cleanup_aux: bool = field(default_factory=lambda: get_env_bool("LATEX_CLEANUP_AUX", True))


@dataclass
class Settings:
    """
    Main settings class that combines all configuration.
    
    Usage:
        from config.settings import settings
        
        print(settings.llm.provider)
        print(settings.database.path)
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    literature: LiteratureConfig = field(default_factory=LiteratureConfig)
    grobid: GROBIDConfig = field(default_factory=GROBIDConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    router: ModelRouterConfig = field(default_factory=ModelRouterConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    peer_review: PeerReviewConfig = field(default_factory=PeerReviewConfig)
    figure_verification: FigureVerificationConfig = field(default_factory=FigureVerificationConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    domain_skills: DomainSkillsConfig = field(default_factory=DomainSkillsConfig)
    latex: LatexConfig = field(default_factory=LatexConfig)
    
    def to_dict(self) -> dict:
        """Convert settings to dictionary."""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "database": {"path": self.database.path},
            "output": {
                "output_dir": self.output.output_dir,
                "save_notebooks": self.output.save_notebooks,
            },
            "execution": {
                "max_cycles": self.execution.max_cycles,
                "tasks_per_cycle": self.execution.tasks_per_cycle,
                "max_retries": self.execution.max_retries,
                "timeout_seconds": self.execution.timeout_seconds,
            },
            "docker": {
                "image": self.docker.image,
                "memory_limit": self.docker.memory_limit,
                "cpu_limit": self.docker.cpu_limit,
            },
            "literature": {
                "max_papers_per_search": self.literature.max_papers_per_search,
                "max_papers_to_read": self.literature.max_papers_to_read,
            },
            "rag": {
                "chunk_size": self.rag.chunk_size,
                "chunk_overlap": self.rag.chunk_overlap,
                "top_k": self.rag.top_k,
            },
            "router": {
                "enabled": self.router.enabled,
                "strong": f"{self.router.strong_provider}/{self.router.strong_model}",
                "fast": f"{self.router.fast_provider}/{self.router.fast_model}",
                "code": f"{self.router.code_provider}/{self.router.code_model}",
                "local": f"{self.router.local_provider}/{self.router.local_model}",
            }
        }
    
    def print_config(self) -> None:
        """Print current configuration (hiding sensitive values)."""
        print("=" * 60)
        print("INQUIRO CONFIGURATION")
        print("=" * 60)
        print(f"\nLLM:")
        print(f"  Provider: {self.llm.provider}")
        print(f"  Model: {self.llm.model}")
        print(f"  Temperature: {self.llm.temperature}")
        print(f"  Max Tokens: {self.llm.max_tokens}")
        print(f"  OpenAI API Key: {'✓ Set' if self.llm.openai_api_key else '✗ Not set'}")
        print(f"  Anthropic API Key: {'✓ Set' if self.llm.anthropic_api_key else '✗ Not set'}")
        print(f"  Google API Key: {'✓ Set' if self.llm.google_api_key else '✗ Not set'}")
        print(f"  Ollama URL: {self.llm.ollama_base_url}")
        print(f"\nRouter:")
        print(f"  Enabled: {self.router.enabled}")
        print(f"  Strong: {self.router.strong_provider}/{self.router.strong_model}")
        print(f"  Fast: {self.router.fast_provider}/{self.router.fast_model}")
        print(f"  Code: {self.router.code_provider}/{self.router.code_model}")
        print(f"  Local: {self.router.local_provider}/{self.router.local_model}")
        print(f"\nExecution:")
        print(f"  Max Cycles: {self.execution.max_cycles}")
        print(f"  Tasks per Cycle: {self.execution.tasks_per_cycle}")
        print("=" * 60)


# =============================================================================
# GLOBAL SETTINGS INSTANCE
# =============================================================================

settings = Settings()
