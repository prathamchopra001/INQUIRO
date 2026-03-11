"""
Inquiro - The main research engine.

This is the top-level class that ties everything together:
  - Initializes all components
  - Runs the research cycle loop
  - Coordinates agents via ThreadPoolExecutor
  - Triggers report generation when done

Usage:
    inquiro = Inquiro(
        objective="Identify drivers of solar cell efficiency",
        data_path="./data/solar_data.csv",
        max_cycles=10
    )
    inquiro.run()
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import settings
from src.utils.llm_client import LLMClient
from src.utils.shared_embeddings import pre_warm_embeddings
from src.world_model.world_model import WorldModel
from src.world_model.models import Task, TaskType, TaskStatus
from src.agents.orchestrator import OrchestratorAgent
from src.orchestration.cycle_phase_manager import CyclePhaseManager
from src.agents.data_analysis import DataAnalysisAgent
from src.agents.literature import LiteratureSearchAgent
from src.execution.native_executor import get_executor, is_docker_available
from src.execution.notebook_manager import NotebookManager
from src.literature.search import SemanticScholarClient
from src.literature.pdf_parser import PDFParser
from src.literature.rag import RAGSystem
from src.reports.generator import ReportGenerator
from src.validation.paper_reviewer import PaperReviewer, PeerReviewResult
from src.validation.figure_verifier import FigureVerifier, FigureVerificationReport
from src.reports.reproducibility import ReproducibilityPackageGenerator
from src.skills.domain_skill_injector import DomainSkillInjector, get_domain_skill_injector
from src.novelty.finding_deduplicator import FindingDeduplicator
from src.core.synthetic_data import SyntheticDatasetGenerator
from src.core.dataset_search import DatasetSearcher
from src.compression.context_compressor import ContextCompressor
from src.tracking.stage_tracker import StageTracker
from enum import Enum

logger = logging.getLogger(__name__)


class RunMode(Enum):
    """Controls which parts of the research pipeline are active."""
    LITERATURE = "literature"   # Literature search only (skip dataset search/synthetic)
    DATA = "data"               # Data analysis only (skip literature tasks)
    FULL = "full"               # Both agents active (default behavior)
    AUTO = "auto"               # LLM analyzes objective and recommends a mode


class Inquiro:
    """
    The autonomous scientific research engine.

    Runs iterative research cycles until the objective is met
    or the maximum cycle limit is reached.

    Each cycle:
      1. Orchestrator reads world model → generates tasks
      2. Tasks execute in parallel (ThreadPoolExecutor)
      3. Findings saved to world model
      4. Orchestrator checks completion
      5. Repeat or generate report
    """

    def __init__(
        self,
        objective: str,
        data_path: str = None,
        max_cycles: int = 10,
        num_tasks_per_cycle: int = 5,
        max_workers: int = 3,
        db_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        cycle_phases: list = None,
        seed_papers_dir: Optional[str] = None,
        mode: str = None,
        research_strategy: str = "standard",
        use_adaptive_decomposition: bool = False,
        resume_from: Optional[str] = None,
        generate_latex: bool = False,
        latex_template: str = "plain",
    ):
        """
        Initialize Inquiro and all its components.

        Args:
            objective:           The research question to investigate
            data_path:           Path to dataset CSV (None for literature-only mode)
            max_cycles:          Hard limit on research cycles (default 10)
            num_tasks_per_cycle: Tasks generated per cycle (default 5)
            max_workers:         Max parallel threads (default 3)
            db_path:             Path to world model database (auto-generated if None)
            output_dir:          Where to save reports (defaults to ./outputs)
            seed_papers_dir:     Path to folder of PDFs to pre-load into RAG
                                 (None = no seed papers, uses SEED_PAPERS_DIR env var as fallback)
            mode:                Research mode — "auto", "literature", "data", or "full"
                                 (None = use RUN_MODE env var, defaults to "auto")
            research_strategy:   Research strategy — "standard" (cycle-based), 
                                 "question_driven" (deep search per question), or "hybrid"
            use_adaptive_decomposition: Whether to use hierarchical pillar-based decomposition
                                        that scales with objective complexity (default False)
            resume_from:         Previous run ID to resume from (e.g. "run_20260309_010941_...")
                                 Loads world model and RAG collection from previous run.
                                 Allows iterative refinement across multiple sessions.
        """
        self.objective = objective
        self.data_path = data_path
        self.seed_papers_dir = seed_papers_dir or settings.literature.seed_papers_dir
        self.mode = (mode or settings.execution.run_mode or "auto").lower()
        assert self.mode in ("auto", "literature", "data", "full"), \
            f"Invalid mode '{self.mode}'. Must be: auto, literature, data, full"
        self.research_strategy = research_strategy.lower()
        assert self.research_strategy in ("standard", "question_driven", "hybrid"), \
            f"Invalid research_strategy '{research_strategy}'. Must be: standard, question_driven, hybrid"
        self.use_adaptive_decomposition = use_adaptive_decomposition
        self.resume_from = resume_from
        self.generate_latex = generate_latex
        self.latex_template = latex_template
        self.has_dataset = data_path is not None and Path(data_path).exists()
        self.is_synthetic_data = False  # Set to True if we generate a dataset
        self.data_source_info = "User-provided dataset" if self.has_dataset else ""
        self.max_cycles = max_cycles
        self.num_tasks_per_cycle = num_tasks_per_cycle
        self.max_workers = max_workers
        self.current_cycle = 0

        # --- Set up run-scoped folders ---
        # Every run gets its own folder derived from timestamp + objective slug.
        # This keeps outputs, notebooks, figures, and PDFs for each run together
        # and prevents cross-contamination between different research projects.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_slug = self._make_collection_name(objective)[:30].rstrip("_")
        self.run_id = f"run_{timestamp}_{run_slug}"

        # Output tree: outputs/<run_id>/{notebooks, figures}
        base_output = Path(output_dir or "./outputs")
        self.run_output_dir = base_output / self.run_id
        self.run_notebooks_dir = self.run_output_dir / "notebooks"
        self.run_figures_dir = self.run_output_dir / "figures"
        for d in [self.run_output_dir, self.run_notebooks_dir, self.run_figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Data tree: data/<run_id>/pdf_cache
        self.run_data_dir = Path("./data") / self.run_id
        self.run_pdf_cache_dir = self.run_data_dir / "pdf_cache"
        self.run_pdf_cache_dir.mkdir(parents=True, exist_ok=True)

        # --- Handle resume_from (multi-run persistence) ---
        previous_db_path = None
        previous_chromadb_dir = None
        previous_cycle = 0
        
        if resume_from:
            logger.info(f"🔄 Resuming from previous run: {resume_from}")
            previous_data_dir = Path("./data") / resume_from
            previous_output_dir = Path(output_dir or "./outputs") / resume_from
            
            # Check for previous world model
            prev_db = previous_data_dir / "world_model.db"
            if prev_db.exists():
                previous_db_path = str(prev_db)
                logger.info(f"   Found previous world model: {previous_db_path}")
            else:
                logger.warning(f"   No world model found at {prev_db}")
            
            # Check for previous ChromaDB
            prev_chromadb = previous_data_dir / "chromadb"
            if prev_chromadb.exists():
                previous_chromadb_dir = str(prev_chromadb)
                logger.info(f"   Found previous ChromaDB: {previous_chromadb_dir}")
            else:
                logger.warning(f"   No ChromaDB found at {prev_chromadb}")
            
            # Copy previous PDF cache to new run
            prev_pdf_cache = previous_data_dir / "pdf_cache"
            if prev_pdf_cache.exists():
                import shutil
                for pdf_file in prev_pdf_cache.glob("*.pdf"):
                    dest = self.run_pdf_cache_dir / pdf_file.name
                    if not dest.exists():
                        shutil.copy2(pdf_file, dest)
                logger.info(f"   Copied PDF cache from previous run")
            
            # Determine starting cycle from previous run
            if previous_db_path:
                try:
                    from src.world_model.world_model import WorldModel as TempWM
                    temp_wm = TempWM(db_path=previous_db_path)
                    stats = temp_wm.get_statistics()
                    # Use max cycle from findings as starting point
                    findings = temp_wm.get_all_findings()
                    if findings:
                        previous_cycle = max(f.cycle for f in findings)
                    temp_wm.close()
                    logger.info(f"   Previous run completed {previous_cycle} cycles with {stats.get('findings', 0)} findings")
                except Exception as e:
                    logger.warning(f"   Could not read previous run stats: {e}")

        # Use previous db_path if resuming and no explicit path given
        if db_path is None:
            if previous_db_path:
                # Copy previous DB to new run location
                import shutil
                new_db_path = str(self.run_data_dir / "world_model.db")
                shutil.copy2(previous_db_path, new_db_path)
                # Also copy WAL files if they exist
                for wal_ext in ["-shm", "-wal"]:
                    wal_src = Path(previous_db_path + wal_ext)
                    if wal_src.exists():
                        shutil.copy2(wal_src, new_db_path + wal_ext)
                db_path = new_db_path
                logger.info(f"   Copied world model to: {db_path}")
            else:
                db_path = str(self.run_data_dir / "world_model.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Set starting cycle if resuming
        if previous_cycle > 0:
            self.current_cycle = previous_cycle
            logger.info(f"   Starting from cycle {self.current_cycle + 1}")

        # --- Initialize shared components ---
        logger.info("Initializing Inquiro components...")

        # Pre-warm the embedding model FIRST, before any agent that uses it.
        # This ensures SemanticMatcher (in QuestionManager) sees is_available()=True
        # at initialization time, not just at validation time.
        if pre_warm_embeddings():
            logger.info("🧠 Embedding model pre-loaded (semantic matching ready)")
        else:
            logger.warning("🧠 Embedding model unavailable (using keyword matching fallback)")

        self.llm = LLMClient()
        self.world_model = WorldModel(db_path=db_path)
        
        # --- Domain skill injection ---
        # Detect research domain and load specialized skill prompts
        self.domain_skill_injector = get_domain_skill_injector()
        self.domain = self.domain_skill_injector.set_objective(objective)
        if self.domain != "general":
            logger.info(f"🎯 Research domain: {self.domain}")

        # --- Initialize agents ---
        phase_manager = CyclePhaseManager(cycle_phases, has_dataset=self.has_dataset) if cycle_phases else None
        self.phase_manager = phase_manager  # keep reference for updating after synthetic gen
        self.orchestrator = OrchestratorAgent(
            llm_client=self.llm,
            world_model=self.world_model,
            cycle_phase_manager=phase_manager,
            has_dataset=self.has_dataset,
            use_adaptive_decomposition=self.use_adaptive_decomposition,
        )

        # --- Finding deduplicator (prevents storing the same result twice) ---
        self.finding_dedup = FindingDeduplicator(
            text_threshold=0.60,
            num_threshold=0.80,
        )

        # Derive a UNIQUE ChromaDB collection name per run.
        # Previous behavior shared collections across runs with the same objective,
        # but stale data caused chunks to silently fail to load.
        # Each run gets its own fresh collection.
        rag_collection = self._make_collection_name(objective + "_" + timestamp)
        logger.info(f"📚 RAG collection: '{rag_collection}'")
        
        # Handle ChromaDB persistence from previous run
        chromadb_dir = str(self.run_data_dir / "chromadb")
        if previous_chromadb_dir and Path(previous_chromadb_dir).exists():
            import shutil
            # Copy entire ChromaDB directory
            shutil.copytree(previous_chromadb_dir, chromadb_dir, dirs_exist_ok=True)
            logger.info(f"   Copied ChromaDB from previous run")
            # Use same collection name as previous run for continuity
            # Find the collection name from the previous run
            try:
                import chromadb
                temp_client = chromadb.PersistentClient(path=chromadb_dir)
                collections = temp_client.list_collections()
                if collections:
                    rag_collection = collections[0].name
                    logger.info(f"   Using previous collection: '{rag_collection}'")
            except Exception as e:
                logger.warning(f"   Could not detect previous collection name: {e}")

        # Use Docker if available, otherwise fall back to native Python execution
        _executor = get_executor(prefer_docker=True)
        _executor.warm_pool()  # D2: pre-warm containers (no-op for native)

        self.data_agent = DataAnalysisAgent(
            llm_client=self.llm,
            executor=_executor,
            notebook_manager=NotebookManager(
                output_dir=str(self.run_notebooks_dir)
            ),
        )

        self.literature_agent = LiteratureSearchAgent(
            llm_client=self.llm,
            search_client=SemanticScholarClient(),
            pdf_parser=PDFParser(
                cache_dir=str(self.run_pdf_cache_dir)
            ),
            # Use per-run ChromaDB directory to prevent index corruption from affecting other runs
            rag_system=RAGSystem(
                collection_name=rag_collection,
                persist_dir=chromadb_dir,
            ),
        )

        self.report_generator = ReportGenerator(
            llm_client=self.llm,
            world_model=self.world_model,
            output_dir=str(self.run_output_dir),
        )

        # Context compressor — reduces world model history to ~500 tokens
        self.compressor = ContextCompressor(llm_client=self.llm)

        # Stage tracker — structured JSONL logging
        self.tracker = StageTracker(
            run_id=self.run_id,
            output_dir=str(self.run_output_dir),
        )
        logger.info(f"📊 Stage tracking: {self.tracker.get_output_path()}")

        logger.info("✅ All components initialized")

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _make_collection_name(objective: str) -> str:
        """
        Derive a stable, ChromaDB-safe collection name from the research objective.

        ChromaDB rules: 3-63 chars, alphanumeric + underscores/hyphens,
        must start and end with alphanumeric.

        Example:
            "Identify key metabolic features in cancer samples"
            → "identify_key_metabolic_features_in_canc"
        """
        # Lowercase and replace all non-alphanumeric chars with underscores
        name = objective.lower()
        name = re.sub(r'[^a-z0-9]+', '_', name)
        # Strip leading/trailing underscores
        name = name.strip('_')
        # Truncate to 40 chars, then strip trailing underscores again
        name = name[:40].rstrip('_')
        # ChromaDB requires at least 3 chars
        if len(name) < 3:
            name = f"inquiro_{name}"
        return name

    # =========================================================================
    # SEED PAPERS
    # =========================================================================

    def _load_seed_papers(self) -> int:
        """
        Pre-load user-provided PDFs into the RAG system before cycle 1.

        Scans self.seed_papers_dir for *.pdf files, extracts text,
        chunks it, and stores it in ChromaDB. This gives the literature
        agent a "head start" — it already knows about the key papers
        the researcher cares about.

        Returns:
            Total number of chunks loaded into RAG.
        """
        if not self.seed_papers_dir:
            return 0

        seed_dir = Path(self.seed_papers_dir)
        if not seed_dir.exists() or not seed_dir.is_dir():
            logger.warning(
                f"Seed papers directory not found: {self.seed_papers_dir}"
            )
            return 0

        # Support both flat and nested structures
        # Looks for PDFs in the directory and any subdirectories
        pdf_files = sorted(seed_dir.glob("*.pdf"))  # Direct PDFs
        pdf_files.extend(sorted(seed_dir.rglob("*/*.pdf")))  # PDFs in subdirectories
        
        if not pdf_files:
            logger.info(f"No PDF files found in seed directory: {seed_dir}")
            return 0

        logger.info(f"\n{'='*60}")
        logger.info(f"\U0001f4d6 LOADING SEED PAPERS ({len(pdf_files)} PDFs)")
        logger.info(f"   Source: {seed_dir}")
        logger.info(f"{'='*60}")

        from src.literature.models import Paper
        import hashlib

        total_chunks = 0
        loaded_count = 0

        for pdf_file in pdf_files:
            # Derive a stable paper_id from the file path
            file_hash = hashlib.md5(str(pdf_file).encode()).hexdigest()[:12]
            paper_id = f"seed_{file_hash}"

            # Use filename (without extension) as the title
            title = pdf_file.stem.replace("_", " ").replace("-", " ")

            # Create a minimal Paper object
            paper = Paper(
                paper_id=paper_id,
                title=title,
                abstract=f"Seed paper loaded from: {pdf_file.name}",
                is_open_access=True,
            )

            try:
                chunks = self.literature_agent.pdf_parser.process_local_pdf(
                    pdf_path=str(pdf_file),
                    paper=paper,
                )

                if chunks:
                    added = self.literature_agent.rag.add_chunks(chunks)
                    total_chunks += added
                    loaded_count += 1
                    logger.info(
                        f"  \u2713 {pdf_file.name}: {len(chunks)} chunks loaded"
                    )
                else:
                    logger.warning(
                        f"  \u2717 {pdf_file.name}: no text extracted (scanned PDF?)"
                    )
            except Exception as e:
                logger.error(f"  \u2717 {pdf_file.name}: failed \u2014 {e}")
                continue

        logger.info(
            f"\n\U0001f4da Seed papers loaded: {loaded_count}/{len(pdf_files)} papers, "
            f"{total_chunks} total chunks in RAG"
        )

        # Log RAG stats so we can verify the chunks are actually there
        try:
            stats = self.literature_agent.rag.get_collection_stats()
            logger.info(
                f"   RAG collection now has {stats['total_chunks']} total chunks"
            )
        except Exception:
            pass

        return total_chunks
    
    def _extract_dataset_context_from_seeds(self) -> str:
        """
        Query the seed papers in RAG for mentions of datasets, 
        data sources, and empirical data.

        Returns a text block that helps the dataset searcher 
        generate much more targeted queries.
        """
        rag_stats = self.literature_agent.rag.get_collection_stats()
        if rag_stats["total_chunks"] == 0:
            return ""

        # Ask RAG specifically about data sources
        queries = [
            "datasets used data sources empirical data",
            "data collection methodology sample statistics",
        ]

        context_parts = []
        for q in queries:
            results = self.literature_agent.rag.query(q, top_k=5)
            for text, metadata, distance in results:
                if distance < 0.8:  # Only reasonably relevant chunks
                    source = metadata.get("paper_title", "unknown")
                    context_parts.append(f"[From: {source}]\n{text[:500]}")

        if not context_parts:
            return ""

        context = "\n\n".join(context_parts[:6])  # Cap at 6 chunks
        logger.info(f"📋 Extracted dataset context from seed papers ({len(context_parts)} relevant chunks)")
        return context

    # =========================================================================
    # OBJECTIVE ANALYSIS (for auto mode)
    # =========================================================================

    # =========================================================================
    # OBJECTIVE ANALYSIS (for auto mode)
    # =========================================================================

    def _analyze_objective(self) -> str:
        """
        Use the LLM to analyze the research objective and suggest a run mode.

        For auto mode: analyzes the objective, shows the recommendation
        to the user, and asks for confirmation before proceeding.

        Returns:
            The confirmed run mode: "literature", "data", or "full"
        """
        from config.prompts.objective_analysis import OBJECTIVE_ANALYSIS_PROMPT

        # Build status strings for context
        if self.has_dataset:
            dataset_status = f"User provided a dataset: {self.data_path}"
        else:
            dataset_status = "No dataset provided."

        if self.seed_papers_dir and Path(self.seed_papers_dir).exists():
            pdf_count = len(list(Path(self.seed_papers_dir).glob("*.pdf")))
            seed_status = f"User provided {pdf_count} seed paper(s) in {self.seed_papers_dir}"
        else:
            seed_status = "No seed papers provided."

        prompt = OBJECTIVE_ANALYSIS_PROMPT.format(
            objective=self.objective,
            dataset_status=dataset_status,
            seed_status=seed_status,
        )

        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="objective_analysis",
                system="You are a research planning assistant. Respond only with JSON.",
            )
            text = response.content.strip()

            # Parse JSON
            import json
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                suggested_mode = result.get("mode", "full").lower()
                reasoning = result.get("reasoning", "No reasoning provided.")
            else:
                logger.warning("Could not parse objective analysis response — defaulting to 'full'")
                suggested_mode = "full"
                reasoning = "Failed to analyze objective — using full mode as safe default."
        except Exception as e:
            logger.error(f"Objective analysis failed: {e} — defaulting to 'full'")
            suggested_mode = "full"
            reasoning = f"Analysis error: {e} — using full mode as safe default."

        # Validate mode
        if suggested_mode not in ("literature", "data", "full"):
            logger.warning(f"Invalid suggested mode '{suggested_mode}' — defaulting to 'full'")
            suggested_mode = "full"

        # Show recommendation and ask for confirmation
        logger.info(f"\n{'='*60}")
        logger.info(f"\U0001f9e0 OBJECTIVE ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"  Suggested mode: {suggested_mode.upper()}")
        logger.info(f"  Reasoning: {reasoning}")
        logger.info(f"{'='*60}")

        print(f"\n\U0001f9e0 INQUIRO suggests running in [{suggested_mode.upper()}] mode")
        print(f"   Reason: {reasoning}")
        print(f"\n   Available modes:")
        print(f"     [L] Literature only — paper search and synthesis")
        print(f"     [D] Data only — dataset analysis, no paper search")
        print(f"     [F] Full — both literature + data analysis")
        print(f"     [Enter] Accept suggestion ({suggested_mode})\n")

        try:
            choice = input("   Your choice [L/D/F/Enter]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = ""

        mode_map = {"l": "literature", "d": "data", "f": "full", "": suggested_mode}
        confirmed_mode = mode_map.get(choice, suggested_mode)

        logger.info(f"  \u2713 Confirmed mode: {confirmed_mode.upper()}")
        print(f"\n   \u2713 Running in {confirmed_mode.upper()} mode\n")

        return confirmed_mode

    # =========================================================================
    # PUBLIC ENTRY POINT
    # =========================================================================

    def run(self) -> str:
        """
        Run the full research process.

        Cycles until completion or max_cycles is reached,
        then generates and returns the path to the final report.

        Returns:
            Path to the generated report file
        """
        logger.info("=" * 60)
        logger.info("🔬 INQUIRO RESEARCH ENGINE STARTING")
        logger.info(f"   Objective: {self.objective[:80]}...")
        if self.has_dataset:
            logger.info(f"   Dataset:   {self.data_path}")
        else:
            logger.info("   Dataset:   None (literature-only mode)")
        logger.info(f"   Max Cycles: {self.max_cycles}")
        logger.info(f"   Run ID:    {self.run_id}")
        logger.info(f"   Outputs:   {self.run_output_dir}")
        logger.info(f"   Data:      {self.run_data_dir}")
        if self.seed_papers_dir:
            logger.info(f"   Seeds:     {self.seed_papers_dir}")
        logger.info(f"   Mode:      {self.mode}")
        logger.info(f"   Strategy:  {self.research_strategy}")
        logger.info("=" * 60)

        # ── Mode resolution ────────────────────────────────────────────
        # If mode is "auto", analyze the objective and ask user to confirm.
        if self.mode == "auto":
            self.mode = self._analyze_objective()

        logger.info(f"\n\U0001f3af Run mode: {self.mode.upper()}")

        # ── Seed papers pipeline ─────────────────────────────────────────
        # Load seed papers FIRST so dataset search can benefit from
        # knowing what data sources the papers actually used.
        if self.mode in ("literature", "full"):
            seed_chunks = self._load_seed_papers()
            if seed_chunks:
                logger.info(f"\n\U0001f4d6 Seed knowledge: {seed_chunks} chunks pre-loaded into RAG")
                logger.info("   Literature agent will build on this foundation.\n")
            
            
        # ── Data acquisition pipeline ─────────────────────────────────────
        # Only search for datasets if mode requires data AND no dataset provided.
        if self.mode in ("data", "full") and not self.has_dataset:
            logger.info("")

            # Tier 1: Search for a public dataset
            logger.info("🔍 No dataset provided — searching for public datasets...")
            searcher = DatasetSearcher(llm_client=self.llm)
            try:
                search_result = searcher.search(
                    objective=self.objective,
                    output_dir=str(self.run_data_dir),
                )
            finally:
                searcher.close()  # Prevent ResourceWarning

            if search_result.success:
                self.data_path = search_result.path
                self.has_dataset = True
                self.is_synthetic_data = False
                self.orchestrator.has_dataset = True
                if self.phase_manager:
                    self.phase_manager.has_dataset = True
                logger.info(
                    f"  ✅ Public dataset found: {search_result.dataset_name} "
                    f"(relevance={search_result.relevance_score:.2f}, "
                    f"source={search_result.source})"
                )
            else:
                # Tier 2: Generate synthetic dataset
                logger.info("  No suitable public dataset found — generating synthetic...")
                synth_gen = SyntheticDatasetGenerator(
                    llm_client=self.llm,
                    executor=self.data_agent.executor if hasattr(self, 'data_agent') else get_executor(prefer_docker=True),
                )
                synth_result = synth_gen.generate(
                    objective=self.objective,
                    output_dir=str(self.run_data_dir),
                )
                if synth_result.success:
                    self.data_path = synth_result.path
                    self.has_dataset = True
                    self.is_synthetic_data = True
                    self.orchestrator.has_dataset = True
                    if self.phase_manager:
                        self.phase_manager.has_dataset = True
                    logger.info(
                        f"  ✅ Synthetic dataset ready: {synth_result.path} "
                        f"({synth_result.row_count} rows × {synth_result.column_count} cols)"
                    )
                    logger.info(f"  📝 Description: {synth_result.description}")
                else:
                    logger.warning(
                        "  ⚠️ Both dataset search and synthetic generation failed — "
                        "falling back to literature-only mode"
                    )

            logger.info("")

        elif self.mode == "literature" and not self.has_dataset:
            logger.info("\n\U0001f4da Literature-only mode \u2014 skipping dataset search entirely.\n")

        start_time = time.time()

        # Initialize research planning for multi-area objectives
        research_plan = self.orchestrator.initialize_research_plan(self.objective)
        if research_plan:
            logger.info(f"\n📋 Research plan created with {research_plan.get('total_areas', 0)} areas")
            logger.info("   Coverage will be tracked across cycles.\n")

        # Initialize research questions (question-driven research)
        # This decomposes the objective into specific, trackable questions
        questions = self.orchestrator.initialize_questions(
            objective=self.objective,
            domain_context=""  # Could integrate with DomainAnchorExtractor here
        )
        if questions:
            logger.info(f"\n❓ Research decomposed into {len(questions)} questions")
            logger.info("   Progress will be tracked by question completion.\n")

        # ══════════════════════════════════════════════════════════════════
        # RESEARCH STRATEGY BRANCH
        # Choose between standard cycle-based or question-driven deep research
        # ══════════════════════════════════════════════════════════════════
        
        if self.research_strategy == "question_driven" and questions:
            # Question-driven deep research mode
            logger.info("\n🎯 Using QUESTION-DRIVEN research strategy")
            logger.info("   Deep search per question with citation chains\n")
            self._run_question_driven_research(questions, start_time)
            
        elif self.research_strategy == "hybrid" and questions:
            # Hybrid: Question-driven first, then standard cycles for gaps
            logger.info("\n🔀 Using HYBRID research strategy")
            logger.info("   Question-driven first, then exploration cycles\n")
            self._run_question_driven_research(questions, start_time)
            
            # Check if gaps remain, run standard cycles
            gaps = self.orchestrator.get_research_gaps(min_unanswered=1)
            if gaps.get("has_gaps", False) and self.current_cycle < self.max_cycles:
                logger.info("\n🔄 Switching to standard cycles for remaining gaps...")
                self._run_standard_cycles(start_time)
        else:
            # Standard cycle-based research
            self._run_standard_cycles(start_time)

        # ══════════════════════════════════════════════════════════════════
        # REPORT VALIDATION LOOP
        # After generating a report, check if significant gaps remain.
        # If so, and we still have cycles left, continue research.
        # ══════════════════════════════════════════════════════════════════
        
        report_path = None
        validation_cycles = 0
        max_validation_cycles = 2  # Prevent infinite loops
        
        while validation_cycles < max_validation_cycles:
            validation_cycles += 1
            
            # Generate report
            elapsed = time.time() - start_time
            logger.info(f"\n📝 Generating report... (total runtime: {elapsed:.1f}s)")

            report_result = self.report_generator.generate_report(
                objective=self.objective,
                cycles_completed=self.current_cycle,
                is_synthetic_data=self.is_synthetic_data,
                generate_latex=self.generate_latex,
                latex_template=self.latex_template,
            )
            
            # Handle both dict (with LaTeX) and string (markdown only) return values
            if isinstance(report_result, dict):
                report_path = report_result.get("markdown") or report_result.get("pdf") or report_result.get("latex")
                if report_result.get("pdf"):
                    logger.info(f"📄 PDF compiled: {report_result['pdf']}")
                elif report_result.get("latex"):
                    logger.info(f"📝 LaTeX source: {report_result['latex']} (PDF compilation unavailable)")
            else:
                report_path = report_result
            
            # Check for research gaps
            gaps = self.orchestrator.get_research_gaps(min_unanswered=2)
            
            if not gaps.get("has_gaps", False):
                # No significant gaps - we're done!
                logger.info("\n✅ All research questions adequately addressed.")
                break
            
            # We have gaps - can we continue?
            remaining_cycles = self.max_cycles - self.current_cycle
            
            if remaining_cycles <= 0:
                # No cycles left - finalize with gaps noted
                logger.info(
                    f"\n⚠️ Research gaps remain but max cycles reached: "
                    f"{gaps.get('gap_summary', 'See report for details')}"
                )
                break
            
            # We have gaps AND cycles remaining - continue research!
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 VALIDATION LOOP: {gaps.get('gap_count', 0)} questions still open")
            logger.info(f"   {gaps.get('gap_summary', '')}")
            logger.info(f"   Continuing research ({remaining_cycles} cycles remaining)...")
            logger.info(f"{'='*60}")
            
            # Run additional cycles to address gaps
            # Limit to 2 extra cycles per validation loop
            extra_cycles = min(2, remaining_cycles)
            
            with self.tracker.track("validation_loop", iteration=validation_cycles):
                for _ in range(extra_cycles):
                    if self.current_cycle >= self.max_cycles:
                        break
                    
                    self.current_cycle += 1
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"🔄 VALIDATION CYCLE {self.current_cycle} / {self.max_cycles}")
                    logger.info(f"{'='*60}")
                    
                    # Execute cycle
                    with self.tracker.track("cycle", cycle=self.current_cycle):
                        self._execute_cycle(max_cycles=self.max_cycles)
                    
                    # Validate questions
                    validation = self.orchestrator.validate_research_questions(
                        cycle=self.current_cycle
                    )
                    if validation.get("overall_progress"):
                        progress = validation["overall_progress"]
                        logger.info(
                            f"\n❓ Question validation: "
                            f"{progress.get('answered', 0)} answered "
                            f"({progress.get('completion_percentage', 0):.0f}% complete)"
                        )
            
            # After extra cycles, loop back to generate updated report
            # and check gaps again
        
        logger.info(f"\n🎉 Done! Report saved to: {report_path}")
        
        # Log final question status
        final_gaps = self.orchestrator.get_research_gaps()
        if final_gaps.get("has_gaps"):
            logger.info(
                f"\n📊 Final status: {final_gaps.get('gap_count', 0)} questions remain open"
            )
            for gap in final_gaps.get("high_priority_gaps", [])[:3]:
                logger.info(f"   🔴 {gap.get('question', '')[:70]}...")
        else:
            logger.info("\n🌟 All research questions have been addressed!")
        
        # ══════════════════════════════════════════════════════════════════
        # VLM FIGURE VERIFICATION
        # Verify generated figures match captions and findings
        # ══════════════════════════════════════════════════════════════════
        figure_verification = None
        if settings.figure_verification.enabled:
            try:
                figures_dir = self.run_output_dir / "figures"
                if not figures_dir.exists():
                    figures_dir = self.run_data_dir / "figures"
                
                if figures_dir.exists() and any(figures_dir.glob("*.png")):
                    logger.info("\n🖼️ Running VLM figure verification...")
                    
                    verifier = FigureVerifier(
                        self.llm,
                        max_figures=settings.figure_verification.max_figures,
                        min_score_threshold=settings.figure_verification.min_score,
                    )
                    
                    # Get findings for context
                    findings = [
                        item["finding"].to_dict() if hasattr(item["finding"], "to_dict") else {}
                        for item in self.world_model.get_top_findings(n=10)
                    ]
                    
                    figure_verification = verifier.verify_figures(
                        figures_dir=str(figures_dir),
                        findings=findings,
                    )
                    
                    # Log results
                    if figure_verification.total_figures > 0:
                        logger.info(f"\n{'='*50}")
                        logger.info("🖼️ FIGURE VERIFICATION RESULTS")
                        logger.info(f"{'='*50}")
                        logger.info(f"   Total Figures: {figure_verification.total_figures}")
                        logger.info(f"   Passed: {figure_verification.figures_passed}")
                        logger.info(f"   Flagged: {figure_verification.figures_flagged}")
                        logger.info(f"   Average Score: {figure_verification.average_score:.1f}/4.0")
                        
                        if figure_verification.figures_flagged > 0:
                            logger.info("\n   ⚠️ Flagged figures:")
                            for result in figure_verification.results:
                                if not result.passes_verification:
                                    logger.info(f"      • {result.figure_name}: {result.overall_score:.1f}/4")
                        
                        logger.info(f"{'='*50}")
                        
                        # Save verification report
                        verification_path = self.run_output_dir / "figure_verification.json"
                        verification_path.write_text(
                            json.dumps(figure_verification.to_dict(), indent=2),
                            encoding="utf-8"
                        )
                        logger.info(f"\n💾 Figure verification saved to: {verification_path}")
                else:
                    logger.debug("No figures found to verify")
                    
            except Exception as e:
                logger.warning(f"Figure verification failed: {e}")
        
        # Store figure verification in instance for external access
        self.figure_verification = figure_verification
        
        # ══════════════════════════════════════════════════════════════════
        # AUTOMATED PEER REVIEW
        # Run academic-style peer review on the generated report
        # ══════════════════════════════════════════════════════════════════
        peer_review = None
        if report_path and settings.peer_review.enabled:
            try:
                logger.info("\n📋 Running automated peer review...")
                reviewer = PaperReviewer(self.llm)
                
                # Read report content
                report_content = Path(report_path).read_text(encoding="utf-8")
                
                # Get findings for context
                findings = [
                    item["finding"].to_dict() if hasattr(item["finding"], "to_dict") else {}
                    for item in self.world_model.get_top_findings(n=20)
                ]
                
                # Run review
                peer_review = reviewer.review_report(
                    report_text=report_content,
                    objective=self.objective,
                    findings=findings,
                )
                
                # Log results
                logger.info(f"\n{'='*60}")
                logger.info("📊 PEER REVIEW RESULTS")
                logger.info(f"{'='*60}")
                logger.info(f"   Overall Score: {peer_review.overall_score:.1f}/4.0")
                logger.info(f"   Recommendation: {peer_review.overall_recommendation}")
                logger.info(f"   Confidence: {peer_review.reviewer_confidence:.0%}")
                logger.info("")
                
                for dim in peer_review.dimensions:
                    stars = "★" * dim.score + "☆" * (4 - dim.score)
                    logger.info(f"   {dim.name:15} {stars}  ({dim.score}/4)")
                
                if peer_review.key_contributions:
                    logger.info("\n   ✅ Key Contributions:")
                    for contrib in peer_review.key_contributions[:3]:
                        logger.info(f"      • {contrib[:60]}...")
                
                if peer_review.major_concerns:
                    logger.info("\n   ⚠️ Areas for Improvement:")
                    for concern in peer_review.major_concerns[:3]:
                        logger.info(f"      • {concern[:60]}...")
                
                logger.info(f"{'='*60}")
                
                # Save review to file
                review_path = Path(report_path).parent / "peer_review.json"
                import json
                review_path.write_text(
                    json.dumps(peer_review.to_dict(), indent=2),
                    encoding="utf-8"
                )
                logger.info(f"\n💾 Peer review saved to: {review_path}")
                
            except Exception as e:
                logger.warning(f"Peer review failed: {e}")
        
        # Store peer review in instance for external access
        self.peer_review = peer_review
        
        # ══════════════════════════════════════════════════════════════════
        # REPRODUCIBILITY PACKAGE
        # Generate everything needed to reproduce this run
        # ══════════════════════════════════════════════════════════════════
        reproducibility_path = None
        if settings.reproducibility.enabled:
            try:
                logger.info("\n📦 Generating reproducibility package...")
                
                # Calculate runtime
                total_runtime = time.time() - start_time
                
                # Get stats
                stats = self.world_model.get_statistics()
                
                # Collect data paths
                data_paths = []
                if self.data_path:
                    data_paths.append(self.data_path)
                
                # Generate package
                repro_gen = ReproducibilityPackageGenerator(str(self.run_output_dir))
                reproducibility_path = repro_gen.generate(
                    objective=self.objective,
                    config=settings,
                    data_paths=data_paths,
                    run_id=self.run_id,
                    cycles_completed=self.current_cycle,
                    findings_count=stats.get("total_findings", 0),
                    runtime_seconds=total_runtime,
                )
                
                logger.info(f"  ✅ Reproducibility package saved to: {reproducibility_path}")
                
            except Exception as e:
                logger.warning(f"Reproducibility package generation failed: {e}")
        
        # Store reproducibility path in instance
        self.reproducibility_path = reproducibility_path
        
        # Clean up HTTP clients to avoid ResourceWarnings
        self._cleanup()
        
        return report_path
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def _cleanup(self) -> None:
        """
        Close all HTTP clients and resources to prevent ResourceWarnings.
        
        Called automatically at the end of run(), but can also be called
        manually if Inquiro is used without run().
        """
        logger.debug("Cleaning up HTTP clients...")
        
        # Close literature agent's search clients
        try:
            if hasattr(self.literature_agent, 'search_client'):
                self.literature_agent.search_client.close()
            if hasattr(self.literature_agent, 'openalex_client'):
                self.literature_agent.openalex_client.close()
            if hasattr(self.literature_agent, 'crossref_client'):
                self.literature_agent.crossref_client.close()
            if hasattr(self.literature_agent, 'core_client'):
                self.literature_agent.core_client.close()
            if hasattr(self.literature_agent, 'dimensions_client'):
                self.literature_agent.dimensions_client.close()
            # ArXiv and PubMed use per-request clients, no cleanup needed
        except Exception as e:
            logger.debug(f"Error closing literature clients: {e}")
        
        # Drain Docker container pool
        try:
            if hasattr(self.data_agent, 'executor'):
                self.data_agent.executor.drain_pool()
        except Exception as e:
            logger.debug(f"Error draining Docker pool: {e}")
        
        logger.debug("Cleanup complete")

    # =========================================================================
    # CYCLE EXECUTION
    # =========================================================================
    def _build_dataset_summary(self) -> str:
        """Build a compact summary of the dataset to anchor literature searches."""
        if not self.has_dataset:
            return "(No dataset provided — this is a literature-only research run.)"
        try:
            import pandas as pd
            df = pd.read_csv(self.data_path)
            
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
            
            lines = [
                f"Dataset: {self.data_path.split('/')[-1]}",
                f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
                f"Numeric variables: {', '.join(numeric_cols)}",
                f"Categorical variables: {', '.join(categorical_cols)}",
            ]
            
            # Group distribution if a 'group' column exists
            if 'group' in df.columns:
                group_counts = df['group'].value_counts().to_dict()
                lines.append(f"Groups: {group_counts}")
            
            lines.append(
                "IMPORTANT: Literature searches must stay relevant to these "
                "specific variables. Do not follow tangential citation chains."
            )
            
            return "\n".join(lines)
        
        except Exception as e:
            logger.warning(f"Could not build dataset summary: {e}")
            return ""

    def _reframe_as_literature_task(self, data_description: str) -> str:
        """
        Convert a data analysis task description into a literature search task.

        Instead of just prepending "Search for papers about...", this extracts
        the core concepts and reframes as a proper literature question.
        """
        desc_lower = data_description.lower()

        # Map common data-task patterns to meaningful literature searches
        reframe_rules = [
            # Implementation tasks → methodology searches
            (["implement", "build", "create", "construct", "code"],
             "Search for papers describing implementations and methodologies for"),
            # Comparison/benchmark tasks → performance comparison searches
            (["compare", "benchmark", "baseline", "rmse", "mae", "forecast"],
             "Search for papers comparing performance and benchmarks of"),
            # Correlation/regression tasks → relationship studies
            (["correlat", "regression", "relationship", "association"],
             "Search for studies examining the relationship between"),
            # Distribution/descriptive tasks → empirical pattern studies
            (["distribution", "descriptive", "histogram", "summary statistic"],
             "Search for empirical studies characterizing the distribution of"),
            # Visualization tasks → review/survey searches
            (["visualiz", "plot", "chart", "heatmap"],
             "Search for survey papers and reviews covering"),
        ]

        prefix = "Search for papers on methods and findings related to"
        for keywords, reframe_prefix in reframe_rules:
            if any(kw in desc_lower for kw in keywords):
                prefix = reframe_prefix
                break

        # Extract just the core topic — strip implementation noise
        import re
        cleaned = data_description
        # Remove parameter assignments (alpha=0.1, gamma=0.95, etc.)
        cleaned = re.sub(r'\b\w+\s*=\s*[\d.\[\]]+', '', cleaned)
        # Remove file paths and code patterns
        cleaned = re.sub(r'[/\\]\S+', '', cleaned)
        cleaned = re.sub(r'df\[.*?\]', '', cleaned)
        # Remove lone numbers and number ranges
        cleaned = re.sub(r'\b\d+[Q.]\d+[-–]\d+[Q.]\d+\b', '', cleaned)
        # Collapse whitespace and dangling punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'[,;]\s*[,;]', ',', cleaned)
        cleaned = cleaned.strip(' ,;.')

        # Truncate to core concept
        if len(cleaned) > 120:
            cleaned = cleaned[:120].rsplit(' ', 1)[0]

        return f"{prefix}: {cleaned}"

    # =========================================================================
    # RESEARCH STRATEGY METHODS
    # =========================================================================
    
    def _run_standard_cycles(self, start_time: float) -> None:
        """
        Run standard cycle-based research.
        
        This is the original INQUIRO workflow:
        - Generate tasks → Execute → Save findings → Check completion → Repeat
        """
        with self.tracker.track("research_loop", max_cycles=self.max_cycles):
            while self.current_cycle < self.max_cycles:
                self.current_cycle += 1

                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 CYCLE {self.current_cycle} / {self.max_cycles}")
                logger.info(f"{'='*60}")

                # Execute one full research cycle
                with self.tracker.track("cycle", cycle=self.current_cycle):
                    self._execute_cycle(max_cycles=self.max_cycles)

                # Validate research questions after each cycle
                validation = self.orchestrator.validate_research_questions(
                    cycle=self.current_cycle
                )
                if validation.get("overall_progress"):
                    progress = validation["overall_progress"]
                    logger.info(
                        f"\n❓ Question validation: "
                        f"{progress.get('answered', 0)}/{progress.get('answered', 0) + progress.get('partial', 0) + progress.get('unanswered', 0)} answered "
                        f"({progress.get('completion_percentage', 0):.0f}% complete)"
                    )

                # Check if we have enough to stop
                is_complete = self.orchestrator.check_completion(
                    objective=self.objective,
                    cycles_completed=self.current_cycle,
                    max_cycles=self.max_cycles,
                )

                if is_complete:
                    logger.info(
                        f"\n✅ Research complete after {self.current_cycle} cycles!"
                    )
                    break
            else:
                logger.info(f"\n⏱️  Max cycles ({self.max_cycles}) reached.")
    
    def _run_question_driven_research(self, questions: list, start_time: float) -> None:
        """
        Run question-driven deep research.
        
        For each research question:
        1. Generate focused queries from question text
        2. Hybrid search (BM25 + dense embeddings)
        3. Rank papers by question relevance
        4. Follow citation chains for top papers
        5. Extract findings and link to question
        6. Store with question_id for persistence
        """
        from src.orchestration.question_research import QuestionDrivenResearch
        
        logger.info(f"\n🔬 Starting Question-Driven Research for {len(questions)} questions")
        
        # Initialize question-driven research orchestrator
        qdr = QuestionDrivenResearch(
            llm_client=self.llm,
            rag_system=self.lit_agent.rag if hasattr(self, 'lit_agent') else None,
            pdf_parser=self.lit_agent.pdf_parser if hasattr(self, 'lit_agent') else None,
            run_id=self.run_id,
        )
        
        # Run deep research for all questions
        results = qdr.run(
            questions=questions,
            objective=self.objective,
            max_papers_per_question=10,
            follow_citations=True,
            min_findings_per_question=2,
        )
        
        # Save findings to world model
        for q_id, q_result in results.get("questions", {}).items():
            if q_result.get("status") == "error":
                continue
            
            findings = q_result.get("findings", [])
            for finding_dict in findings:
                try:
                    self.world_model.add_finding(
                        claim=finding_dict.get("claim", ""),
                        confidence=finding_dict.get("confidence", 0.7),
                        finding_type="literature",
                        cycle=1,  # Question-driven treats all as cycle 1
                        source=finding_dict.get("source", {}),
                    )
                except Exception as e:
                    logger.warning(f"Failed to save finding: {e}")
        
        # Update cycle count for reporting
        self.current_cycle = 1
        
        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Question-Driven Research Summary")
        logger.info(f"   Papers found: {results.get('total_papers_found', 0)}")
        logger.info(f"   Papers processed: {results.get('total_papers_processed', 0)}")
        logger.info(f"   Findings: {results.get('total_findings', 0)}")
        logger.info(f"   Questions answered: {results.get('questions_answered', 0)}/{len(questions)}")
        logger.info(f"{'='*60}")

    def _execute_cycle(self, max_cycles: int = 10) -> None:
        """
        Run one complete research cycle:
          1. Generate tasks via Orchestrator
          2. Save tasks to world model
          3. Execute tasks in parallel
          4. Save findings to world model
        """
        # Step 1: Generate tasks
        tasks = self.orchestrator.generate_tasks(
            objective=self.objective,
            cycle=self.current_cycle,
            num_tasks=self.num_tasks_per_cycle,
            max_cycles=max_cycles,
        )

        if not tasks:
            logger.warning("No tasks generated — skipping cycle")
            return

        # Literature-only mode: convert data tasks to literature tasks
        if self.mode == "literature" or not self.has_dataset:
            converted = 0
            for task in tasks:
                if task.task_type == TaskType.DATA_ANALYSIS:
                    task.task_type = TaskType.LITERATURE_SEARCH
                    task.description = self._reframe_as_literature_task(
                        task.description
                    )
                    converted += 1
            if converted:
                logger.info(
                    f"  📚 Literature-only mode: converted {converted} "
                    f"data tasks → literature tasks"
                )

        # Step 2: Save tasks to world model
        for task in tasks:
            self.world_model.add_task(task)
        logger.info(
            f"📋 {len(tasks)} tasks queued: "
            f"{sum(1 for t in tasks if t.task_type == TaskType.DATA_ANALYSIS)} data, "
            f"{sum(1 for t in tasks if t.task_type == TaskType.LITERATURE_SEARCH)} literature"
        )

        # Step 3: Execute tasks in parallel
        all_results = self._execute_tasks_parallel(tasks)

        # Step 4: Save all findings to world model
        total_findings_added = 0
        total_findings_skipped = 0

        # Pre-seed deduplicator with existing findings so cross-cycle
        # duplicates are caught (not just within-cycle duplicates)
        existing = self.world_model.get_all_findings()
        for f in existing:
            self.finding_dedup.register(
                f.claim if hasattr(f, "claim") else str(f)
            )

        for task, result in all_results:
            if result is None:
                continue
            findings = result.get("findings", [])
            finding_ids = []

            for raw_finding in findings:
                claim = raw_finding.get("claim", "")

                # ── Deduplication check ───────────────────────────────
                dedup_result = self.finding_dedup.check(claim)
                if dedup_result.is_duplicate:
                    logger.info(
                        f"  🔁 Finding skipped (duplicate via {dedup_result.match_method}): "
                        f"{claim[:70]}..."
                    )
                    total_findings_skipped += 1
                    continue

                # Determine source type based on task type
                if task.task_type == TaskType.DATA_ANALYSIS:
                    source = {
                        "type": "notebook",
                        "path": str(result.get("notebook_path", "")),
                        "cell": result.get("code_cell_index", 0),
                    }
                else:
                    source = {
                        "type": "paper",
                        "paper_id": raw_finding.get("paper_id", ""),
                        "doi": raw_finding.get("doi", ""),
                        "title": raw_finding.get("paper_title", ""),
                    }

                try:
                    # Get figures for data analysis tasks and copy to run output
                    figures = []
                    if task.task_type == TaskType.DATA_ANALYSIS:
                        raw_figures = result.get("figures", [])
                        for fig_path in raw_figures:
                            try:
                                fig_path = Path(fig_path)
                                if fig_path.exists():
                                    # Copy to run's figures directory
                                    dest = self.run_figures_dir / fig_path.name
                                    import shutil
                                    shutil.copy2(fig_path, dest)
                                    # Store relative path for markdown embedding
                                    figures.append(f"figures/{fig_path.name}")
                                    logger.debug(f"  📊 Copied figure: {fig_path.name}")
                            except Exception as e:
                                logger.warning(f"Failed to copy figure {fig_path}: {e}")
                        if figures:
                            logger.info(f"  📊 {len(figures)} figures saved")
                    
                    fid = self.world_model.add_finding(
                        claim=claim,
                        finding_type=(
                            "data_analysis"
                            if task.task_type == TaskType.DATA_ANALYSIS
                            else "literature"
                        ),
                        source=source,
                        cycle=self.current_cycle,
                        confidence=float(raw_finding.get("confidence", 0.5)),
                        tags=raw_finding.get("tags", []),
                        evidence=raw_finding.get("evidence", ""),
                        figures=figures,
                    )
                    if fid is None:
                        continue
                    finding_ids.append(fid)
                    total_findings_added += 1

                    # Register so future findings in this cycle are also checked
                    self.finding_dedup.register(claim)

                except Exception as e:
                    logger.error(f"Failed to save finding: {e}")
                    continue

            # Mark task complete in world model
            self.world_model.update_task(
                task_id=task.id,
                status="completed",
                result_finding_ids=finding_ids,
            )

        # Propose relationships between new and existing findings
        relationships_created = self.orchestrator.propose_relationships(
            objective=self.objective,
            cycle=self.current_cycle,
        )

        dedup_msg = f", {total_findings_skipped} duplicates skipped" if total_findings_skipped else ""
        logger.info(
            f"💾 Cycle {self.current_cycle} complete: "
            f"{total_findings_added} new findings, "
            f"{relationships_created} relationships created"
            f"{dedup_msg}"
        )

        # Record cycle summary in stage tracker
        self.tracker.cycle_summary(
            cycle=self.current_cycle,
            findings=total_findings_added,
            relationships=relationships_created,
            tasks=len(tasks),
        )

        # Compress this cycle's findings into tier-2 summary
        cycle_findings = [
            {"claim": r.get("claim", ""), "confidence": r.get("confidence", 0),
             "evidence": r.get("evidence", ""), "finding_type": r.get("finding_type", ""),
             "schol_eval": r.get("schol_eval", {})}
            for _, result in all_results if result
            for r in result.get("findings", [])
        ]
        self.compressor.compress_cycle(
            cycle=self.current_cycle,
            findings=cycle_findings,
            relationships_count=relationships_created,
        )
        stats = self.compressor.get_compression_stats()
        logger.info(f"🗅️ Compression: {stats['cycles_compressed']} cycles compressed")

    def _execute_tasks_parallel(
        self, tasks: list[Task]
    ) -> list[tuple[Task, Optional[dict]]]:
        """
        Execute multiple tasks in parallel using ThreadPoolExecutor.

        I/O-bound work (LLM calls, API requests) benefits from threads
        even though Python has the GIL — threads release it during I/O.

        Args:
            tasks: List of Task objects to execute

        Returns:
            List of (task, result_dict) tuples.
            result is None if the task failed.
        """
        results = []

        # Pre-compute world model summary ONCE before submitting to threads.
        # This avoids 3 threads all hitting SQLite simultaneously at startup.
        precomputed_summary = self._get_world_model_summary()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._execute_single_task, task, precomputed_summary): task
                for task in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append((task, result))
                    logger.info(
                        f"  ✓ Task done: '{task.description[:50]}...' "
                        f"→ {len(result.get('findings', []))} findings"
                    )
                    # Compress task output for context management
                    task_dict = {
                        "id": task.id,
                        "description": task.description,
                        "type": task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type),
                        "cycle": task.cycle,
                    }
                    self.compressor.compress_task(task_dict, result)
                except Exception as e:
                    import traceback
                    tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    logger.error(
                        f"  ✗ Task failed: '{task.description[:50]}...' → {e}\n"
                        f"    Full traceback:\n{tb_str}"
                    )
                    # Mark as failed in world model
                    self.world_model.update_task(
                        task_id=task.id,
                        status="failed",
                        error_message=str(e),
                    )
                    results.append((task, None))

        return results

    def _get_world_model_summary(self) -> str:
        """
        Get the world model summary, using compressed context if available.
        Called ONCE before parallel task execution to avoid concurrent SQLite reads.
        """
        raw_summary = self.world_model.get_summary()

        if self.compressor._cycle_summaries or any(self.compressor._task_summaries.values()):
            top_findings_raw = self.world_model.get_top_findings(n=5) if hasattr(self.world_model, 'get_top_findings') else []
            top_findings = [
                {"claim": item["finding"].claim, "confidence": item["finding"].confidence}
                if isinstance(item, dict) and "finding" in item
                else item
                for item in top_findings_raw
            ]
            run_context = self.compressor.build_run_context(
                objective=self.objective,
                total_findings=self.world_model.get_finding_count() if hasattr(self.world_model, 'get_finding_count') else 0,
                total_relationships=self.world_model.get_relationship_count() if hasattr(self.world_model, 'get_relationship_count') else 0,
                top_findings_global=top_findings,
            )
            return run_context.compressed_summary
        return raw_summary

    def _execute_single_task(self, task: Task, world_model_summary: str = "") -> dict:
        """
        Route a single task to the correct agent and execute it.

        This runs inside a thread — each task gets its own thread.

        Args:
            task: Task object with type, description, goal
            world_model_summary: Pre-computed summary (avoids concurrent SQLite reads)

        Returns:
            Result dict from the agent's execute() method
        """

        # Mark task as running (may fail under concurrent access — non-fatal)
        try:
            self.world_model.update_task(task_id=task.id, status="running")
        except Exception as e:
            logger.debug(f"Non-fatal: could not mark task {task.id[:8]} as running: {e}")

        task_dict = {
            "description": task.description,
            "goal": task.goal or "",
            "cycle": task.cycle,
            "id": task.id,
        }

        if task.task_type == TaskType.DATA_ANALYSIS:
            if not self.has_dataset:
                logger.warning(
                    f"  ⏭️ Skipping data task (no dataset): {task.description[:50]}..."
                )
                return {
                    "findings": [],
                    "notebook_path": "",
                    "execution_result": None,
                    "code": "",
                    "plan": "Skipped — no dataset provided",
                    "code_cell_index": 0,
                    "attempts": 0,
                }
            logger.info(f"  🔢 Data task starting: {task.description[:50]}...")
            return self.data_agent.execute(
                task=task_dict,
                data_path=self.data_path,
                objective=self.objective,
                world_model_summary=world_model_summary,
            )

        elif task.task_type == TaskType.LITERATURE_SEARCH:
            logger.info(f"  📚 Literature task starting: {task.description[:50]}...")

            # Build a compact dataset summary to anchor literature queries
            dataset_summary = self._build_dataset_summary()

            return self.literature_agent.execute(
                task=task_dict,
                objective=self.objective,
                world_model_summary=world_model_summary,
                dataset_summary=dataset_summary,
            )

        else:
            raise ValueError(f"Unknown task type: {task.task_type}")