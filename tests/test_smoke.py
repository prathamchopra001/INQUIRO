# -*- coding: utf-8 -*-
"""
INQUIRO Smoke Test Suite
=======================
Verifies all modules and components are importable and minimally functional.
Does NOT run LLM calls or Docker containers — just checks the plumbing.

Run with:
    pytest tests/test_smoke.py -v --log-file=logs/smoke_test.log --log-file-level=DEBUG

Logs are saved to: logs/smoke_test.log
"""

import pytest
import logging
import os
import json
import tempfile
from pathlib import Path

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFTEST-STYLE FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def tmp_dir():
    """Shared temp directory for the test session."""
    # ignore_cleanup_errors=True prevents Windows PermissionError
    # when ChromaDB holds chroma.sqlite3 open at session teardown
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        yield d


@pytest.fixture(scope="session")
def sample_finding():
    return {
        "claim": "Metabolite A shows 1.53-fold change (p=6.32e-31)",
        "evidence": "ANOVA with Benjamini-Hochberg correction, n=50 samples",
        "confidence": 0.98,
        "tags": ["metabolomics"],
        "paper_id": "nb_001",
    }


@pytest.fixture(scope="session")
def irb_finding():
    return {
        "claim": "The study protocol was approved by the institutional review board IRB No. 030763",
        "evidence": "Ethics committee approval obtained from Vanderbilt University Medical Center",
        "confidence": 0.95,
        "tags": [],
        "paper_id": "lit_001",
    }


# =============================================================================
# TRACK A: OUTPUT QUALITY
# =============================================================================

class TestScholarEval:
    """A1: ScholarEval 8-dimension validation."""

    def test_import(self):
        from src.validation.schol_eval import ScholarEval, ScholarEvalResult
        logger.info("ScholarEval imported successfully")

    def test_strong_finding_passes(self, sample_finding):
        from src.validation.schol_eval import ScholarEval
        ev = ScholarEval(min_score=0.40)
        result = ev.evaluate(sample_finding, source_type="data_analysis")
        assert result.passes, f"Strong finding should pass (score={result.composite_score:.2f})"
        logger.info(f"Strong finding score: {result.composite_score:.2f}")

    def test_irb_finding_rejected(self, irb_finding):
        from src.validation.schol_eval import ScholarEval
        ev = ScholarEval(min_score=0.40)
        result = ev.evaluate(irb_finding, source_type="literature")
        assert not result.passes, f"IRB finding should be rejected (score={result.composite_score:.2f})"
        logger.info(f"IRB finding correctly rejected (score={result.composite_score:.2f})")

    def test_all_dimensions_present(self, sample_finding):
        from src.validation.schol_eval import ScholarEval
        ev = ScholarEval()
        result = ev.evaluate(sample_finding, source_type="data_analysis")
        dims = [
            "statistical_validity", "reproducibility", "novelty", "significance",
            "methodological_soundness", "evidence_quality", "claim_calibration",
            "citation_support",
        ]
        result_dict = result.to_dict()["schol_eval"]
        for d in dims:
            assert d in result_dict, f"Missing dimension: {d}"
        logger.info("All 8 ScholarEval dimensions present")

    def test_composite_score_in_range(self, sample_finding):
        from src.validation.schol_eval import ScholarEval
        ev = ScholarEval()
        result = ev.evaluate(sample_finding, source_type="data_analysis")
        assert 0.0 <= result.composite_score <= 1.0
        logger.info(f"Composite score in valid range: {result.composite_score:.2f}")


class TestContextCompressor:
    """A2: Hierarchical context compression."""

    def test_import(self):
        from src.compression.context_compressor import ContextCompressor
        logger.info("ContextCompressor imported successfully")

    def test_tier1_task_compression(self):
        from src.compression.context_compressor import ContextCompressor, TaskSummary
        c = ContextCompressor()
        task = {"id": "t1", "description": "PCA analysis", "type": "data_analysis", "cycle": 1}
        result = {"findings": [{"claim": "PC1 explains 74%", "confidence": 0.88, "evidence": "PCA"}]}
        summary = c.compress_task(task, result)
        assert isinstance(summary, TaskSummary)
        assert summary.findings_count == 1
        assert len(summary.compressed_text) <= ContextCompressor.TIER1_MAX_CHARS
        logger.info(f"Tier 1 compression: {len(summary.compressed_text)} chars")

    def test_tier2_cycle_compression(self):
        from src.compression.context_compressor import ContextCompressor, CycleSummary
        c = ContextCompressor()
        # Need a task registered first
        c.compress_task(
            {"id": "t1", "description": "test", "type": "data_analysis", "cycle": 1},
            {"findings": []}
        )
        findings = [{"claim": "Finding A", "confidence": 0.9, "evidence": "test", "finding_type": "data"}]
        cs = c.compress_cycle(cycle=1, findings=findings, relationships_count=2)
        assert isinstance(cs, CycleSummary)
        assert len(cs.compressed_text) <= ContextCompressor.TIER2_MAX_CHARS
        logger.info(f"Tier 2 compression: {len(cs.compressed_text)} chars")

    def test_tier3_run_context(self):
        from src.compression.context_compressor import ContextCompressor
        c = ContextCompressor()
        c.compress_task({"id": "t1", "description": "test", "type": "data_analysis", "cycle": 1}, {"findings": []})
        c.compress_cycle(cycle=1, findings=[], relationships_count=0)
        ctx = c.build_run_context("Test objective", 5, 3, [])
        assert "RESEARCH OBJECTIVE" in ctx.compressed_summary
        assert len(ctx.compressed_summary) <= ContextCompressor.TIER3_MAX_CHARS
        logger.info(f"Tier 3 context: {len(ctx.compressed_summary)} chars")


class TestNoveltyDetector:
    """A3: Novelty detection."""

    def test_import(self):
        from src.novelty.novelty_detector import NoveltyDetector
        logger.info("NoveltyDetector imported successfully")

    def test_novel_task_passes(self):
        from src.novelty.novelty_detector import NoveltyDetector
        d = NoveltyDetector(threshold=0.65)
        result = d.check("Perform correlation analysis between metabolite E and age")
        assert result.is_novel
        logger.info("Novel task correctly identified")

    def test_duplicate_task_rejected(self):
        from src.novelty.novelty_detector import NoveltyDetector
        d = NoveltyDetector(threshold=0.65)
        d.register_task("Perform PCA on the metabolomics dataset to identify groupings")
        result = d.check("Perform PCA on the metabolomics dataset to identify groupings and clusters")
        assert not result.is_novel
        logger.info(f"Duplicate task correctly rejected (sim={result.similarity_score:.2f})")

    def test_batch_check(self):
        from src.novelty.novelty_detector import NoveltyDetector
        d = NoveltyDetector(threshold=0.65)
        d.register_task("Perform PCA on the metabolomics dataset to identify groupings")
        tasks = [
            "Perform differential abundance analysis with ANOVA",
            "Run pathway enrichment on significant metabolites",
            "Perform PCA on the metabolomics dataset to identify groupings",
        ]
        results = d.check_batch(tasks)
        novel = [t for t, r in results if r.is_novel]
        assert len(novel) == 2
        logger.info(f"Batch check: {len(novel)}/3 novel tasks")

    def test_stats(self):
        from src.novelty.novelty_detector import NoveltyDetector
        d = NoveltyDetector()
        d.register_task("some task")
        d.register_finding("some finding")
        stats = d.get_stats()
        assert stats["registered_tasks"] == 1
        assert stats["registered_findings"] == 1
        logger.info(f"Novelty detector stats: {stats}")


# =============================================================================
# TRACK B: LITERATURE EXPANSION
# =============================================================================

class TestArXivClient:
    """B1: ArXiv search client."""

    def test_import(self):
        from src.literature.arxiv_search import ArXivClient
        logger.info("ArXivClient imported successfully")

    def test_initialization(self):
        from src.literature.arxiv_search import ArXivClient
        client = ArXivClient(max_results=5)
        assert client.max_results == 5
        assert client._min_interval == 3.0
        logger.info("ArXivClient initialized correctly")

    def test_https_url(self):
        from src.literature import arxiv_search
        assert arxiv_search.ARXIV_API_URL.startswith("https://")
        logger.info(f"ArXiv URL uses HTTPS: {arxiv_search.ARXIV_API_URL}")

    def test_rate_limit_method_exists(self):
        from src.literature.arxiv_search import ArXivClient
        client = ArXivClient()
        assert hasattr(client, "_rate_limit")
        logger.info("ArXiv rate limiting method present")


class TestPubMedClient:
    """B2: PubMed search client."""

    def test_import(self):
        from src.literature.pubmed_search import PubMedClient
        logger.info("PubMedClient imported successfully")

    def test_initialization(self):
        from src.literature.pubmed_search import PubMedClient
        client = PubMedClient(max_results=10)
        assert client.max_results == 10
        logger.info("PubMedClient initialized correctly")

    def test_summary_parsing(self):
        from src.literature.pubmed_search import PubMedClient
        client = PubMedClient()
        summary = {
            "title": "Metabolomics study",
            "authors": [{"name": "Smith J"}],
            "pubdate": "2024",
            "fulljournalname": "Journal of Metabolomics",
            "articleids": [
                {"idtype": "doi", "value": "10.1000/test"},
                {"idtype": "pmc", "value": "PMC1234567"},
            ],
        }
        paper = client._parse_summary("12345678", summary)
        assert paper is not None
        assert paper.year == 2024
        assert paper.paper_id == "pubmed_12345678"
        assert paper.is_open_access is True
        assert isinstance(paper.authors, list)
        assert all(isinstance(a, str) for a in paper.authors)
        logger.info(f"PubMed paper parsed: {paper.title}")

    def test_authors_are_strings(self):
        """Ensure authors are List[str] not List[dict]."""
        from src.literature.pubmed_search import PubMedClient
        client = PubMedClient()
        summary = {
            "title": "Test paper",
            "authors": [{"name": "Jones A"}, {"name": "Smith B"}],
            "pubdate": "2023",
            "fulljournalname": "Test Journal",
            "articleids": [],
        }
        paper = client._parse_summary("99999999", summary)
        assert paper is not None
        assert paper.authors == ["Jones A", "Smith B"]
        logger.info("PubMed authors correctly returned as List[str]")


class TestLiteratureAgentMultiSource:
    """B: Multi-source literature agent wiring."""

    def test_agent_has_all_clients(self):
        from src.agents.literature import LiteratureSearchAgent
        from src.utils.llm_client import LLMClient
        from src.literature.arxiv_search import ArXivClient
        from src.literature.pubmed_search import PubMedClient
        agent = LiteratureSearchAgent(
            llm_client=LLMClient(),
            arxiv_client=ArXivClient(),
            pubmed_client=PubMedClient(),
        )
        assert hasattr(agent, "search_client")
        assert hasattr(agent, "arxiv_client")
        assert hasattr(agent, "pubmed_client")
        logger.info("LiteratureSearchAgent has all 3 search clients")


# =============================================================================
# TRACK C: ORCHESTRATION INTELLIGENCE
# =============================================================================

class TestPlanReviewer:
    """C: Plan reviewer and explore/exploit ratio."""

    def test_import(self):
        from src.orchestration.plan_reviewer import PlanReviewer, TaskScore
        logger.info("PlanReviewer imported successfully")

    def test_ratio_cycle_1(self):
        from src.orchestration.plan_reviewer import PlanReviewer
        pr = PlanReviewer()
        r = pr.get_explore_exploit_ratio(cycle=1, max_cycles=10)
        assert r["mode"] == "exploratory"
        assert r["explore"] > 0.5
        logger.info(f"Cycle 1 mode: {r['mode']} (explore={r['explore']})")

    def test_ratio_cycle_5(self):
        from src.orchestration.plan_reviewer import PlanReviewer
        pr = PlanReviewer()
        r = pr.get_explore_exploit_ratio(cycle=5, max_cycles=10)
        assert r["mode"] == "balanced"
        logger.info(f"Cycle 5 mode: {r['mode']}")

    def test_ratio_cycle_10(self):
        from src.orchestration.plan_reviewer import PlanReviewer
        pr = PlanReviewer()
        r = pr.get_explore_exploit_ratio(cycle=10, max_cycles=10)
        assert r["mode"] == "exploitative"
        assert r["exploit"] > 0.5
        logger.info(f"Cycle 10 mode: {r['mode']} (exploit={r['exploit']})")

    def test_ratio_decreases_over_time(self):
        from src.orchestration.plan_reviewer import PlanReviewer
        pr = PlanReviewer()
        r1 = pr.get_explore_exploit_ratio(cycle=1, max_cycles=10)
        r10 = pr.get_explore_exploit_ratio(cycle=10, max_cycles=10)
        assert r1["explore"] > r10["explore"]
        logger.info("Explore ratio correctly decreases over cycles")

    def test_mode_instruction_format(self):
        from src.orchestration.plan_reviewer import PlanReviewer
        pr = PlanReviewer()
        instruction = pr.get_mode_instruction(cycle=1, max_cycles=10)
        assert "EXPLORATORY" in instruction
        assert "Cycle 1/10" in instruction
        logger.info("Mode instruction correctly formatted")

    def test_task_scoring(self):
        from src.orchestration.plan_reviewer import PlanReviewer, TaskScore
        from src.world_model.models import Task, TaskType
        pr = PlanReviewer(min_score=0.45)
        task = Task(
            task_type=TaskType.DATA_ANALYSIS,
            description="Perform differential abundance analysis with Mann-Whitney test",
            goal="Identify significant metabolite differences",
            cycle=1,
        )
        results = pr.review_tasks([task], "Identify metabolic features", 1, 10)
        assert len(results) == 1
        _, score = results[0]
        assert isinstance(score, TaskScore)
        assert 0.0 <= score.composite <= 1.0
        logger.info(f"Task score: {score.composite:.2f}")


# =============================================================================
# TRACK D: EXECUTION RELIABILITY
# =============================================================================

class TestCircuitBreaker:
    """D1: Circuit breaker."""

    def test_import(self):
        from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError
        logger.info("CircuitBreaker imported successfully")

    def test_starts_closed(self):
        from src.utils.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(name="smoke_test", failure_threshold=3)
        assert cb.state.value == "closed"
        logger.info("Circuit breaker starts CLOSED")

    def test_trips_open_after_failures(self):
        from src.utils.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(name="smoke_test_trip", failure_threshold=3)
        for _ in range(3):
            try:
                with cb:
                    raise ValueError("simulated")
            except ValueError:
                pass
        assert cb.state.value == "open"
        logger.info("Circuit breaker correctly trips OPEN")

    def test_blocked_call_raises(self):
        from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError
        cb = CircuitBreaker(name="smoke_test_block", failure_threshold=3)
        for _ in range(3):
            try:
                with cb:
                    raise ValueError("simulated")
            except ValueError:
                pass
        with pytest.raises(CircuitOpenError):
            with cb:
                pass
        logger.info("Blocked call raises CircuitOpenError correctly")

    def test_reset(self):
        from src.utils.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(name="smoke_test_reset", failure_threshold=3)
        for _ in range(3):
            try:
                with cb:
                    raise ValueError("simulated")
            except ValueError:
                pass
        cb.reset()
        assert cb.state.value == "closed"
        logger.info("Circuit breaker resets correctly")

    def test_stats(self):
        from src.utils.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(name="smoke_test_stats", failure_threshold=3)
        stats = cb.get_stats()
        assert "state" in stats
        assert "total_calls" in stats
        assert "total_failures" in stats
        logger.info(f"Circuit breaker stats: {stats}")


class TestPackageResolver:
    """D3: Package resolver."""

    def test_import(self):
        from src.execution.package_resolver import PackageResolver
        logger.info("PackageResolver imported successfully")

    def test_detects_module_not_found(self):
        from src.execution.package_resolver import PackageResolver
        pr = PackageResolver()
        pkgs = pr.detect_missing_packages("ModuleNotFoundError: No module named 'gseapy'")
        assert "gseapy" in pkgs
        logger.info(f"Detected missing package: {pkgs}")

    def test_maps_sklearn(self):
        from src.execution.package_resolver import PackageResolver
        pr = PackageResolver()
        pkgs = pr.detect_missing_packages("cannot import name 'linkage' from 'sklearn.cluster'")
        assert "scikit-learn" in pkgs
        logger.info(f"Mapped sklearn → scikit-learn: {pkgs}")

    def test_patches_code(self):
        from src.execution.package_resolver import PackageResolver
        pr = PackageResolver()
        code = "# -*- coding: utf-8 -*-\nimport gseapy\nprint('hello')"
        patched = pr.patch_code(code, "ModuleNotFoundError: No module named 'gseapy'")
        assert "pip" in patched
        assert patched != code
        logger.info("Code successfully patched with pip install")

    def test_no_double_patch(self):
        from src.execution.package_resolver import PackageResolver
        pr = PackageResolver()
        code = "import gseapy"
        pr.patch_code(code, "ModuleNotFoundError: No module named 'gseapy'")
        patched2 = pr.patch_code(code, "ModuleNotFoundError: No module named 'gseapy'")
        assert patched2 == code
        logger.info("Already-installed package not patched again")

    def test_docker_executor_has_resolver(self):
        from src.execution.docker_executor import DockerExecutor
        ex = DockerExecutor()
        assert hasattr(ex, "_resolver")
        assert ex.pool_size == 2
        logger.info(f"DockerExecutor has resolver, pool_size={ex.pool_size}")


# =============================================================================
# TRACK E: LLM OPTIMIZATION
# =============================================================================

class TestLLMClient:
    """E: Model routing and circuit breaker on LLMClient."""

    def test_import(self):
        from src.utils.llm_client import LLMClient
        logger.info("LLMClient imported successfully")

    def test_has_routing_attributes(self):
        from src.utils.llm_client import LLMClient
        llm = LLMClient()
        assert hasattr(llm, "complete_for_task")
        assert hasattr(llm, "_complex_model")
        assert hasattr(llm, "_simple_model")
        logger.info(f"LLM routing: complex={llm._complex_model}, simple={llm._simple_model}")

    def test_has_circuit_breaker(self):
        from src.utils.llm_client import LLMClient
        llm = LLMClient()
        assert hasattr(llm, "_circuit_breaker")
        assert llm._circuit_breaker.state.value == "closed"
        assert llm._circuit_breaker.name == llm.provider
        logger.info(f"LLM circuit breaker: provider={llm.provider}, state=closed")

    def test_anthropic_routes_to_haiku(self):
        from src.utils.llm_client import LLMClient
        llm = LLMClient()
        simple = llm._resolve_simple_model("anthropic", "claude-sonnet-4-6")
        assert simple == "claude-haiku-4-5-20251001"
        logger.info(f"Anthropic routing: sonnet → haiku ✓")

    def test_openai_routes_to_mini(self):
        from src.utils.llm_client import LLMClient
        llm = LLMClient()
        simple = llm._resolve_simple_model("openai", "gpt-4o")
        assert simple == "gpt-4o-mini"
        logger.info("OpenAI routing: gpt-4o → gpt-4o-mini ✓")

    def test_ollama_same_model(self):
        from src.utils.llm_client import LLMClient
        llm = LLMClient()
        if llm.provider == "ollama":
            assert llm._simple_model == llm._complex_model
            logger.info("Ollama: simple == complex (correct local fallback)")


# =============================================================================
# TRACK F: STAGE TRACKING
# =============================================================================

class TestStageTracker:
    """F: JSONL stage tracking."""

    def test_import(self):
        from src.tracking.stage_tracker import StageTracker
        logger.info("StageTracker imported successfully")

    def test_creates_jsonl_file(self, tmp_dir):
        from src.tracking.stage_tracker import StageTracker
        tracker = StageTracker(run_id="smoke_test_001", output_dir=tmp_dir)
        path = tracker.get_output_path()
        assert Path(path).exists()
        logger.info(f"JSONL file created: {path}")

    def test_context_manager_writes_entries(self, tmp_dir):
        from src.tracking.stage_tracker import StageTracker
        tracker = StageTracker(run_id="smoke_test_002", output_dir=tmp_dir)
        with tracker.track("cycle", cycle=1):
            with tracker.track("task", substage="data", cycle=1):
                pass
        with open(tracker.get_output_path()) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        stages = [e["stage"] for e in entries]
        assert "run_start" in stages
        assert "cycle" in stages
        assert "task" in stages
        logger.info(f"JSONL entries written: {len(entries)}")

    def test_duration_recorded(self, tmp_dir):
        from src.tracking.stage_tracker import StageTracker
        tracker = StageTracker(run_id="smoke_test_003", output_dir=tmp_dir)
        with tracker.track("test_stage"):
            pass
        with open(tracker.get_output_path()) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        completed = [e for e in entries if e.get("status") == "completed"]
        assert all("duration_ms" in e for e in completed)
        logger.info("Duration recorded for completed stages")

    def test_run_id_consistent(self, tmp_dir):
        from src.tracking.stage_tracker import StageTracker
        # Use unique subdir so previous test entries don't bleed in
        unique_dir = str(Path(tmp_dir) / "run_id_test")
        tracker = StageTracker(run_id="smoke_test_004", output_dir=unique_dir)
        tracker.event("test_event")
        with open(tracker.get_output_path()) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        assert all(e["run_id"] == "smoke_test_004" for e in entries)
        logger.info("Run ID consistent across all entries")

    def test_finding_added_event(self, tmp_dir):
        from src.tracking.stage_tracker import StageTracker
        tracker = StageTracker(run_id="smoke_test_005", output_dir=tmp_dir)
        tracker.finding_added("Metabolite A significant", 0.95, "data_analysis", 1)
        with open(tracker.get_output_path()) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        finding_entries = [e for e in entries if e["stage"] == "finding_added"]
        assert len(finding_entries) == 1
        logger.info("Finding added event recorded correctly")


# =============================================================================
# WORLD MODEL
# =============================================================================

class TestWorldModel:
    """Core world model smoke tests."""

    def test_import(self):
        from src.world_model.world_model import WorldModel
        logger.info("WorldModel imported successfully")

    def test_initialization(self, tmp_dir):
        from src.world_model.world_model import WorldModel
        wm = WorldModel(db_path=str(Path(tmp_dir) / "smoke_wm.db"))
        assert wm is not None
        wm.close()
        logger.info("WorldModel initialized successfully")

    def test_add_and_get_finding(self, tmp_dir):
        from src.world_model.world_model import WorldModel
        wm = WorldModel(db_path=str(Path(tmp_dir) / "smoke_wm2.db"))
        fid = wm.add_finding(
            claim="Metabolite A significant (p=0.001)",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "analysis_001.ipynb"},
            cycle=1,
            confidence=0.9,
        )
        assert fid is not None
        finding = wm.get_finding(fid)
        assert finding is not None
        assert finding.claim == "Metabolite A significant (p=0.001)"
        wm.close()
        logger.info(f"Finding added and retrieved: {fid[:8]}...")

    def test_get_finding_count(self, tmp_dir):
        from src.world_model.world_model import WorldModel
        wm = WorldModel(db_path=str(Path(tmp_dir) / "smoke_wm3.db"))
        wm.add_finding(
            claim="Test claim",
            finding_type="data_analysis",
            source={"type": "notebook", "path": "test.ipynb"},
            cycle=1,
        )
        count = wm.get_finding_count()
        assert count == 1
        wm.close()
        logger.info(f"Finding count: {count}")

    def test_get_relationship_count(self, tmp_dir):
        from src.world_model.world_model import WorldModel
        wm = WorldModel(db_path=str(Path(tmp_dir) / "smoke_wm4.db"))
        count = wm.get_relationship_count()
        assert count == 0
        wm.close()
        logger.info(f"Relationship count: {count}")

    def test_safe_datetime_fix(self, tmp_dir):
        """Ensure _safe_datetime handles empty strings gracefully."""
        from src.world_model.database import Database
        from datetime import datetime
        result = Database._safe_datetime("")
        assert isinstance(result, datetime)
        result2 = Database._safe_datetime(None)
        assert isinstance(result2, datetime)
        result3 = Database._safe_datetime("2026-02-20 10:00:00")
        assert isinstance(result3, datetime)
        logger.info("_safe_datetime handles edge cases correctly")


# =============================================================================
# RAG SYSTEM
# =============================================================================

class TestRAGSystem:
    """ChromaDB RAG system smoke tests."""

    def test_import(self):
        from src.literature.rag import RAGSystem
        logger.info("RAGSystem imported successfully")

    def test_initialization(self, tmp_dir):
        from src.literature.rag import RAGSystem
        rag = RAGSystem(
            collection_name="smoke_test",
            persist_dir=str(Path(tmp_dir) / "chroma"),
        )
        assert rag is not None
        logger.info("RAGSystem initialized successfully")

    def test_empty_collection_returns_empty(self, tmp_dir):
        from src.literature.rag import RAGSystem
        rag = RAGSystem(
            collection_name="smoke_empty",
            persist_dir=str(Path(tmp_dir) / "chroma_empty"),
        )
        results = rag.query("test query", top_k=5)
        assert results == []
        logger.info("Empty collection returns [] without crashing")

    def test_n_results_guard(self, tmp_dir):
        """ChromaDB should not crash when n_results > collection size."""
        from src.literature.rag import RAGSystem
        rag = RAGSystem(
            collection_name="smoke_guard",
            persist_dir=str(Path(tmp_dir) / "chroma_guard"),
        )
        # Query with large top_k on empty collection
        results = rag.query("metabolomics", top_k=100)
        assert results == []
        logger.info("n_results guard works on empty collection")


# =============================================================================
# INTEGRATION: ALL IMPORTS TOGETHER
# =============================================================================

class TestIntegration:
    """Verify all Phase 6 modules import cleanly together."""

    def test_all_phase6_imports(self):
        from src.validation.schol_eval          import ScholarEval
        from src.compression.context_compressor import ContextCompressor
        from src.novelty.novelty_detector        import NoveltyDetector
        from src.literature.arxiv_search         import ArXivClient
        from src.literature.pubmed_search        import PubMedClient
        from src.orchestration.plan_reviewer     import PlanReviewer
        from src.utils.circuit_breaker           import CircuitBreaker
        from src.execution.package_resolver      import PackageResolver
        from src.execution.docker_executor       import DockerExecutor
        from src.tracking.stage_tracker          import StageTracker
        from src.utils.llm_client                import LLMClient
        from src.agents.orchestrator             import OrchestratorAgent
        from src.agents.data_analysis            import DataAnalysisAgent
        from src.agents.literature               import LiteratureSearchAgent
        from src.core.inquiro                     import Inquiro
        logger.info("All Phase 6 modules imported cleanly together")

    def test_orchestrator_has_novelty_and_reviewer(self):
        from src.agents.orchestrator import OrchestratorAgent
        from src.novelty.novelty_detector import NoveltyDetector
        from src.orchestration.plan_reviewer import PlanReviewer
        assert hasattr(OrchestratorAgent, '__init__')
        # Instantiate with mocks
        import unittest.mock as mock
        orch = OrchestratorAgent(
            llm_client=mock.MagicMock(),
            world_model=mock.MagicMock(),
        )
        assert hasattr(orch, "novelty_detector")
        assert hasattr(orch, "plan_reviewer")
        assert isinstance(orch.novelty_detector, NoveltyDetector)
        assert isinstance(orch.plan_reviewer, PlanReviewer)
        logger.info("OrchestratorAgent has NoveltyDetector and PlanReviewer")

    def test_inquiro_has_all_phase6_components(self):
        """Verify Inquiro.__init__ wires up all Phase 6 components."""
        import inspect
        from src.core.inquiro import Inquiro
        source = inspect.getsource(Inquiro.__init__)
        assert "ContextCompressor" in source
        assert "StageTracker" in source
        assert "warm_pool" in source
        logger.info("Inquiro.__init__ contains all Phase 6 components")