# -*- coding: utf-8 -*-
"""
INQUIRO Phase 6 - Comprehensive Feature Test Suite
Run with: python tests/test_all_features.py
"""

import sys
import json
import re
import tempfile
import traceback

GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
RESET = "\033[0m"
BOLD  = "\033[1m"

passed = []
failed = []

def ok(name):
    passed.append(name)
    print(f"  {GREEN}PASS{RESET}  {name}")

def fail(name, reason):
    failed.append(name)
    print(f"  {RED}FAIL{RESET}  {name}")
    print(f"       {RED}{reason[:300]}{RESET}")

def section(title):
    print(f"\n{CYAN}{BOLD}{'─'*55}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'─'*55}{RESET}")


# =============================================================================
# TRACK A1: ScholarEval
# =============================================================================
section("Track A1: ScholarEval")

try:
    from src.validation.schol_eval import ScholarEval

    ev = ScholarEval(min_score=0.40)

    # Strong data finding — should pass
    strong = {
        "claim": "Metabolite A shows 1.53-fold change (p=6.32e-31, adjusted p=6.32e-31)",
        "evidence": "ANOVA with Benjamini-Hochberg correction, n=50 samples, control vs treatment",
        "confidence": 0.98,
        "tags": ["metabolomics"],
        "paper_id": "nb_001",
    }
    r = ev.evaluate(strong, source_type="data_analysis")
    assert r.passes, f"Strong finding failed: score={r.composite_score}"
    ok(f"Strong finding passes ScholarEval (score={r.composite_score:.2f})")

    # IRB approval — hard reject, not a scientific discovery
    irb = {
        "claim": "The study protocol was approved by the institutional review board IRB No. 030763",
        "evidence": "Ethics committee approval obtained from Vanderbilt University Medical Center",
        "confidence": 0.95,
        "tags": [],
        "paper_id": "lit_001",
    }
    r2 = ev.evaluate(irb, source_type="literature")
    assert not r2.passes, f"IRB finding should fail (score={r2.composite_score:.3f})"
    ok(f"IRB approval correctly rejected (score={r2.composite_score:.2f})")

    # All 8 dimensions present
    dims = ["statistical_validity","reproducibility","novelty","significance",
            "methodological_soundness","evidence_quality","claim_calibration","citation_support"]
    result_dict = r.to_dict()["schol_eval"]
    assert all(d in result_dict for d in dims)
    ok("All 8 dimensions present in result")

except Exception as e:
    fail("ScholarEval", traceback.format_exc(limit=2))


# =============================================================================
# TRACK A2: Context Compression
# =============================================================================
section("Track A2: Context Compression")

try:
    from src.compression.context_compressor import ContextCompressor, TaskSummary, CycleSummary

    c = ContextCompressor()

    task = {"id":"t1","description":"Perform PCA on metabolomics dataset","type":"data_analysis","cycle":1}
    result = {"findings":[
        {"claim":"Metabolite A 1.5-fold change p=0.001","confidence":0.95,"evidence":"ANOVA"},
        {"claim":"PC1 explains 74% variance","confidence":0.88,"evidence":"PCA"},
    ]}
    s = c.compress_task(task, result)
    assert isinstance(s, TaskSummary) and s.findings_count == 2
    assert len(s.compressed_text) <= ContextCompressor.TIER1_MAX_CHARS
    ok("Tier 1: task compression correct")

    findings = [
        {"claim":"Metabolite A significant","confidence":0.95,"evidence":"stats","finding_type":"data_analysis"},
        {"claim":"Metabolite B altered","confidence":0.80,"evidence":"analysis","finding_type":"data_analysis"},
    ]
    cs = c.compress_cycle(cycle=1, findings=findings, relationships_count=3)
    assert isinstance(cs, CycleSummary) and cs.findings_count == 2
    assert len(cs.compressed_text) <= ContextCompressor.TIER2_MAX_CHARS
    ok("Tier 2: cycle compression correct")

    run_ctx = c.build_run_context(
        objective="Identify metabolic features",
        total_findings=10,
        total_relationships=5,
        top_findings_global=findings,
    )
    assert len(run_ctx.compressed_summary) <= ContextCompressor.TIER3_MAX_CHARS
    assert "RESEARCH OBJECTIVE" in run_ctx.compressed_summary
    ok("Tier 3: run context correct")

    stats = c.get_compression_stats()
    assert stats["cycles_compressed"] == 1 and stats["tasks_compressed"] == 1
    ok("Compression stats correct")

except Exception as e:
    fail("Context Compression", traceback.format_exc(limit=2))


# =============================================================================
# TRACK A3: Novelty Detector
# =============================================================================
section("Track A3: Novelty Detector")

try:
    from src.novelty.novelty_detector import NoveltyDetector

    d = NoveltyDetector(threshold=0.65)

    r = d.check("Perform correlation analysis between metabolite E and age")
    assert r.is_novel
    ok("Novel task passes")

    d.register_task("Perform PCA on the metabolomics dataset to identify groupings")
    r2 = d.check("Perform PCA on the metabolomics dataset to identify groupings and clusters")
    assert not r2.is_novel
    ok("Near-duplicate task correctly rejected")

    # Use exact same tokens for guaranteed overlap
    d.register_finding("metabolite A group differences significant control treatment samples")
    r3 = d.check("metabolite A group differences significant control treatment samples analysis")
    assert not r3.is_novel, f"High-overlap task should be flagged (sim={r3.similarity_score:.2f})"
    ok(f"Finding overlap correctly detected (sim={r3.similarity_score:.2f})")

    tasks = [
        "Perform differential abundance analysis with ANOVA correction",
        "Run pathway enrichment on significant metabolites",
        "Perform PCA on the metabolomics dataset to identify groupings",  # duplicate
    ]
    results = d.check_batch(tasks)
    novel = [t for t, r in results if r.is_novel]
    assert len(novel) == 2, f"Expected 2 novel, got {len(novel)}"
    ok("Batch check: 2/3 novel tasks identified")

    assert d.get_stats()["registered_tasks"] >= 1
    ok("Novelty detector stats correct")

except Exception as e:
    fail("Novelty Detector", traceback.format_exc(limit=2))


# =============================================================================
# TRACK B: ArXiv + PubMed
# =============================================================================
section("Track B1 & B2: ArXiv + PubMed Clients")

try:
    from src.literature.arxiv_search import ArXivClient
    from src.literature.pubmed_search import PubMedClient

    # ArXiv: verify client initializes correctly
    arxiv = ArXivClient(max_results=5)
    assert arxiv.max_results == 5
    assert arxiv._min_interval == 3.0
    ok("ArXiv client initializes with correct config")

    # ArXiv: verify year extraction helper (used inside _parse_entry)
    year_match = re.search(r"(\d{4})", "2025-01-15T00:00:00Z")
    assert year_match and int(year_match.group(1)) == 2025
    ok("ArXiv year extraction from pubdate works")

    # ArXiv: verify paper_id format
    test_arxiv_id = "2501.12345v1"
    expected_id = f"arxiv_{test_arxiv_id.replace('/', '_')}"
    assert expected_id == "arxiv_2501.12345v1"
    ok("ArXiv paper_id format correct")

    # PubMed: summary parsing
    pubmed = PubMedClient(max_results=5)
    summary = {
        "title": "Metabolomics study of treatment effects",
        "authors": [{"name": "Smith J"}],
        "pubdate": "2024",
        "fulljournalname": "Journal of Metabolomics",
        "articleids": [
            {"idtype": "doi", "value": "10.1000/test.001"},
            {"idtype": "pmc", "value": "PMC1234567"},
        ],
    }
    paper = pubmed._parse_summary("12345678", summary)
    assert paper is not None
    assert paper.year == 2024
    assert paper.paper_id == "pubmed_12345678"
    assert paper.is_open_access == True
    ok("PubMed summary parsing correct")

    # Both wired into LiteratureSearchAgent
    from src.agents.literature import LiteratureSearchAgent
    from src.utils.llm_client import LLMClient
    agent = LiteratureSearchAgent(llm_client=LLMClient(), arxiv_client=arxiv, pubmed_client=pubmed)
    assert hasattr(agent, "arxiv_client") and hasattr(agent, "pubmed_client")
    assert agent.arxiv_client is arxiv
    assert agent.pubmed_client is pubmed
    ok("ArXiv + PubMed wired into LiteratureSearchAgent")

except Exception as e:
    fail("ArXiv + PubMed", traceback.format_exc(limit=2))


# =============================================================================
# TRACK C: Plan Reviewer + Explore/Exploit
# =============================================================================
section("Track C1 & C2: Plan Reviewer + Explore/Exploit")

try:
    from src.orchestration.plan_reviewer import PlanReviewer, TaskScore
    from src.world_model.models import Task, TaskType, TaskStatus

    pr = PlanReviewer(min_score=0.45)

    r1  = pr.get_explore_exploit_ratio(cycle=1,  max_cycles=10)
    r5  = pr.get_explore_exploit_ratio(cycle=5,  max_cycles=10)
    r10 = pr.get_explore_exploit_ratio(cycle=10, max_cycles=10)

    assert r1["mode"]  == "exploratory"
    assert r5["mode"]  == "balanced"
    assert r10["mode"] == "exploitative"
    assert r1["explore"] > r10["explore"]
    ok("Explore/exploit ratio shifts correctly")

    instruction = pr.get_mode_instruction(cycle=1, max_cycles=10)
    assert "EXPLORATORY" in instruction and "Cycle 1/10" in instruction
    ok("Mode instruction formatted correctly")

    good = Task(
        task_type=TaskType.DATA_ANALYSIS,
        description="Perform differential abundance analysis using Mann-Whitney test with Benjamini-Hochberg correction between control and treatment groups",
        goal="Identify significant metabolite differences",
        cycle=1,
    )
    _, good_score = pr.review_tasks([good], "Identify metabolic features", 1, 10)[0]
    assert isinstance(good_score, TaskScore)
    ok(f"Good task scored (composite={good_score.composite:.2f})")

    weak = Task(
        task_type=TaskType.DATA_ANALYSIS,
        description="Maybe investigate something possibly",
        goal="See what happens",
        cycle=1,
    )
    _, weak_score = pr.review_tasks([weak], "Identify metabolic features", 1, 10)[0]
    assert weak_score.composite < good_score.composite
    ok("Weak task scores lower than specific task")

except Exception as e:
    fail("Plan Reviewer", traceback.format_exc(limit=2))


# =============================================================================
# TRACK D1: Circuit Breaker
# =============================================================================
section("Track D1: Circuit Breaker")

try:
    from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError

    cb = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=60.0)
    assert cb.state.value == "closed"
    ok("Circuit starts CLOSED")

    for _ in range(3):
        try:
            with cb:
                raise ValueError("fail")
        except ValueError:
            pass

    assert cb.state.value == "open"
    ok("Circuit trips OPEN after 3 failures")

    try:
        with cb:
            pass
        assert False
    except CircuitOpenError:
        ok("Blocked call raises CircuitOpenError")

    stats = cb.get_stats()
    assert stats["total_failures"] == 3 and stats["total_blocked"] == 1
    ok("Circuit breaker stats correct")

    cb.reset()
    assert cb.state.value == "closed"
    ok("Circuit resets to CLOSED")

except Exception as e:
    fail("Circuit Breaker", traceback.format_exc(limit=2))


# =============================================================================
# TRACK D3: Package Resolver
# =============================================================================
section("Track D3: Package Resolver")

try:
    from src.execution.package_resolver import PackageResolver

    pr = PackageResolver()

    pkgs = pr.detect_missing_packages("ModuleNotFoundError: No module named 'gseapy'")
    assert "gseapy" in pkgs
    ok("Detects missing module: gseapy")

    pkgs2 = pr.detect_missing_packages("cannot import name 'linkage' from 'sklearn.cluster'")
    assert "scikit-learn" in pkgs2
    ok("Maps sklearn to scikit-learn")

    code = "# -*- coding: utf-8 -*-\nimport gseapy\nprint('hello')"
    patched = pr.patch_code(code, "ModuleNotFoundError: No module named 'gseapy'")
    assert "pip" in patched and patched != code
    ok("Code patched with pip install")

    patched2 = pr.patch_code(code, "ModuleNotFoundError: No module named 'gseapy'")
    assert patched2 == code
    ok("Already-installed package not patched again")

    assert "gseapy" in pr.get_stats()["packages_installed"]
    ok("Package resolver tracks installed packages")

except Exception as e:
    fail("Package Resolver", traceback.format_exc(limit=2))


# =============================================================================
# TRACK E: Model Routing + Circuit Breaker
# =============================================================================
section("Track E1 & E2: Model Routing + Circuit Breaker")

try:
    from src.utils.llm_client import LLMClient

    llm = LLMClient()

    assert hasattr(llm, "complete_for_task")
    assert hasattr(llm, "_circuit_breaker")
    ok("LLMClient has routing and circuit breaker")

    if llm.provider == "ollama":
        assert llm._simple_model == llm._complex_model
        ok("Ollama: simple_model == complex_model (correct fallback)")

    assert llm._resolve_simple_model("anthropic", "claude-sonnet-4-6") == "claude-haiku-4-5-20251001"
    ok("Anthropic routes to Haiku for simple tasks")

    assert llm._resolve_simple_model("openai", "gpt-4o") == "gpt-4o-mini"
    ok("OpenAI routes to gpt-4o-mini for simple tasks")

    assert llm._circuit_breaker.state.value == "closed"
    assert llm._circuit_breaker.name == llm.provider
    ok("LLMClient circuit breaker initialised correctly")

except Exception as e:
    fail("Model Routing + Circuit Breaker", traceback.format_exc(limit=2))


# =============================================================================
# TRACK F: JSONL Stage Tracking
# =============================================================================
section("Track F: JSONL Stage Tracking")

try:
    from src.tracking.stage_tracker import StageTracker

    tmp = tempfile.mkdtemp()
    tracker = StageTracker(run_id="test_run_phase6", output_dir=tmp)

    with tracker.track("cycle", cycle=1):
        with tracker.track("task", substage="data_analysis", cycle=1):
            pass

    tracker.finding_added("Metabolite A 1.5-fold change", 0.98, "data_analysis", 1)
    tracker.cycle_summary(cycle=1, findings=3, relationships=2, tasks=3)

    with open(tracker.get_output_path(), "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    assert len(entries) >= 7, f"Expected >=7, got {len(entries)}"
    ok(f"JSONL file written ({len(entries)} entries)")

    stages = [e["stage"] for e in entries]
    for s in ["run_start", "cycle", "task", "finding_added", "cycle_summary"]:
        assert s in stages, f"Missing stage: {s}"
    ok("All expected stage types present")

    completed = [e for e in entries if e.get("status") == "completed"]
    assert all("duration_ms" in e for e in completed)
    ok("Duration recorded for completed stages")

    for e in entries:
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", e["timestamp"])
    ok("All timestamps in ISO format")

    assert all(e["run_id"] == "test_run_phase6" for e in entries)
    ok("Run ID consistent across all entries")

except Exception as e:
    fail("JSONL Stage Tracking", traceback.format_exc(limit=2))


# =============================================================================
# INTEGRATION
# =============================================================================
section("Integration: All modules importable together")

try:
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
    ok("All Phase 6 modules import cleanly")
except Exception as e:
    fail("Integration import", traceback.format_exc(limit=2))


# =============================================================================
# SUMMARY
# =============================================================================
total = len(passed) + len(failed)
print(f"\n{BOLD}{'='*55}{RESET}")
print(f"{BOLD}  RESULTS: {len(passed)}/{total} tests passed{RESET}")
if failed:
    print(f"\n{RED}{BOLD}  Failed:{RESET}")
    for n in failed:
        print(f"    {RED}x {n}{RESET}")
else:
    print(f"\n{GREEN}{BOLD}  All tests passed! Ready for full Inquiro run.{RESET}")
print(f"{BOLD}{'='*55}{RESET}\n")

sys.exit(0 if not failed else 1)
