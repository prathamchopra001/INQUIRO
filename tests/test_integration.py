"""
Integration tests for the full Inquiro pipeline.

Run with:
    $env:PYTHONPATH="."; python tests/test_integration.py
"""

import logging
import sys
from pathlib import Path

# Set up logging so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def test_orchestrator_standalone():
    """
    Test OrchestratorAgent in isolation.
    
    Verifies:
    - generate_tasks() returns valid Task objects
    - check_completion() handles edge cases correctly
    - rank_discoveries() works on an empty world model
    """
    print("\n" + "="*60)
    print("TEST 1: OrchestratorAgent standalone")
    print("="*60)

    from src.utils.llm_client import LLMClient
    from src.world_model.world_model import WorldModel
    from src.agents.orchestrator import OrchestratorAgent
    from src.world_model.models import TaskType

    # Use a fresh in-memory-style test database
    wm = WorldModel(db_path="./data/test_orchestrator.db")
    llm = LLMClient()
    orchestrator = OrchestratorAgent(llm_client=llm, world_model=wm)

    objective = "Identify the key environmental factors affecting perovskite solar cell efficiency"

    # --- Test generate_tasks() ---
    print("\n[1a] Testing generate_tasks()...")
    tasks = orchestrator.generate_tasks(
        objective=objective,
        cycle=1,
        num_tasks=3,  # Keep it small for the test
    )

    assert len(tasks) > 0, "Should generate at least one task"
    assert all(hasattr(t, "task_type") for t in tasks), "Tasks must have task_type"
    assert all(t.description for t in tasks), "Tasks must have descriptions"
    
    data_tasks = [t for t in tasks if t.task_type == TaskType.DATA_ANALYSIS]
    lit_tasks = [t for t in tasks if t.task_type == TaskType.LITERATURE_SEARCH]
    
    print(f"  ✓ Generated {len(tasks)} tasks: {len(data_tasks)} data, {len(lit_tasks)} literature")
    for t in tasks:
        print(f"    [{t.task_type}] {t.description[:70]}...")

    # --- Test check_completion() with too few findings ---
    print("\n[1b] Testing check_completion() with empty world model...")
    is_done = orchestrator.check_completion(
        objective=objective,
        cycles_completed=1,
        max_cycles=10,
    )
    assert not is_done, "Should NOT be complete with 0 findings"
    print(f"  ✓ Correctly returned False (no findings yet)")

    # --- Test check_completion() at max cycles ---
    print("\n[1c] Testing check_completion() at max cycles...")
    is_done = orchestrator.check_completion(
        objective=objective,
        cycles_completed=10,
        max_cycles=10,
    )
    assert is_done, "Should be complete at max cycles"
    print(f"  ✓ Correctly returned True (max cycles reached)")

    # --- Test rank_discoveries() on empty world model ---
    print("\n[1d] Testing rank_discoveries() on empty world model...")
    ranked = orchestrator.rank_discoveries(objective=objective)
    assert ranked == [], "Should return empty list when no findings"
    print(f"  ✓ Correctly returned empty list")

    print("\n✅ TEST 1 PASSED")
    return True


def test_report_generator_standalone():
    """
    Test ReportGenerator in isolation with manually seeded findings.
    
    Verifies:
    - Report generates without crashing
    - Output file is created on disk
    - Report contains expected sections
    """
    print("\n" + "="*60)
    print("TEST 2: ReportGenerator standalone")
    print("="*60)

    from src.utils.llm_client import LLMClient
    from src.world_model.world_model import WorldModel
    from src.reports.generator import ReportGenerator

    # Seed a world model with some fake findings
    wm = WorldModel(db_path="./data/test_report.db")
    llm = LLMClient()

    objective = "Identify key factors affecting perovskite solar cell efficiency"

    print("\n[2a] Seeding world model with test findings...")
    fid1 = wm.add_finding(
        claim="Thermal annealing humidity above 60% reduces PCE by 15% on average",
        finding_type="data_analysis",
        source={"type": "notebook", "path": "analysis_001.ipynb", "cell": 3},
        cycle=1,
        confidence=0.87,
        tags=["humidity", "PCE", "thermal"],
        evidence="Correlation analysis across 45 samples showed r=-0.71, p<0.001",
    )

    fid2 = wm.add_finding(
        claim="Solvent partial pressure during spin coating is the dominant predictor of efficiency",
        finding_type="data_analysis",
        source={"type": "notebook", "path": "analysis_002.ipynb", "cell": 7},
        cycle=2,
        confidence=0.79,
        tags=["SPP", "spin-coating", "efficiency"],
        evidence="SHAP analysis ranked SPP highest among all features",
    )

    fid3 = wm.add_finding(
        claim="Prior literature confirms humidity as a critical degradation factor in perovskite cells",
        finding_type="literature",
        source={"type": "paper", "paper_id": "abc123", "doi": "10.1038/s41560-021-00815-y", "title": "Humidity effects on perovskite stability"},
        cycle=2,
        confidence=0.91,
        tags=["humidity", "literature", "degradation"],
        evidence="Meta-analysis of 12 studies consistently shows humidity correlation",
    )

    # Add a supporting relationship
    wm.add_relationship(
        from_id=fid3,
        to_id=fid1,
        relationship_type="supports",
        strength=0.85,
        reasoning="Literature finding validates the data analysis result on humidity",
    )

    print(f"  ✓ Seeded 3 findings and 1 relationship")

    # --- Generate report ---
    print("\n[2b] Generating report...")
    generator = ReportGenerator(
        llm_client=llm,
        world_model=wm,
        output_dir="./outputs/test",
        top_n_discoveries=3,
    )

    report_path = generator.generate_report(
        objective=objective,
        cycles_completed=2,
    )

    # Verify file was created
    assert Path(report_path).exists(), f"Report file not found: {report_path}"
    content = Path(report_path).read_text(encoding="utf-8")
    
    assert "# INQUIRO Discovery Report" in content, "Report missing header"
    assert objective[:30] in content, "Report missing objective"
    assert "Discovery 1" in content, "Report missing discovery sections"

    print(f"  ✓ Report created: {report_path}")
    print(f"  ✓ Report size: {len(content)} characters")
    print(f"\n--- REPORT PREVIEW (first 500 chars) ---")
    print(content[:500])
    print("--- END PREVIEW ---")

    print("\n✅ TEST 2 PASSED")
    return True


def test_inquiro_initialization():
    """
    Test that Inquiro initializes cleanly without running any cycles.
    
    Verifies all components initialize without errors.
    """
    print("\n" + "="*60)
    print("TEST 3: Inquiro initialization")
    print("="*60)

    from src.core.inquiro import Inquiro

    print("\n[3a] Initializing Inquiro...")
    inquiro = Inquiro(
        objective="Test initialization of all components",
        data_path="./data/sample_metabolomics.csv",  # Must exist!
        max_cycles=5,
        num_tasks_per_cycle=3,
        db_path="./data/test_inquiro_init.db",
        output_dir="./outputs/test",
    )

    assert inquiro.orchestrator is not None
    assert inquiro.data_agent is not None
    assert inquiro.literature_agent is not None
    assert inquiro.report_generator is not None
    assert inquiro.world_model is not None

    print("  ✓ All components initialized")
    print("  ✓ Orchestrator ready")
    print("  ✓ Data Agent ready")
    print("  ✓ Literature Agent ready")
    print("  ✓ Report Generator ready")
    print("  ✓ World Model ready")

    print("\n✅ TEST 3 PASSED")
    return True


if __name__ == "__main__":
    results = []

    try:
        results.append(("OrchestratorAgent standalone", test_orchestrator_standalone()))
    except Exception as e:
        logger.error(f"TEST 1 FAILED: {e}", exc_info=True)
        results.append(("OrchestratorAgent standalone", False))

    try:
        results.append(("ReportGenerator standalone", test_report_generator_standalone()))
    except Exception as e:
        logger.error(f"TEST 2 FAILED: {e}", exc_info=True)
        results.append(("ReportGenerator standalone", False))

    try:
        results.append(("Inquiro initialization", test_inquiro_initialization()))
    except Exception as e:
        logger.error(f"TEST 3 FAILED: {e}", exc_info=True)
        results.append(("Inquiro initialization", False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}  {name}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("🎉 All tests passed!" if all_passed else "⚠️  Some tests failed"))
    sys.exit(0 if all_passed else 1)