"""
INQUIRO Phase 2 - Day 10: Full Pipeline Demo
============================================

This script tests the complete Data Analysis Agent pipeline:
1. Preview data
2. Plan analysis (LLM)
3. Generate code (LLM)
4. Execute in Docker sandbox
5. Extract findings (LLM)
6. Save Jupyter notebook

Run from project root:
    python tests/test_data_agent_demo.py
"""

from src.utils.llm_client import LLMClient
from src.execution.docker_executor import DockerExecutor
from src.execution.notebook_manager import NotebookManager
from src.agents.data_analysis import DataAnalysisAgent
from pathlib import Path


def main():
    print("=" * 60)
    print("INQUIRO Phase 2 - Data Analysis Agent Demo")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────
    print("\n[1/7] Initializing components...")
    
    llm = LLMClient()
    executor = DockerExecutor()
    notebook_mgr = NotebookManager()
    agent = DataAnalysisAgent(
        llm_client=llm,
        executor=executor,
        notebook_manager=notebook_mgr
    )
    print("  ✓ LLM Client ready")
    print("  ✓ Docker Executor ready")
    print("  ✓ Notebook Manager ready")
    print("  ✓ Data Analysis Agent ready")

    # ── Define the task ────────────────────────────────────
    data_path = "./data/sample_metabolomics.csv"
    
    # Verify dataset exists
    if not Path(data_path).exists():
        print(f"\n  ✗ ERROR: Dataset not found at {data_path}")
        print("    Please ensure sample_metabolomics.csv is in the data/ folder.")
        return
    
    print(f"\n[2/7] Dataset found: {data_path}")

    task = {
        "description": "Compare metabolite levels between treatment and control groups",
        "goal": "Identify which metabolites are significantly different between treatment and control groups using statistical tests",
        "cycle": 1,
    }
    
    objective = "Identify metabolic changes associated with the treatment condition"
    
    print(f"  Task: {task['description']}")
    print(f"  Objective: {objective}")

    # ── Preview data ───────────────────────────────────────
    print("\n[3/7] Previewing dataset...")
    preview = agent._preview_data(data_path)
    print(f"  Preview (first 300 chars):\n{preview[:300]}...")

    # ── Run the full pipeline ──────────────────────────────
    print("\n[4/7] Running full analysis pipeline...")
    print("  This will: Plan → Generate Code → Execute → Extract Findings → Save Notebook")
    print("  (This may take a minute...)\n")
    
    result = agent.execute(
        task=task,
        data_path=data_path,
        objective=objective,
        world_model_summary="No prior findings. This is the first analysis cycle."
    )

    # ── Display results ────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n[5/7] Execution Summary:")
    print(f"  Success:  {result['execution_result'].success}")
    print(f"  Attempts: {result['attempts']}")
    print(f"  Time:     {result['execution_result'].execution_time:.1f}s")
    
    if result['execution_result'].figures:
        print(f"  Figures:  {result['execution_result'].figures}")

    print(f"\n[6/7] Analysis Plan:")
    print(f"  {result['plan'][:500]}...")

    print(f"\n[7/7] Extracted Findings:")
    if result['findings']:
        for i, finding in enumerate(result['findings'], 1):
            print(f"\n  Finding {i}:")
            print(f"    Claim:      {finding.get('claim', 'N/A')}")
            print(f"    Confidence: {finding.get('confidence', 'N/A')}")
            print(f"    Evidence:   {finding.get('evidence', 'N/A')[:100]}")
            print(f"    Tags:       {finding.get('tags', [])}")
    else:
        print("  No findings extracted (code may have failed).")
        print(f"  Stderr: {result['execution_result'].stderr[:300]}")

    print(f"\n  Notebook saved: {result['notebook_path']}")
    print(f"  Code cell index: {result['code_cell_index']}")

    # ── Show what a Finding.source would look like ─────────
    if result['findings']:
        print(f"\n{'=' * 60}")
        print("TRACEABILITY EXAMPLE")
        print(f"{'=' * 60}")
        print(f"  When storing this in the World Model, the source would be:")
        print(f"  source = {{")
        print(f"      'type': 'notebook',")
        print(f"      'path': '{result['notebook_path']}',")
        print(f"      'cell': {result['code_cell_index']}")
        print(f"  }}")

    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
