# -*- coding: utf-8 -*-
"""
INQUIRO Research Wizard — Interactive CLI for configuring and running research.

This is the main entry point for INQUIRO. It guides users through:
1. Research objective
2. Research mode (data/literature/full)
3. Model configuration (smart routing/local/cloud)
4. Execution mode (sequential/swarm) [future]

"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Literal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class ResearchConfig:
    """Configuration collected from the wizard."""
    objective: str
    research_mode: Literal["data", "literature", "full"]
    data_path: Optional[str] = None
    search_datasets: bool = False
    model_mode: Literal["hybrid", "local_smart", "local_simple", "cloud"] = "hybrid"
    cloud_provider: Optional[str] = None
    local_model: Optional[str] = None
    local_code_model: Optional[str] = None
    local_reasoning_model: Optional[str] = None
    execution_mode: Literal["sequential", "swarm"] = "sequential"
    research_strategy: Literal["standard", "question_driven", "hybrid"] = "standard"
    use_adaptive_decomposition: bool = False  # Hierarchical pillar-based decomposition
    max_cycles: int = 5
    tasks_per_cycle: int = 3
    resume_from: Optional[str] = None  # Previous run ID to resume from
    generate_latex: bool = False  # Compile report to LaTeX/PDF
    latex_template: str = "plain"  # LaTeX template: plain, arxiv, neurips, ieee


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print INQUIRO header."""
    print("\n" + "=" * 60)
    print("""
    ██╗███╗   ██╗ ██████╗ ██╗   ██╗██╗██████╗  ██████╗ 
    ██║████╗  ██║██╔═══██╗██║   ██║██║██╔══██╗██╔═══██╗
    ██║██╔██╗ ██║██║   ██║██║   ██║██║██████╔╝██║   ██║
    ██║██║╚██╗██║██║▄▄ ██║██║   ██║██║██╔══██╗██║   ██║
    ██║██║ ╚████║╚██████╔╝╚██████╔╝██║██║  ██║╚██████╔╝
    ╚═╝╚═╝  ╚═══╝ ╚══▀▀═╝  ╚═════╝ ╚═╝╚═╝  ╚═╝ ╚═════╝
    """)
    print("           Autonomous Scientific Research System")
    print("=" * 60)


def print_step(step_num: int, total: int, title: str):
    """Print step header."""
    print(f"\n{'─' * 60}")
    print(f"  STEP {step_num}/{total}: {title}")
    print(f"{'─' * 60}\n")


def get_choice(prompt: str, options: list, default: int = None) -> int:
    """Get user choice from numbered options."""
    for i, option in enumerate(options, 1):
        marker = " (default)" if default == i else ""
        print(f"  [{i}] {option}{marker}")
    
    print()
    while True:
        try:
            if default:
                raw = input(f"  Enter choice [1-{len(options)}] (default={default}): ").strip()
                if not raw:
                    return default
                choice = int(raw)
            else:
                choice = int(input(f"  Enter choice [1-{len(options)}]: ").strip())
            
            if 1 <= choice <= len(options):
                return choice
            print(f"  ⚠️  Please enter a number between 1 and {len(options)}")
        except ValueError:
            print(f"  ⚠️  Please enter a valid number")


def get_input(prompt: str, default: str = None, required: bool = True) -> str:
    """Get text input from user."""
    if default:
        value = input(f"  {prompt} (default: {default}): ").strip()
        return value if value else default
    else:
        while True:
            value = input(f"  {prompt}: ").strip()
            if value or not required:
                return value
            print("  ⚠️  This field is required")


def get_multiline_input(prompt: str) -> str:
    """Get multiline text input (end with empty line)."""
    print(f"  {prompt}")
    print("  (Press Enter twice to finish)\n")
    
    lines = []
    empty_count = 0
    while empty_count < 1:
        line = input("  > ")
        if not line:
            empty_count += 1
        else:
            empty_count = 0
            lines.append(line)
    
    return "\n".join(lines)


# =============================================================================
# WIZARD STEPS
# =============================================================================

def list_previous_runs(limit: int = 10) -> list:
    """List previous runs that can be resumed.
    
    Returns:
        List of tuples: (run_id, objective_snippet, findings_count, timestamp)
    """
    runs = []
    data_dir = Path("./data")
    
    if not data_dir.exists():
        return runs
    
    # Find all run directories
    for run_dir in sorted(data_dir.glob("run_*"), reverse=True):
        if not run_dir.is_dir():
            continue
            
        db_path = run_dir / "world_model.db"
        if not db_path.exists():
            continue
        
        try:
            from src.world_model.world_model import WorldModel
            wm = WorldModel(db_path=str(db_path))
            stats = wm.get_statistics()
            findings_count = stats.get("total_findings", 0)
            questions_count = stats.get("total_questions", 0)
            
            # Extract timestamp from run_id (format: run_YYYYMMDD_HHMMSS_slug)
            parts = run_dir.name.split("_")
            if len(parts) >= 3:
                date_str = parts[1]
                time_str = parts[2]
                slug = "_".join(parts[3:]) if len(parts) > 3 else ""
                
                # Format nicely
                try:
                    timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}"
                except:
                    timestamp = "unknown"
            else:
                timestamp = "unknown"
                slug = run_dir.name
            
            wm.close()
            runs.append((run_dir.name, slug[:40], findings_count, questions_count, timestamp))
            
            if len(runs) >= limit:
                break
                
        except Exception as e:
            continue
    
    return runs


def step_resume() -> Optional[str]:
    """Ask if user wants to resume from a previous run.
    
    Returns:
        run_id to resume from, or None for fresh start
    """
    runs = list_previous_runs(limit=8)
    
    if not runs:
        return None  # No previous runs, skip this step
    
    print_step(0, 4, "RESUME PREVIOUS RUN?")
    
    print("  Found previous research runs:\n")
    
    options = ["🆕 Start fresh (new research)"]
    for run_id, slug, findings, questions, timestamp in runs:
        options.append(f"📂 {slug}... ({findings} findings, {questions} questions, {timestamp})")
    
    choice = get_choice("Select an option", options, default=1)
    
    if choice == 1:
        return None  # Fresh start
    else:
        run_id = runs[choice - 2][0]  # -2 because choice 1 = fresh, runs[0] = choice 2
        print(f"\n  ✅ Will resume from: {run_id}")
        return run_id


def step_objective() -> str:
    """Step 1: Get research objective."""
    from src.core.query_classifier import QueryClassifier, QueryType
    
    print_step(1, 4, "RESEARCH OBJECTIVE")
    
    print("  What do you want to research?\n")
    print("  Examples:")
    print("  • Predicting 30-day hospital readmission in diabetic patients")
    print("  • Factors affecting renewable energy adoption in urban areas")
    print("  • Machine learning approaches for protein structure prediction")
    print()
    
    while True:
        objective = get_multiline_input("Enter your research objective")
        
        print("\n  🔍 Analyzing your query...")
        
        # Use LLM to classify the query
        classifier = QueryClassifier()
        result = classifier.classify(objective)
        
        if result.query_type == QueryType.CALCULATION:
            print("\n" + "=" * 60)
            print("  🧮 CALCULATION RESULT")
            print("=" * 60)
            print(f"\n  {result.direct_answer}\n")
            print("=" * 60)
            input("\n  Press Enter to exit...")
            return None  # Exit INQUIRO
        
        if result.query_type == QueryType.SIMPLE_QUESTION:
            # LLM answered directly - no research needed
            print("\n" + "=" * 60)
            print("  📚 ANSWER")
            print("=" * 60)
            print(f"\n  {result.direct_answer}\n")
            print("=" * 60)
            input("\n  Press Enter to exit...")
            return None  # Exit INQUIRO
        
        # It's a research query - proceed
        print("  ✓ Research objective confirmed\n")
        break
    
    if len(objective) < 20:
        print("\n  💡 Tip: A more detailed objective helps INQUIRO generate better research tasks.")
        expand = input("  Would you like to expand it? [y/N]: ").strip().lower()
        if expand == 'y':
            print("\n  Add more details (research questions, scope, expected outcomes):")
            additional = get_multiline_input("Additional details")
            objective = objective + "\n\n" + additional
    
    return objective


def step_research_mode() -> tuple:
    """Step 2: Get research mode."""
    print_step(2, 4, "RESEARCH MODE")
    
    print("  How should INQUIRO conduct this research?\n")
    
    options = [
        "📊 Data Analysis + Literature — Full research with datasets (recommended)",
        "📚 Literature Only — Paper synthesis, no data analysis",
        "🔢 Data Analysis Only — Focus on dataset analysis, skip literature",
    ]
    
    choice = get_choice("Select research mode", options, default=1)
    
    mode_map = {1: "full", 2: "literature", 3: "data"}
    mode = mode_map[choice]
    
    data_path = None
    search_datasets = False
    
    if mode in ["full", "data"]:
        print(f"\n  {'─' * 40}")
        print("  DATA SOURCE")
        print(f"  {'─' * 40}\n")
        
        data_options = [
            "🔍 Search public datasets (HuggingFace, Kaggle, OpenML)",
            "📁 I have a dataset (provide path)",
        ]
        
        data_choice = get_choice("Where should we get data?", data_options, default=1)
        
        if data_choice == 1:
            search_datasets = True
            print("\n  ✅ INQUIRO will search for relevant public datasets")
        else:
            data_path = get_input("Enter dataset path (CSV, Excel, or folder)")
            if data_path and not Path(data_path).exists():
                print(f"\n  ⚠️  Warning: Path '{data_path}' doesn't exist")
                confirm = input("  Continue anyway? [y/N]: ").strip().lower()
                if confirm != 'y':
                    data_path = None
                    search_datasets = True
                    print("  → Falling back to dataset search")
    
    return mode, data_path, search_datasets


def step_model_config() -> tuple:
    """Step 3: Get model configuration."""
    print_step(3, 4, "MODEL CONFIGURATION")
    
    print("  How should INQUIRO run AI models?\n")
    
    options = [
        "🎯 Hybrid Routing — Local code + Cloud reasoning (recommended)\n"
        "       • Code generation: Local (qwen2.5-coder:7b, free)\n"
        "       • Reasoning/Reports: Cloud API (Gemini)\n"
        "       • Best balance of cost and quality",
        
        "🏠 Fully Local (Smart) — Multiple specialized local models\n"
        "       • Code: qwen2.5-coder:7b\n"
        "       • Reasoning: qwen3:8b or gemma3:12b\n"
        "       • 100% free, 100% private",
        
        "🏠 Fully Local (Simple) — Single local model for everything\n"
        "       • One model handles all tasks\n"
        "       • Simpler setup, less VRAM swapping",
        
        "☁️  Cloud API Only — Use a single cloud provider\n"
        "       • Best quality, fastest\n"
        "       • Requires API key",
    ]
    
    choice = get_choice("Select model configuration", options, default=1)
    
    mode_map = {1: "hybrid", 2: "local_smart", 3: "local_simple", 4: "cloud"}
    model_mode = mode_map[choice]
    
    cloud_provider = None
    local_model = None
    local_code_model = None
    local_reasoning_model = None
    
    if model_mode == "local_smart":
        print(f"\n  {'─' * 40}")
        print("  LOCAL SMART ROUTING CONFIGURATION")
        print(f"  {'─' * 40}\n")
        
        available_models = _get_ollama_models()
        
        if available_models:
            print("  Available models on your system:")
            for model in available_models[:8]:
                print(f"    • {model}")
            if len(available_models) > 8:
                print(f"    ... and {len(available_models) - 8} more")
            print()
        
        print("  Configure specialized models:")
        print("  (Press Enter to use defaults)\n")
        
        local_code_model = get_input("Code model", default="qwen2.5-coder:7b")
        local_reasoning_model = get_input("Reasoning model", default="qwen3:8b")
        
        print(f"\n  ✅ Local smart routing configured:")
        print(f"     • Code tasks → {local_code_model}")
        print(f"     • Reasoning → {local_reasoning_model}")
    
    elif model_mode == "local_simple":
        print(f"\n  {'─' * 40}")
        print("  LOCAL MODEL SELECTION")
        print(f"  {'─' * 40}\n")
        
        available_models = _get_ollama_models()
        
        if available_models:
            print("  Available models on your system:")
            for i, model in enumerate(available_models[:6], 1):
                print(f"  [{i}] {model}")
            
            if len(available_models) > 6:
                print(f"  ... and {len(available_models) - 6} more")
            
            print(f"\n  Recommended: qwen3:8b or gemma3:12b")
            local_model = get_input("Enter model name", default="qwen3:8b")
        else:
            print("  ⚠️  No Ollama models found. Make sure Ollama is running.")
            print("  Run: ollama pull qwen3:8b")
            local_model = get_input("Enter model name", default="qwen3:8b")
    
    elif model_mode == "cloud":
        print(f"\n  {'─' * 40}")
        print("  CLOUD PROVIDER SELECTION")
        print(f"  {'─' * 40}\n")
        
        provider_options = [
            "🔷 Google Gemini — gemini-2.0-flash (fast, good quality)",
            "🟣 Anthropic Claude — claude-sonnet-4 (excellent reasoning)",
            "🟢 OpenAI GPT-4 — gpt-4o (versatile)",
            "⚡ Groq — llama-3.3-70b (very fast, free tier)",
        ]
        
        provider_choice = get_choice("Select cloud provider", provider_options, default=1)
        provider_map = {1: "gemini", 2: "anthropic", 3: "openai", 4: "groq"}
        cloud_provider = provider_map[provider_choice]
        
        # Check if API key is set
        key_env_map = {
            "gemini": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
        }
        key_env = key_env_map[cloud_provider]
        
        if not os.getenv(key_env):
            print(f"\n  ⚠️  {key_env} not found in environment")
            print(f"  Please set it in your .env file before running")
    
    return model_mode, cloud_provider, local_model, local_code_model, local_reasoning_model


def step_execution_config() -> tuple:
    """Step 4: Get execution configuration."""
    print_step(4, 4, "EXECUTION SETTINGS")
    
    # For now, swarm mode is future work
    execution_mode = "sequential"
    
    # Research strategy selection
    print("  Select research strategy:\n")
    
    strategy_options = [
        "📋 Standard — Cycle-based task generation (default)",
        "🎯 Question-Driven — Deep search per question with citation chains",
        "🔀 Hybrid — Question-driven first, then cycles for gaps",
    ]
    
    strategy_choice = get_choice("Select research strategy", strategy_options, default=1)
    strategy_map = {1: "standard", 2: "question_driven", 3: "hybrid"}
    research_strategy = strategy_map[strategy_choice]
    
    print("")  # Spacing
    
    # Adaptive decomposition option (for question-driven strategies)
    use_adaptive = False
    if research_strategy in ("question_driven", "hybrid"):
        print("  Question decomposition mode:\n")
        decomp_options = [
            "📋 Standard — 3-5 research questions (default)",
            "🎯 Adaptive — 4-24 questions organized by pillars (scales with complexity)",
        ]
        decomp_choice = get_choice("Select decomposition mode", decomp_options, default=1)
        use_adaptive = (decomp_choice == 2)
        
        if use_adaptive:
            print("\n  ℹ️  Adaptive mode will:")
            print("     • Assess objective complexity")
            print("     • Generate research pillars (high-level themes)")
            print("     • Create 4-24 focused questions based on scope")
            print("     • Produce appropriately-sized reports (8-50 pages)")
        
        print("")  # Spacing
    
    print("  Configure research depth:\n")
    
    cycle_options = [
        "⚡ Quick (2 cycles) — Fast results, less depth (~10 min)",
        "📊 Standard (5 cycles) — Balanced (recommended, ~25 min)",
        "🔬 Deep (10 cycles) — Comprehensive research (~50 min)",
        "🎯 Custom — Set your own parameters",
    ]
    
    choice = get_choice("Select research depth", cycle_options, default=2)
    
    cycle_map = {1: (2, 3), 2: (5, 3), 3: (10, 3)}
    
    if choice in cycle_map:
        max_cycles, tasks_per_cycle = cycle_map[choice]
    else:
        max_cycles = int(get_input("Max cycles", default="5"))
        tasks_per_cycle = int(get_input("Tasks per cycle", default="3"))
    
    return execution_mode, research_strategy, use_adaptive, max_cycles, tasks_per_cycle


def step_output_config() -> tuple:
    """Step 5: Get output format configuration."""
    print(f"\n{'─' * 60}")
    print("  OUTPUT FORMAT")
    print(f"{'─' * 60}\n")
    
    print("  Report format options:\n")
    
    output_options = [
        "📝 Markdown only — Fast, viewable anywhere (default)",
        "📄 Markdown + LaTeX/PDF — Academic paper format (requires LaTeX)",
    ]
    
    choice = get_choice("Select output format", output_options, default=1)
    
    generate_latex = (choice == 2)
    latex_template = "plain"
    
    if generate_latex:
        # Check if LaTeX is available
        try:
            from src.reports.latex_compiler import check_latex_installation
            latex_available = check_latex_installation()
            has_latex = any(latex_available.values())
        except:
            has_latex = False
        
        if not has_latex:
            print("\n  ⚠️  No LaTeX compiler detected on your system.")
            print("     Install TeX Live, MiKTeX, or MacTeX for PDF output.")
            print("     Will generate .tex source file instead.")
        
        print(f"\n  {'─' * 40}")
        print("  LATEX TEMPLATE")
        print(f"  {'─' * 40}\n")
        
        template_options = [
            "📋 Plain — Simple article format (default)",
            "🔬 arXiv — arXiv preprint style",
            "🎓 NeurIPS — NeurIPS/ML conference format",
            "🔌 IEEE — IEEE journal format",
        ]
        
        template_choice = get_choice("Select LaTeX template", template_options, default=1)
        template_map = {1: "plain", 2: "arxiv", 3: "neurips", 4: "ieee"}
        latex_template = template_map[template_choice]
        
        print(f"\n  ✅ Will generate LaTeX output using {latex_template} template")
    
    return generate_latex, latex_template


def _get_ollama_models() -> list:
    """Get list of available Ollama models."""
    try:
        import urllib.request
        import json
        
        from config.settings import settings
        url = f"{settings.llm.ollama_base_url}/api/tags"
        
        with urllib.request.urlopen(url, timeout=3) as response:
            data = json.loads(response.read())
            return [m['name'] for m in data.get('models', [])]
    except:
        return []


# =============================================================================
# CONFIGURATION SUMMARY & CONFIRMATION
# =============================================================================

def show_summary(config: ResearchConfig) -> bool:
    """Show configuration summary and get confirmation."""
    print("\n" + "=" * 60)
    print("  📋 RESEARCH CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"\n  🎯 Objective:")
    for line in config.objective.split('\n')[:3]:
        print(f"     {line[:55]}{'...' if len(line) > 55 else ''}")
    if len(config.objective.split('\n')) > 3:
        print("     ...")
    
    mode_labels = {
        "full": "📊 Full Research (Data + Literature)",
        "literature": "📚 Literature Only",
        "data": "🔢 Data Analysis Only",
    }
    print(f"\n  📂 Research Mode: {mode_labels[config.research_mode]}")
    
    if config.data_path:
        print(f"     Data source: {config.data_path}")
    elif config.search_datasets:
        print(f"     Data source: Search public datasets")
    
    model_labels = {
        "hybrid": "🎯 Hybrid (local code + cloud reasoning)",
        "local_smart": f"🏠 Local Smart ({config.local_code_model} + {config.local_reasoning_model})",
        "local_simple": f"🏠 Local Simple ({config.local_model or 'qwen3:8b'})",
        "cloud": f"☁️  Cloud API ({config.cloud_provider})",
    }
    print(f"\n  🤖 Model Config: {model_labels[config.model_mode]}")
    
    print(f"\n  ⚙️  Execution:")
    print(f"     • Cycles: {config.max_cycles}")
    print(f"     • Tasks per cycle: {config.tasks_per_cycle}")
    print(f"     • Mode: {config.execution_mode.title()}")
    strategy_labels = {
        "standard": "📋 Standard (cycle-based)",
        "question_driven": "🎯 Question-Driven (deep search per question)",
        "hybrid": "🔀 Hybrid (questions first, then cycles)",
    }
    print(f"     • Strategy: {strategy_labels.get(config.research_strategy, config.research_strategy)}")
    if config.use_adaptive_decomposition:
        print(f"     • Decomposition: 🎯 Adaptive (pillar-based, scales with complexity)")
    if config.resume_from:
        print(f"     • Resuming from: 🔄 {config.resume_from}")
    
    # Output format
    print(f"\n  📄 Output Format:")
    if config.generate_latex:
        template_names = {"plain": "Plain article", "arxiv": "arXiv", "neurips": "NeurIPS", "ieee": "IEEE"}
        print(f"     • Format: LaTeX + PDF ({template_names.get(config.latex_template, config.latex_template)})")
    else:
        print(f"     • Format: Markdown")
    
    # Estimate time
    est_minutes = config.max_cycles * config.tasks_per_cycle * 2  # ~2 min per task
    print(f"     • Estimated time: ~{est_minutes} minutes")
    
    print("\n" + "=" * 60)
    
    print("\n  [1] ✅ Start Research")
    print("  [2] ✏️  Edit Settings")
    print("  [3] ❌ Cancel")
    
    choice = get_choice("What would you like to do?", ["Start", "Edit", "Cancel"], default=1)
    
    return choice == 1


# =============================================================================
# MAIN WIZARD
# =============================================================================

def run_wizard() -> Optional[ResearchConfig]:
    """Run the interactive wizard and return configuration."""
    clear_screen()
    print_header()
    
    while True:
        # Step 0: Check for previous runs (optional resume)
        resume_from = step_resume()
        
        # Step 1: Objective
        objective = step_objective()
        
        # Handle early exit (e.g., user just wanted a calculation)
        if objective is None:
            print("\n  👋 Goodbye!")
            return None
        
        # Step 2: Research Mode
        research_mode, data_path, search_datasets = step_research_mode()
        
        # Step 3: Model Config
        model_mode, cloud_provider, local_model, local_code_model, local_reasoning_model = step_model_config()
        
        # Step 4: Execution Config
        execution_mode, research_strategy, use_adaptive, max_cycles, tasks_per_cycle = step_execution_config()
        
        # Step 5: Output Format (LaTeX)
        generate_latex, latex_template = step_output_config()
        
        # Build config
        config = ResearchConfig(
            objective=objective,
            research_mode=research_mode,
            data_path=data_path,
            search_datasets=search_datasets,
            model_mode=model_mode,
            cloud_provider=cloud_provider,
            local_model=local_model,
            local_code_model=local_code_model,
            local_reasoning_model=local_reasoning_model,
            execution_mode=execution_mode,
            research_strategy=research_strategy,
            use_adaptive_decomposition=use_adaptive,
            max_cycles=max_cycles,
            tasks_per_cycle=tasks_per_cycle,
            resume_from=resume_from,
            generate_latex=generate_latex,
            latex_template=latex_template,
        )
        
        # Show summary
        if show_summary(config):
            return config
        else:
            # Check if user wants to edit or cancel
            edit = input("\n  Press Enter to edit settings, or 'q' to quit: ").strip().lower()
            if edit == 'q':
                return None
            clear_screen()
            print_header()


# =============================================================================
# RUN INQUIRO WITH CONFIG
# =============================================================================

def apply_config(config: ResearchConfig):
    """Apply wizard configuration to environment and reload settings."""
    
    # Set research mode
    os.environ['RUN_MODE'] = config.research_mode
    
    # Set model configuration based on mode
    if config.model_mode == "hybrid":
        # Local for code, Cloud for reasoning
        os.environ['ROUTER_ENABLED'] = 'true'
        os.environ['ROUTER_CODE_PROVIDER'] = 'ollama'
        os.environ['ROUTER_CODE_MODEL'] = 'qwen2.5-coder:7b'
        os.environ['ROUTER_STRONG_PROVIDER'] = 'gemini'
        os.environ['ROUTER_STRONG_MODEL'] = 'gemini-2.0-flash'
        os.environ['ROUTER_FAST_PROVIDER'] = 'gemini'
        os.environ['ROUTER_FAST_MODEL'] = 'gemini-2.0-flash'
        os.environ['ROUTER_LOCAL_PROVIDER'] = 'ollama'
        os.environ['ROUTER_LOCAL_MODEL'] = 'qwen3:8b'
        
    elif config.model_mode == "local_smart":
        # Smart routing with ALL local models
        os.environ['ROUTER_ENABLED'] = 'true'
        os.environ['ROUTER_CODE_PROVIDER'] = 'ollama'
        os.environ['ROUTER_CODE_MODEL'] = config.local_code_model or 'qwen2.5-coder:7b'
        os.environ['ROUTER_STRONG_PROVIDER'] = 'ollama'
        os.environ['ROUTER_STRONG_MODEL'] = config.local_reasoning_model or 'qwen3:8b'
        os.environ['ROUTER_FAST_PROVIDER'] = 'ollama'
        os.environ['ROUTER_FAST_MODEL'] = config.local_reasoning_model or 'qwen3:8b'
        os.environ['ROUTER_LOCAL_PROVIDER'] = 'ollama'
        os.environ['ROUTER_LOCAL_MODEL'] = config.local_reasoning_model or 'qwen3:8b'
        
    elif config.model_mode == "local_simple":
        # Single local model for everything
        os.environ['ROUTER_ENABLED'] = 'false'
        os.environ['LLM_PROVIDER'] = 'ollama'
        os.environ['LLM_MODEL'] = config.local_model or 'qwen3:8b'
        
    elif config.model_mode == "cloud":
        # Single cloud provider for everything
        os.environ['ROUTER_ENABLED'] = 'false'
        os.environ['LLM_PROVIDER'] = config.cloud_provider
        
        model_defaults = {
            "gemini": "gemini-2.0-flash",
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
            "groq": "llama-3.3-70b-versatile",
        }
        os.environ['LLM_MODEL'] = model_defaults.get(config.cloud_provider, "")
    
    # Set execution parameters
    os.environ['MAX_CYCLES'] = str(config.max_cycles)
    os.environ['TASKS_PER_CYCLE'] = str(config.tasks_per_cycle)
    
    # CRITICAL: Directly modify the settings object
    # Environment variables only work at import time, but settings is already loaded
    from config.settings import settings
    
    # Apply router settings directly to the singleton
    if config.model_mode == "hybrid":
        settings.router.enabled = True
        settings.router.code_provider = 'ollama'
        settings.router.code_model = 'qwen2.5-coder:7b'
        settings.router.strong_provider = 'gemini'
        settings.router.strong_model = 'gemini-2.0-flash'
        settings.router.fast_provider = 'gemini'
        settings.router.fast_model = 'gemini-2.0-flash'
        settings.router.local_provider = 'ollama'
        settings.router.local_model = 'qwen3:8b'
        
    elif config.model_mode == "local_smart":
        settings.router.enabled = True
        settings.router.code_provider = 'ollama'
        settings.router.code_model = config.local_code_model or 'qwen2.5-coder:7b'
        settings.router.strong_provider = 'ollama'
        settings.router.strong_model = config.local_reasoning_model or 'qwen3:8b'
        settings.router.fast_provider = 'ollama'
        settings.router.fast_model = config.local_reasoning_model or 'qwen3:8b'
        settings.router.local_provider = 'ollama'
        settings.router.local_model = config.local_reasoning_model or 'qwen3:8b'
        
    elif config.model_mode == "local_simple":
        settings.router.enabled = False
        settings.llm.provider = 'ollama'
        settings.llm.model = config.local_model or 'qwen3:8b'
        
    elif config.model_mode == "cloud":
        settings.router.enabled = False
        settings.llm.provider = config.cloud_provider
        model_defaults = {
            "gemini": "gemini-2.0-flash",
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
            "groq": "llama-3.3-70b-versatile",
        }
        settings.llm.model = model_defaults.get(config.cloud_provider, "")
    
    # Apply execution settings
    settings.execution.max_cycles = config.max_cycles
    settings.execution.tasks_per_cycle = config.tasks_per_cycle
    settings.execution.run_mode = config.research_mode
    
    # Print confirmation
    print(f"\n  ✅ Configuration applied:")
    print(f"     Router enabled: {settings.router.enabled}")
    if settings.router.enabled:
        print(f"     Code tier: {settings.router.code_provider}/{settings.router.code_model}")
        print(f"     Strong tier: {settings.router.strong_provider}/{settings.router.strong_model}")
        print(f"     Fast tier: {settings.router.fast_provider}/{settings.router.fast_model}")
    else:
        print(f"     Provider: {settings.llm.provider}/{settings.llm.model}")


def run_inquiro(config: ResearchConfig):
    """Run INQUIRO with the given configuration."""
    
    # Apply configuration to environment
    apply_config(config)
    
    # Setup logging
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"inquiro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filename, encoding="utf-8"),
        ]
    )
    
    # Reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 60)
    print("  🚀 STARTING INQUIRO RESEARCH")
    print("=" * 60)
    print(f"\n  📝 Log file: {log_filename}")
    
    # Import and run INQUIRO
    from src.core.inquiro import Inquiro
    
    # Determine data path
    data_path = None
    if config.data_path:
        data_path = config.data_path
    elif config.search_datasets:
        data_path = None  # Inquiro will search
    elif config.research_mode == "literature":
        data_path = None  # No data needed
    
    # Determine mode
    mode = config.research_mode
    if mode == "full":
        mode = "full"
    elif mode == "literature":
        mode = "literature"
    else:
        mode = "data"
    
    try:
        inquiro = Inquiro(
            objective=config.objective,
            data_path=data_path,
            max_cycles=config.max_cycles,
            num_tasks_per_cycle=config.tasks_per_cycle,
            max_workers=1,  # Sequential to avoid Gemini rate limit bursts
            mode=mode,
            research_strategy=config.research_strategy,
            use_adaptive_decomposition=config.use_adaptive_decomposition,
            resume_from=config.resume_from,
            generate_latex=config.generate_latex,
            latex_template=config.latex_template,
        )
        
        print(f"\n  ✅ INQUIRO initialized")
        print(f"     Run ID: {inquiro.run_id}")
        print(f"     Output: {inquiro.run_output_dir}")
        print(f"     Strategy: {config.research_strategy}")
        if config.use_adaptive_decomposition:
            print(f"     Decomposition: Adaptive (pillar-based)")
        if config.resume_from:
            print(f"     Resuming from: {config.resume_from}")
        
        print("\n" + "=" * 60)
        print("  🔬 RESEARCH IN PROGRESS...")
        print("=" * 60 + "\n")
        
        report_path = inquiro.run()
        
        print("\n" + "=" * 60)
        print("  ✅ RESEARCH COMPLETE!")
        print("=" * 60)
        print(f"\n  📄 Report: {report_path}")
        print(f"  📝 Log: {log_filename}")
        
        # Print statistics
        stats = inquiro.world_model.get_statistics()
        print(f"\n  📊 Statistics:")
        print(f"     • Findings: {stats.get('total_findings', 0)}")
        print(f"     • Relationships: {stats.get('total_relationships', 0)}")
        print(f"     • Cycles: {config.max_cycles}")
        
        # Print LLM usage summary
        try:
            from src.utils.usage_tracker import get_usage_tracker
            tracker = get_usage_tracker()
            if tracker.total_calls > 0:
                print(f"\n  💰 LLM Usage:")
                print(f"     • Total calls: {tracker.total_calls}")
                print(f"     • Total tokens: {tracker.total_tokens:,}")
                if tracker.total_cache_read > 0:
                    efficiency = tracker.cache_efficiency()
                    print(f"     • Cache efficiency: {efficiency:.1%}")
                    savings = tracker.estimated_savings()
                    print(f"     • Est. cost savings: ~{savings['estimated_savings_pct']:.0f}%")
        except Exception:
            pass  # Usage tracking is optional
        
        return report_path
        
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Research interrupted by user")
        print("  Partial results may be available in the output folder")
        return None
        
    except Exception as e:
        logger.exception(f"Research failed: {e}")
        print(f"\n  ❌ Error: {e}")
        print(f"  Check log file for details: {log_filename}")
        raise


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    try:
        config = run_wizard()
        
        if config is None:
            print("\n  👋 Research cancelled. Goodbye!")
            return 0
        
        run_inquiro(config)
        return 0
        
    except KeyboardInterrupt:
        print("\n\n  👋 Interrupted. Goodbye!")
        return 1
    except Exception as e:
        print(f"\n  ❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
