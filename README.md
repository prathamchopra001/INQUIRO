# INQUIRO — Autonomous AI Scientist

![version](https://img.shields.io/badge/version-0.3.0--alpha-blue)
![status](https://img.shields.io/badge/status-alpha-orange)
![tests](https://img.shields.io/badge/tests-154%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> *An autonomous scientific research system that analyzes datasets, searches literature, generates hypotheses, and produces discovery reports — inspired by the INQUIRO paper (arXiv:2511.02824).*

---

## What is INQUIRO?

INQUIRO is an autonomous AI research engine. You give it a dataset and a research question. It works through the problem in iterative cycles — analyzing your data, searching the scientific literature, building a structured knowledge base of findings, and synthesizing everything into a cited discovery report.

It is not a chatbot. It does not wait for instructions. It plans its own research tasks, executes them in parallel, filters out low-quality findings, avoids redundant work, and adapts its strategy as it learns more — all without human intervention.

Under the hood, INQUIRO coordinates three specialized agents:

* **Data Analysis Agent** — generates and executes Python code inside a Docker sandbox to analyze your dataset
* **Literature Search Agent** — searches Semantic Scholar, ArXiv, and PubMed, downloads open-access PDFs, and extracts findings using RAG
* **Orchestrator Agent** — reads the accumulated knowledge state and decides what to investigate next

All findings are stored in a structured World Model (SQLite + NetworkX graph) and validated by an 8-dimension quality scorer (ScholarEval) before being accepted. The system runs until it reaches the cycle limit or determines the objective has been sufficiently addressed.

---

## What Can It Do?

**Analyze any tabular dataset**
Give INQUIRO a CSV file and a research question. It will perform differential abundance analysis, PCA, correlation analysis, regression, clustering, pathway enrichment, and more — generating Jupyter notebooks with full code and output for every analysis.

**Search across three literature databases simultaneously**
Every research cycle includes a literature search across Semantic Scholar, ArXiv, and PubMed. Open-access PDFs are downloaded, chunked, and stored in a ChromaDB vector database for semantic retrieval.

**Filter out junk findings automatically**
ScholarEval scores every finding on 8 scientific quality dimensions: statistical validity, reproducibility, novelty, significance, methodological soundness, evidence quality, claim calibration, and citation support. IRB approvals, methodology descriptions, and other non-discoveries are automatically rejected before they reach the report.

**Avoid repeating itself**
The Novelty Detector checks every proposed task against existing findings and completed tasks before execution. If the orchestrator tries to run PCA three cycles in a row, the second and third attempts are blocked.

**Adapt its research strategy over time**
Early cycles explore broadly. Later cycles exploit the strongest leads. This explore/exploit ratio shifts automatically based on cycle progress (70/30 in cycle 1 → 30/70 in the final cycle).

**Produce a cited discovery report**
Every finding in the report is traceable to either a specific Jupyter notebook cell (for data findings) or a specific paper with DOI (for literature findings). Nothing in the report is unsourced.

**Generate publication-ready LaTeX output**
INQUIRO can compile reports to LaTeX with four templates: plain article, arXiv preprint, NeurIPS conference, or IEEE journal format. PDF compilation with proper academic formatting.

**Automated peer review**
An AI reviewer scores reports on Soundness, Significance, Novelty, and Clarity — mimicking academic peer review with actionable feedback.

**Parse PDFs with GROBID**
When GROBID is running, PDFs are parsed into structured sections (abstract, methods, results, references) for more accurate literature extraction.

**Run reliably in production**
Circuit breakers prevent cascade failures when APIs are unavailable. A package resolver automatically installs missing Python packages inside the Docker sandbox. A pre-warmed container pool reduces cold-start latency. Every workflow step is logged to structured JSONL for debugging.

---

## What INQUIRO Is Not

* It is not a substitute for domain expertise or human judgment
* It does not have access to paywalled papers (open access only)
* It works best with structured tabular data (CSV); unstructured data requires modification
* Results should be treated as hypotheses to investigate, not validated conclusions
* With local 7B–12B models, output quality is lower than with frontier models (GPT-4o, Claude Sonnet)

---

## How to Get Started

### Prerequisites

* Python 3.11+
* Docker Desktop (for code execution sandbox)
* One of: Ollama (local), OpenAI API key, Anthropic API key, or Google Gemini API key

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/inquiro.git
cd inquiro

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

> **CPU vs GPU note:** if you plan to use local embeddings via
> `sentence-transformers` you will need a compatible PyTorch build. The
> `requirements.txt` file lists only a minimum; the package will pull in
> whatever `torch`/`torchvision` wheel is available. Mismatched binaries
> (for example, installing a CUDA wheel on a machine without the
> appropriate CUDA runtime) can cause import errors such as
> ``RuntimeError: operator torchvision::nms does not exist`` or the
> misleading ``ModuleNotFoundError: Could not import module 'PreTrainedModel'``
> that occurs when the underlying transformers library fails to load.
>
> To avoid issues, you can explicitly install the CPU-only wheels before
> running INQUIRO, e.g.:
>
> ```bash
> pip install torch==2.10.0+cpu torchvision==0.25.0+cpu torchaudio==2.10.0+cpu \
>     --index-url https://download.pytorch.org/whl/cpu
> pip install --upgrade --force-reinstall sentence-transformers
> ```
>
> Or, if you *do* have a CUDA‑capable GPU, uncomment the extra-index-url
> lines in `requirements.txt` and install the appropriate versions.

afterwards continue building the docker image if needed

```bash
# Build the Docker sandbox image
docker build -t inquiro-sandbox .
```

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Choose one provider
LLM_PROVIDER=ollama          # ollama | openai | anthropic | gemini

# Ollama (local, free)
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=gemma3:12b         # or llama3.1:8b, qwen2.5:7b, etc.

# OpenAI (optional)
OPENAI_API_KEY=sk-...

# Anthropic (optional)
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini (optional)
GOOGLE_API_KEY=...

# Semantic Scholar (optional but recommended — raises rate limits)
SEMANTIC_SCHOLAR_API_KEY=...
```

### Running Your First Experiment

**Option 1: Interactive Wizard (Recommended)**

```bash
python run_inquiro.py
```

The wizard guides you through:
1. Research objective
2. Research mode (Data + Literature / Literature Only / Data Only)
3. Dataset selection (upload or search public datasets)
4. Model configuration (hybrid local+cloud / local only / cloud only)
5. Research strategy (Standard / Question-Driven / Hybrid)
6. Advanced options (cycles, LaTeX output, etc.)

**Option 2: Direct CLI**

```bash
# Basic run — 5 cycles
python inquiro_cli.py run \
  --objective "Identify the key metabolic features that distinguish sample groups" \
  --data "./data/sample_metabolomics.csv" \
  --cycles 5

# Literature-only research
python inquiro_cli.py run \
  --objective "What are the mechanisms of exercise-induced cognitive improvement?" \
  --mode literature \
  --cycles 3
```

Outputs are saved to `outputs/run_<timestamp>_<objective_slug>/`:

```
outputs/run_20260220_095441_identify_the_key_metabolic_fea/
├── report_*.md          ← Discovery report with citations
├── stages.jsonl         ← Structured execution log
├── notebooks/
│   ├── analysis_001.ipynb
│   ├── analysis_002.ipynb
│   └── ...
└── figures/
    ├── pca_biplot.png
    └── ...
```

### Running the Test Suite

```bash
# Smoke tests (structure and imports — ~16s)
pytest tests/test_smoke.py -v --log-file=logs/smoke_test.log

# End-to-end tests (full component pipelines — ~28s)
pytest tests/test_e2e.py -v --log-file=logs/e2e_test.log

# Full suite
pytest tests/ -q
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    INQUIRO CLI                        │
│              inquiro_cli.py / src/cli.py              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 Research Engine                      │
│                  src/core/inquiro.py                  │
│                                                      │
│  ┌─────────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Orchestrator│  │ Compressor│  │  StageTracker  │  │
│  │   Agent     │  │ (3-tier) │  │  (JSONL log)   │  │
│  └──────┬──────┘  └──────────┘  └────────────────┘  │
│         │                                            │
│  ┌──────▼───────────────────────┐                   │
│  │        World Model            │                   │
│  │  SQLite + NetworkX graph      │                   │
│  └──────┬───────────────────────┘                   │
│         │                                            │
│  ┌──────▼──────┐     ┌─────────────────────────┐    │
│  │    Data     │     │     Literature Search    │    │
│  │  Analysis   │     │          Agent           │    │
│  │   Agent     │     │                          │    │
│  └──────┬──────┘     └────────────┬─────────────┘   │
│         │                         │                  │
│  ┌──────▼──────┐     ┌────────────▼─────────────┐   │
│  │   Docker    │     │  Semantic Scholar + ArXiv │   │
│  │  Sandbox    │     │      + PubMed + RAG       │   │
│  └─────────────┘     └──────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Key Components

| Component                 | Location                              | Purpose                                       |
| ------------------------- | ------------------------------------- | --------------------------------------------- |
| `Inquiro`                | `src/core/inquiro.py`                | Main research engine and cycle loop           |
| `OrchestratorAgent`     | `src/agents/orchestrator.py`        | Task planning, novelty detection, plan review |
| `DataAnalysisAgent`     | `src/agents/data_analysis.py`       | Code generation and execution                 |
| `LiteratureSearchAgent` | `src/agents/literature.py`          | Multi-source search and PDF extraction        |
| `WorldModel`            | `src/world_model/`                  | SQLite + NetworkX knowledge store             |
| `ScholarEval`           | `src/validation/schol_eval.py`      | 8-dimension finding quality scorer            |
| `ContextCompressor`     | `src/compression/`                  | 3-tier hierarchical context compression       |
| `NoveltyDetector`       | `src/novelty/`                      | Redundant task detection                      |
| `PlanReviewer`          | `src/orchestration/`                | Task quality scoring + explore/exploit        |
| `CircuitBreaker`        | `src/utils/circuit_breaker.py`      | LLM API failure protection                    |
| `PackageResolver`       | `src/execution/package_resolver.py` | Auto pip install on ImportError               |
| `StageTracker`          | `src/tracking/stage_tracker.py`     | Structured JSONL execution logging            |
| `LatexCompiler`         | `src/reports/latex_compiler.py`     | LaTeX/PDF report compilation                  |
| `PaperReviewer`         | `src/validation/paper_reviewer.py`  | Automated peer review scoring                 |
| `QuestionDeepSearcher`  | `src/literature/question_deep_search.py` | Question-driven deep research (QDDR)     |

---

## Project Structure

```
inquiro/
├── run_inquiro.py              ← Interactive wizard (recommended)
├── inquiro_cli.py              ← Direct CLI entry point
├── src/
│   ├── core/
│   │   ├── inquiro.py          ← Main engine
│   │   └── dataset_search.py  ← Public dataset discovery
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── data_analysis.py
│   │   └── literature.py
│   ├── world_model/
│   │   ├── world_model.py
│   │   ├── database.py
│   │   └── models.py
│   ├── validation/
│   │   ├── schol_eval.py      ← 8-dimension quality scorer
│   │   ├── paper_reviewer.py  ← Automated peer review
│   │   └── figure_verifier.py ← VLM figure verification
│   ├── compression/
│   │   └── context_compressor.py
│   ├── novelty/
│   │   └── novelty_detector.py
│   ├── orchestration/
│   │   ├── plan_reviewer.py
│   │   └── research_plan.py   ← Question decomposition
│   ├── tracking/
│   │   └── stage_tracker.py
│   ├── execution/
│   │   ├── docker_executor.py
│   │   ├── package_resolver.py
│   │   └── notebook_manager.py
│   ├── literature/
│   │   ├── search.py              ← Semantic Scholar
│   │   ├── openalex_search.py     ← OpenAlex (primary)
│   │   ├── arxiv_search.py        ← ArXiv
│   │   ├── pubmed_search.py       ← PubMed
│   │   ├── pdf_parser.py
│   │   ├── rag.py                 ← ChromaDB (hybrid BM25+dense)
│   │   └── question_deep_search.py ← QDDR engine
│   ├── utils/
│   │   ├── llm_client.py       ← Multi-provider LLM + routing
│   │   ├── usage_tracker.py    ← Token usage & caching
│   │   └── circuit_breaker.py
│   └── reports/
│       ├── generator.py
│       ├── latex_compiler.py   ← LaTeX/PDF compilation
│       ├── reproducibility.py  ← Reproducibility packages
│       └── templates/          ← LaTeX templates (plain, arxiv, neurips, ieee)
├── config/
│   ├── settings.py
│   └── prompts/
├── skills/                     ← Domain-specific prompts
├── tests/
│   ├── test_smoke.py           ← 67 structural tests
│   ├── test_e2e.py             ← 41 end-to-end tests
│   └── test_latex.py           ← 46 LaTeX tests
├── Docs/                       ← Project documentation
├── data/                       ← Run outputs and datasets
└── outputs/                    ← Generated reports and notebooks
```

---

## Supported LLM Providers

| Provider                | Models                              | Notes                                           |
| ----------------------- | ----------------------------------- | ----------------------------------------------- |
| **Ollama**(local) | gemma3:12b, llama3.1:8b, qwen2.5:7b | Free, runs locally, recommended for development |
| **Anthropic**     | claude-sonnet-4-6, claude-haiku-4-5 | Best quality, prompt caching enabled            |
| **OpenAI**        | gpt-4o, gpt-4o-mini                 | Strong performance                              |
| **Google Gemini** | gemini-2.5-flash, gemini-2.0-flash  | Good value                                      |
| **LM Studio**     | Any GGUF model                      | Local alternative to Ollama                     |

INQUIRO automatically routes complex tasks (planning, reasoning) to the full model and simple tasks (extraction, formatting) to a cheaper model when using cloud providers.

---

## Configuration Reference

Key settings in `.env`:

```env
# Research parameters
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:12b
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=8192

# Literature search
SEMANTIC_SCHOLAR_API_KEY=     # Optional, raises rate limits
MAX_PAPERS_PER_SEARCH=20
MAX_PAPERS_TO_READ=15
RAG_TOP_K=10

# Execution
DOCKER_TIMEOUT=300
```

---

## How a Research Cycle Works

```
Cycle N
  │
  ├─ 1. Orchestrator reads World Model
  │       ↓ compressed context (not full dump)
  │       ↓ novelty detector synced with existing findings
  │
  ├─ 2. Generate tasks (LLM)
  │       ↓ novelty filter (Jaccard similarity > 0.65 → rejected)
  │       ↓ plan reviewer (composite score < 0.45 → rejected)
  │       ↓ explore/exploit ratio adjusts by cycle
  │
  ├─ 3. Execute tasks in parallel (ThreadPoolExecutor)
  │       ├─ Data tasks → LLM generates code → Docker executes
  │       │               package resolver handles ImportErrors
  │       └─ Literature tasks → search 3 APIs → download PDFs
  │                             → chunk → ChromaDB → RAG query
  │
  ├─ 4. Validate findings (ScholarEval 8 dimensions)
  │       ↓ score < 0.40 → rejected
  │       ↓ IRB/ethics claims → hard rejected
  │
  ├─ 5. Store findings in World Model
  │       ↓ SQLite (persistence)
  │       ↓ NetworkX graph (relationships)
  │
  ├─ 6. Orchestrator proposes relationships
  │
  └─ 7. Compress cycle into tier-2 summary
         ↓ ready for next cycle
```

---

## Output Example

A typical 2-cycle run on a metabolomics dataset produces:

* **16–20 findings** across data analysis and literature
* **8–10 relationships** between findings
* **4 analysis notebooks** with executable Python code
* **1 discovery report** with 5 ranked discoveries, each with narrative and citations

Example finding:

```
Discovery 1: Age explains 95–99% of variance in TMAO levels (R²=0.95–0.99)
Confidence: 97% | Source: data_analysis | Cycle: 2

Analysis revealed a striking relationship between age and trimethylamine
N-oxide (TMAO) levels within both control (r=0.97) and treatment (r=1.00)
groups. This near-perfect correlation suggests age is a primary determinant
of TMAO concentrations in this cohort...

Sources:
- [Data] Notebook: analysis_004.ipynb, Cell 2
```

---

## Limitations and Known Issues

* **Local models** : 7B–12B models occasionally generate code with wrong variable names or fail to produce findings. Larger models (70B+) or cloud APIs produce substantially better results.
* **Rate limiting** : Semantic Scholar's free tier limits searches aggressively. A free API key at [semanticscholar.org](https://www.semanticscholar.org/product/api) is strongly recommended.
* **PDF access** : Many publishers block automated PDF downloads. Typically 30–50% of papers are successfully retrieved; the rest fall back to abstract-only analysis.
* **Literature drift** : The literature agent occasionally retrieves papers tangentially related to the objective. The dataset summary anchor and ScholarEval reduce this but do not eliminate it.

---

## Running Tests

```bash
# All tests (154 total)
pytest tests/ -q

# With logging
pytest tests/ -v --log-file=logs/test.log --log-file-level=DEBUG

# Individual test files
pytest tests/test_smoke.py -q    # 67 structural tests (~20s)
pytest tests/test_e2e.py -q      # 41 end-to-end tests (~50s)
pytest tests/test_latex.py -q    # 46 LaTeX tests (~10s)
```

All tests use mocked LLM calls and Docker — no API keys or running containers needed.

---

## Versioning

| Version    | Description                                                                                                                                              |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v0.1.0` | Phase 5 complete: CLI, quality fixes, merge conflict resolved                                                                                            |
| `v0.2.0` | Phase 6 complete: ScholarEval, compression, novelty detection, ArXiv/PubMed, circuit breaker, package resolver, model routing, stage tracking, 108 tests |
| `v0.3.0` | Phase 7 complete: Model router, gap analysis, QDDR (Question-Driven Deep Research), hybrid search, GROBID PDF parsing, domain skills, LaTeX compilation, peer review, figure verification, reproducibility packages, 154 tests |

---

## References

**Primary Inspiration**

[1] Mitchener, L., Yiu, A., Chang, J., et al. (2025).  *KOSMOS: An AI Scientist for Autonomous Discovery* . arXiv preprint arXiv:2511.02824v2. Edison Scientific. https://arxiv.org/abs/2511.02824

**Community Implementations**

[2] jimmc414.  *KOSMOS: An AI Scientist for Autonomous Discovery — An implementation and adaptation to be driven by Claude Code or API* . GitHub. https://github.com/jimmc414/Inquiro

**Libraries and Frameworks**

[3] McKinney, W. (2010). Data Structures for Statistical Computing in Python.  *Proceedings of the 9th Python in Science Conference* , 56–61. https://pandas.pydata.org

[4] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.  *Journal of Machine Learning Research* , 12, 2825–2830. https://scikit-learn.org

[5] Virtanen, P., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.  *Nature Methods* , 17, 261–272. https://scipy.org

[6] Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and Statistical Modeling with Python.  *Proceedings of the 9th Python in Science Conference* . https://www.statsmodels.org

[7] Chroma. (2023).  *ChromaDB: The AI-native open-source embedding database* . https://www.trychroma.com

[8] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.  *Proceedings of EMNLP 2019* . https://www.sbert.net

[9] Lo, K., et al. (2020). S2ORC: The Semantic Scholar Open Research Corpus.  *Proceedings of ACL 2020* . Semantic Scholar API: https://api.semanticscholar.org

[10] Sayers, E. (2010).  *A General Introduction to the E-utilities* . NCBI Entrez Programming Utilities. PubMed API: https://www.ncbi.nlm.nih.gov/books/NBK25501

[11] arXiv.org.  *arXiv API User's Manual* . Cornell University. https://arxiv.org/help/api

[12] Hunter, J.D. (2007). Matplotlib: A 2D Graphics Environment.  *Computing in Science & Engineering* , 9(3), 90–95. https://matplotlib.org

[13] Harris, C.R., et al. (2020). Array programming with NumPy.  *Nature* , 585, 357–362. https://numpy.org

[14] Click. (2023).  *Click: Python composable command line interface toolkit* . https://click.palletsprojects.com

[15] NetworkX Developers. (2008).  *NetworkX: Network Analysis in Python* . https://networkx.org

**Design Patterns**

[16] Fowler, M. (2014).  *Circuit Breaker Pattern* . martinfowler.com. https://martinfowler.com/bliki/CircuitBreaker.html

[17] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.  *Advances in Neural Information Processing Systems* , 33. https://arxiv.org/abs/2005.11401
