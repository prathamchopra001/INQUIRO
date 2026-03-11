"""
Task Skill Generator for INQUIRO.

Analyzes data analysis tasks to identify required libraries and techniques,
then generates task-specific SKILL.md files with correct API patterns.

This is different from the role-based SkillGenerator:
- SkillGenerator: Creates skills for ROLES (finding_extraction, code_generation)
- TaskSkillGenerator: Creates skills for TECHNIQUES (BLOSUM alignment, Spearman correlation)

Flow:
1. Data Analysis Agent receives task: "Calculate BLOSUM62 sequence similarity"
2. TaskSkillGenerator analyzes task → identifies: biopython, Bio.Align, substitution_matrices
3. Generates SKILL.md with correct modern API patterns (not deprecated pairwise2)
4. Code generation uses this skill → produces working code
5. Skill cached for future similar tasks
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# KNOWN API PATTERNS (from research - correct modern patterns)
# =============================================================================

KNOWN_PATTERNS = {
    # BioPython alignment - CRITICAL: pairwise2 and SubsMat are deprecated/removed
    "biopython_alignment": {
        "keywords": ["blosum", "blosum62", "blosum80", "blosum45", "pairwise", "sequence alignment", 
                     "global alignment", "local alignment", "smith-waterman", "needleman-wunsch", 
                     "substitution matrix", "pam250", "alignment score"],
        "packages": ["biopython"],
        "pip_names": {"biopython": "biopython"},
        "correct_imports": [
            "from Bio import Align",
            "from Bio.Align import substitution_matrices",
        ],
        "wrong_imports": [
            "from Bio import pairwise2  # DEPRECATED since BioPython 1.80",
            "from Bio.SubsMat import MatrixInfo  # REMOVED in BioPython 1.82",
            "from Bio.SubsMat.MatrixInfo import blosum62  # REMOVED",
        ],
        "example_code": '''
from Bio import Align
from Bio.Align import substitution_matrices

# Create aligner with BLOSUM62
aligner = Align.PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5

# For local alignment (Smith-Waterman style)
# aligner.mode = "local"

# Align sequences
seq1 = "EVQLVESGGGLVQ"
seq2 = "EVQLVQSGGGLVK"
alignments = aligner.align(seq1, seq2)
score = alignments[0].score
print(f"Alignment score: {score}")
''',
        "pitfalls": [
            "Bio.pairwise2 emits BiopythonDeprecationWarning on import (deprecated since 1.80)",
            "Bio.SubsMat was REMOVED in BioPython ~1.82 - causes ModuleNotFoundError",
            "Default gap score changed from 0.0 to -1.0 in BioPython 1.86 - always set explicitly",
        ],
    },
    
    # Statsmodels multiple testing
    "statsmodels_multitest": {
        "keywords": ["multiple testing", "bonferroni", "fdr", "benjamini", "p-value correction",
                     "multipletests", "multitest"],
        "packages": ["statsmodels"],
        "pip_names": {"statsmodels": "statsmodels"},
        "correct_imports": [
            "from statsmodels.stats.multitest import multipletests",
        ],
        "wrong_imports": [
            "from statsmodels.stats.multicomp import multipletests  # WRONG MODULE",
        ],
        "example_code": '''
from statsmodels.stats.multitest import multipletests
import numpy as np

pvals = np.array([0.001, 0.04, 0.03, 0.80, 0.005])
reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
    pvals, alpha=0.05, method='fdr_bh'  # Benjamini-Hochberg
)
# Other methods: 'bonferroni', 'sidak', 'holm', 'fdr_by'
print(f"Corrected p-values: {pvals_corrected}")
print(f"Reject null: {reject}")
''',
        "pitfalls": [
            "multipletests is in statsmodels.stats.multitest, NOT multicomp",
            "multicomp contains pairwise_tukeyhsd - completely different function",
        ],
    },
    
    # Statsmodels regression
    "statsmodels_regression": {
        "keywords": ["ols", "regression", "linear model", "statsmodels"],
        "packages": ["statsmodels"],
        "pip_names": {"statsmodels": "statsmodels"},
        "correct_imports": [
            "import statsmodels.formula.api as smf  # For formula API",
            "import statsmodels.api as sm  # For matrix API",
        ],
        "example_code": '''
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd

# Option 1: Formula API (auto-adds intercept)
model = smf.ols('y ~ x1 + x2', data=df).fit()

# Option 2: Matrix API (MUST add constant manually!)
X = df[['x1', 'x2']]
X = sm.add_constant(X)  # CRITICAL - without this, no intercept!
y = df['y']
model = sm.OLS(y, X).fit()

print(model.summary())
''',
        "pitfalls": [
            "Matrix API (sm.OLS) requires sm.add_constant(X) - silently fits no-intercept model otherwise",
            "Formula API (smf.ols - lowercase) auto-adds intercept",
        ],
    },
    
    # Sequence similarity (Levenshtein)
    "sequence_similarity": {
        "keywords": ["levenshtein", "edit distance", "sequence similarity", "string distance",
                     "rapidfuzz", "fuzzy matching"],
        "packages": ["rapidfuzz"],  # Prefer over python-Levenshtein (faster, MIT licensed)
        "pip_names": {"rapidfuzz": "rapidfuzz"},
        "correct_imports": [
            "from rapidfuzz.distance import Levenshtein",
            "from rapidfuzz import fuzz",
        ],
        "example_code": '''
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

seq1 = "EVQLVESGGGLVQ"
seq2 = "EVQLVQSGGGLVK"

# Edit distance
dist = Levenshtein.distance(seq1, seq2)

# Normalized similarity (0-1)
sim = Levenshtein.normalized_similarity(seq1, seq2)

# Fuzzy ratio (0-100)
ratio = fuzz.ratio(seq1, seq2)

print(f"Distance: {dist}, Similarity: {sim:.2f}, Ratio: {ratio}")
''',
        "pitfalls": [
            "python-Levenshtein requires GPL license - use rapidfuzz (MIT) instead",
            "rapidfuzz is 5-100x faster than python-Levenshtein",
        ],
    },
    
    # Protein physicochemical properties
    "protein_properties": {
        "keywords": ["molecular weight", "isoelectric point", "gravy", "hydrophobicity",
                     "instability index", "aromaticity", "protein properties", "physicochemical"],
        "packages": ["biopython"],
        "pip_names": {"biopython": "biopython"},
        "correct_imports": [
            "from Bio.SeqUtils.ProtParam import ProteinAnalysis",
        ],
        "example_code": '''
from Bio.SeqUtils.ProtParam import ProteinAnalysis

sequence = "EVQLVESGGGLVQPGGSLRL"
pa = ProteinAnalysis(sequence)

mw = pa.molecular_weight()
pi = pa.isoelectric_point()
gravy = pa.gravy()  # Grand Average of Hydropathy
instability = pa.instability_index()
aromaticity = pa.aromaticity()

print(f"MW: {mw:.2f}, pI: {pi:.2f}, GRAVY: {gravy:.2f}")
''',
        "pitfalls": [],
    },
    
    # Spearman/Pearson correlation
    "correlation_analysis": {
        "keywords": ["spearman", "pearson", "correlation", "rank correlation", "scipy.stats"],
        "packages": ["scipy"],
        "pip_names": {"scipy": "scipy"},
        "correct_imports": [
            "from scipy.stats import spearmanr, pearsonr",
        ],
        "example_code": '''
from scipy.stats import spearmanr, pearsonr
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 6, 7, 8, 7])

# Spearman rank correlation
rho, pvalue = spearmanr(x, y)
print(f"Spearman rho: {rho:.3f}, p-value: {pvalue:.4f}")

# Pearson correlation
r, pvalue = pearsonr(x, y)
print(f"Pearson r: {r:.3f}, p-value: {pvalue:.4f}")
''',
        "pitfalls": [
            "spearmanr returns (correlation, pvalue) tuple - unpack both",
            "For DataFrames, use df.corr(method='spearman') for correlation matrix",
        ],
    },
    
    # Antibody numbering (conda-only)
    "antibody_numbering": {
        "keywords": ["antibody numbering", "imgt", "chothia", "kabat", "cdr", "abnumber", "anarci"],
        "packages": ["abnumber"],
        "pip_names": {"abnumber": "abnumber"},  # Note: requires conda, not pip
        "correct_imports": [
            "from abnumber import Chain",
        ],
        "example_code": '''
# NOTE: abnumber requires conda installation (not pip) due to HMMER3 dependency
# conda install -c bioconda abnumber

from abnumber import Chain

sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDKILWFGEPVFDYWGQGTLVTVSS"
chain = Chain(sequence, scheme='imgt')

print(f"CDR1: {chain.cdr1_seq}")
print(f"CDR2: {chain.cdr2_seq}")
print(f"CDR3: {chain.cdr3_seq}")
''',
        "pitfalls": [
            "abnumber requires conda (bioconda channel) - pip install will fail",
            "Requires HMMER3 which has no Windows support",
            "If on Windows, use alternative like AbRSA or ANARCI web service",
        ],
    },
    
    # Clustering
    "clustering": {
        "keywords": ["kmeans", "clustering", "hierarchical", "dbscan", "agglomerative"],
        "packages": ["scikit-learn"],
        "pip_names": {"sklearn": "scikit-learn"},
        "correct_imports": [
            "from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN",
            "from sklearn.preprocessing import StandardScaler",
        ],
        "example_code": '''
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Always scale features before clustering
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
print(f"Cluster labels: {labels}")
''',
        "pitfalls": [
            "Always scale features before clustering (StandardScaler)",
            "Set random_state for reproducibility",
            "n_init='auto' in newer sklearn, explicit n_init=10 for older versions",
        ],
    },
    
    # INFEASIBLE: Large protein language models (ESM, ProtBERT, AlphaFold)
    # These require GPU, large memory, and aren't available in the Docker sandbox
    "infeasible_protein_llm": {
        "keywords": ["esm", "esm-1", "esm-2", "esm1v", "esm-1v", "protbert", "prot-bert",
                     "alphafold", "openfold", "esmfold", "protein language model",
                     "protein embeddings", "fine-tune esm", "fine-tune protbert",
                     "protein transformer", "ankh", "prottrans"],
        "packages": [],  # Not installable in sandbox
        "pip_names": {},
        "correct_imports": [],
        "wrong_imports": [
            "import esm  # Requires GPU, 700MB+ model, not available in sandbox",
            "from transformers import AutoModel  # ProtBERT requires ~500MB download",
            "import torch  # Heavy dependency, may not be available",
        ],
        "example_code": '''
# ============================================================
# ⚠️  THIS TASK CANNOT BE COMPLETED IN THE SANDBOX
# ============================================================
#
# The task requires large protein language models (ESM, ProtBERT, etc.)
# which need:
#   - GPU acceleration (not available in sandbox)
#   - Large model downloads (500MB - 2GB+)
#   - PyTorch with CUDA support
#
# ALTERNATIVE APPROACHES for this sandbox:
# 1. Use pre-computed embeddings if available in the dataset
# 2. Use BioPython's ProteinAnalysis for physicochemical features
# 3. Use sequence similarity (Levenshtein, BLOSUM) instead of embeddings
#
# If you need protein embeddings, consider:
# - Using the HuggingFace Inference API (external)
# - Pre-computing embeddings outside the sandbox
# - Using smaller feature sets (amino acid composition, k-mer counts)

from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np

def compute_simple_features(sequence):
    """Compute interpretable protein features (sandbox-friendly)."""
    pa = ProteinAnalysis(sequence)
    return {
        'length': len(sequence),
        'molecular_weight': pa.molecular_weight(),
        'isoelectric_point': pa.isoelectric_point(),
        'gravy': pa.gravy(),
        'instability_index': pa.instability_index(),
        'aromaticity': pa.aromaticity(),
    }

# This is a feasible alternative to ESM embeddings
sequence = "EVQLVESGGGLVQ"
features = compute_simple_features(sequence)
print(f"Features: {features}")
''',
        "pitfalls": [
            "ESM/ProtBERT require GPU and large downloads - NOT available in sandbox",
            "Fine-tuning protein models requires significant compute resources",
            "Use physicochemical features (BioPython) or pre-computed embeddings instead",
            "Consider sequence similarity metrics as an alternative to embeddings",
        ],
        "is_infeasible": True,  # Special flag for infeasible tasks
        "alternative_approach": "Use BioPython ProteinAnalysis or pre-computed embeddings",
    },
}


# =============================================================================
# TASK ANALYSIS PROMPT
# =============================================================================

TASK_ANALYSIS_PROMPT = '''Analyze this data analysis task and identify the required techniques and libraries.

## Task Description
{task_description}

## Dataset Context
{dataset_context}

## Instructions
Identify:
1. What computational techniques are needed (e.g., sequence alignment, correlation analysis, clustering)
2. What Python libraries would be required
3. Any domain-specific considerations

## Output Format
Return a JSON object:
```json
{{
    "techniques": ["technique1", "technique2"],
    "libraries": ["library1", "library2"],
    "keywords": ["keyword1", "keyword2"],
    "domain": "bioinformatics|statistics|machine_learning|general",
    "complexity": "low|medium|high",
    "notes": "Any special considerations"
}}
```

Return ONLY the JSON object, no other text.'''


# =============================================================================
# SKILL GENERATION PROMPT
# =============================================================================

SKILL_GENERATION_PROMPT = '''Generate a coding skill guide for this data analysis task.

## Task
{task_description}

## Required Techniques
{techniques}

## Known Correct Patterns
{known_patterns}

## Instructions
Create a concise SKILL.md that helps a code generation model write correct, working code.

Include:
1. **Required Packages**: pip install commands
2. **Correct Imports**: Exact import statements to use
3. **DO NOT USE**: Deprecated/wrong patterns to avoid
4. **Example Code**: A minimal working example
5. **Common Pitfalls**: Errors to watch for

## Format
Use markdown. Keep it under 400 words. Be specific and actionable.

Start directly with the content (no preamble).'''


# =============================================================================
# TASK SKILL GENERATOR
# =============================================================================

class TaskSkillGenerator:
    """
    Generates task-specific skills for data analysis.
    
    Unlike role-based skills (finding_extraction, code_generation),
    these are technique-specific (BLOSUM alignment, statistical tests).
    
    Process:
    1. Analyze task to identify techniques/libraries
    2. Check for known patterns (pre-researched correct APIs)
    3. Generate skill with correct patterns
    4. Cache for reuse on similar tasks
    """
    
    def __init__(
        self,
        llm_client: "LLMClient",
        skills_dir: str = "./skills/task_skills",
        enabled: bool = True,
    ):
        self.llm = llm_client
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._cache: Dict[str, str] = {}  # task_hash -> skill content
        
    def _hash_task(self, task: str) -> str:
        """Create a short hash of the task for caching."""
        return hashlib.md5(task.lower().encode()).hexdigest()[:12]
    
    def _find_matching_patterns(self, task: str, keywords: List[str] = None) -> List[Dict]:
        """Find known patterns that match the task."""
        task_lower = task.lower()
        all_keywords = set(keywords or [])
        
        # Extract keywords from task text
        for word in re.findall(r'\b\w+\b', task_lower):
            all_keywords.add(word)
        
        matches = []
        for pattern_name, pattern in KNOWN_PATTERNS.items():
            pattern_keywords = set(kw.lower() for kw in pattern["keywords"])
            overlap = all_keywords & pattern_keywords
            if overlap:
                matches.append({
                    "name": pattern_name,
                    "match_count": len(overlap),
                    "matched_keywords": list(overlap),
                    **pattern,
                })
        
        # Sort by match count (most relevant first)
        matches.sort(key=lambda x: x["match_count"], reverse=True)
        return matches
    
    def _analyze_task(self, task: str, dataset_context: str = "") -> Dict[str, Any]:
        """Analyze a task to identify required techniques and libraries."""
        # First, check for known pattern matches
        pattern_matches = self._find_matching_patterns(task)
        
        if pattern_matches:
            # We have known patterns - use them
            techniques = [m["name"] for m in pattern_matches[:3]]
            libraries = []
            for m in pattern_matches[:3]:
                libraries.extend(m.get("packages", []))
            libraries = list(set(libraries))
            
            return {
                "techniques": techniques,
                "libraries": libraries,
                "pattern_matches": pattern_matches[:3],
                "source": "known_patterns",
            }
        
        # Fall back to LLM analysis
        try:
            prompt = TASK_ANALYSIS_PROMPT.format(
                task_description=task,
                dataset_context=dataset_context or "No specific dataset context provided.",
            )
            
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="fast",  # Use fast tier for analysis
                system="You are a data science expert. Analyze tasks and identify required techniques.",
            )
            
            # Parse JSON response
            content = response.content.strip()
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                analysis = json.loads(json_match.group())
                analysis["source"] = "llm_analysis"
                analysis["pattern_matches"] = []
                return analysis
        except Exception as e:
            logger.warning(f"Task analysis failed: {e}")
        
        # Default fallback
        return {
            "techniques": ["general_analysis"],
            "libraries": ["pandas", "numpy", "scipy"],
            "source": "fallback",
            "pattern_matches": [],
        }
    
    def _format_known_patterns(self, matches: List[Dict]) -> str:
        """Format known patterns for inclusion in skill generation prompt."""
        if not matches:
            return "No pre-defined patterns available. Use standard library documentation."
        
        sections = []
        for match in matches[:3]:  # Top 3 matches
            section = f"### {match['name'].replace('_', ' ').title()}\n\n"
            
            if match.get("correct_imports"):
                section += "**Correct imports:**\n```python\n"
                section += "\n".join(match["correct_imports"])
                section += "\n```\n\n"
            
            if match.get("wrong_imports"):
                section += "**DO NOT USE (deprecated/wrong):**\n```python\n"
                section += "\n".join(match["wrong_imports"])
                section += "\n```\n\n"
            
            if match.get("example_code"):
                section += "**Example:**\n```python"
                section += match["example_code"]
                section += "```\n\n"
            
            if match.get("pitfalls"):
                section += "**Pitfalls:**\n"
                for pitfall in match["pitfalls"]:
                    section += f"- {pitfall}\n"
                section += "\n"
            
            sections.append(section)
        
        return "\n".join(sections)
    
    def generate_skill(
        self,
        task: str,
        dataset_context: str = "",
        force: bool = False,
    ) -> Optional[str]:
        """
        Generate a task-specific skill.
        
        Args:
            task: The data analysis task description
            dataset_context: Optional context about the dataset
            force: If True, regenerate even if cached
            
        Returns:
            Skill content as markdown string, or None if generation fails
        """
        if not self.enabled:
            return None
        
        task_hash = self._hash_task(task)
        
        # Check cache
        if not force and task_hash in self._cache:
            logger.debug(f"Using cached skill for task hash {task_hash}")
            return self._cache[task_hash]
        
        # Check disk cache
        skill_path = self.skills_dir / f"{task_hash}.md"
        if not force and skill_path.exists():
            content = skill_path.read_text(encoding="utf-8")
            self._cache[task_hash] = content
            logger.info(f"📚 Loaded cached task skill: {task_hash}")
            return content
        
        logger.info(f"🔧 Generating task skill for: {task[:60]}...")
        
        # Analyze the task
        analysis = self._analyze_task(task, dataset_context)
        
        # Get known patterns
        pattern_matches = analysis.get("pattern_matches", [])
        if not pattern_matches:
            pattern_matches = self._find_matching_patterns(
                task, 
                analysis.get("keywords", [])
            )
        
        known_patterns_text = self._format_known_patterns(pattern_matches)
        
        # Generate skill
        try:
            prompt = SKILL_GENERATION_PROMPT.format(
                task_description=task,
                techniques=", ".join(analysis.get("techniques", ["general analysis"])),
                known_patterns=known_patterns_text,
            )
            
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="fast",
                system="You are an expert at creating concise coding guides. Focus on correct, working code patterns.",
            )
            
            skill_content = response.content.strip()
            
            # Validate
            if len(skill_content) < 100:
                logger.warning("Generated skill is too short")
                return None
            
            # Add metadata header
            metadata = f"""---
task_hash: {task_hash}
generated: {datetime.now().isoformat()}
techniques: {analysis.get('techniques', [])}
libraries: {analysis.get('libraries', [])}
source: {analysis.get('source', 'unknown')}
---

"""
            full_content = metadata + skill_content
            
            # Cache to disk
            skill_path.write_text(full_content, encoding="utf-8")
            self._cache[task_hash] = full_content
            
            logger.info(f"✅ Generated task skill: {task_hash}")
            return full_content
            
        except Exception as e:
            logger.error(f"Failed to generate task skill: {e}")
            return None
    
    def get_required_packages(self, task: str) -> List[Tuple[str, str]]:
        """
        Get required packages for a task.
        
        Returns:
            List of (import_name, pip_name) tuples
        """
        analysis = self._analyze_task(task)
        packages = []
        
        for match in analysis.get("pattern_matches", []):
            pip_names = match.get("pip_names", {})
            for pkg in match.get("packages", []):
                pip_name = pip_names.get(pkg, pkg)
                packages.append((pkg, pip_name))
        
        # Deduplicate
        seen = set()
        unique = []
        for import_name, pip_name in packages:
            if import_name not in seen:
                seen.add(import_name)
                unique.append((import_name, pip_name))
        
        return unique
    
    def get_skill_for_code_generation(self, task: str, dataset_context: str = "") -> str:
        """
        Get a skill block ready for injection into code generation prompt.
        
        This is the main entry point for the Data Analysis Agent.
        
        Returns:
            Formatted skill block or empty string if none generated
        """
        logger.debug(f"TaskSkillGenerator.get_skill_for_code_generation called for: {task[:50]}...")
        
        if not self.enabled:
            logger.debug("TaskSkillGenerator is disabled")
            return ""
        
        # First, try to get skill from known patterns (no LLM needed)
        pattern_matches = self._find_matching_patterns(task)
        
        if pattern_matches:
            # We have known patterns - use them directly without LLM
            logger.info(f"📚 Found {len(pattern_matches)} matching patterns for task")
            skill = self._format_known_patterns(pattern_matches)
            
            # Check for infeasible tasks
            for match in pattern_matches:
                if match.get("is_infeasible"):
                    logger.warning(f"⚠️ Task flagged as INFEASIBLE: {match.get('alternative_approach', 'No alternative')}")
            
            return f"""
<task_specific_skill>
## Technique-Specific Guidance

{skill}
</task_specific_skill>
"""
        
        # No known patterns - fall back to LLM-generated skill
        skill = self.generate_skill(task, dataset_context)
        
        if not skill:
            logger.debug("No skill generated (no patterns matched, LLM generation failed)")
            return ""
        
        # Strip metadata for injection
        if skill.startswith("---"):
            # Remove YAML frontmatter
            end_idx = skill.find("---", 3)
            if end_idx > 0:
                skill = skill[end_idx + 3:].strip()
        
        return f"""
<task_specific_skill>
{skill}
</task_specific_skill>
"""
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()
        logger.info("Task skill cache cleared")
