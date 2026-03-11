# -*- coding: utf-8 -*-
"""
Domain-Specific Skill Prompts for INQUIRO.

Provides specialized knowledge injection for different research domains.
Each domain skill contains:
    - Domain terminology and concepts
    - Common methodologies
    - Quality indicators specific to the field
    - Citation and evidence standards
    - Common pitfalls to avoid

Usage:
    from config.prompts.domain_skills import get_domain_skill, detect_domain
    
    domain = detect_domain(objective)
    skill = get_domain_skill(domain)
    
    enhanced_prompt = f"{skill}\\n\\n{original_prompt}"
"""

from typing import Optional, Dict, Tuple
import re

# =============================================================================
# DOMAIN DETECTION
# =============================================================================

# Keywords for domain detection
DOMAIN_KEYWORDS: Dict[str, list] = {
    "economics": [
        "gdp", "inflation", "unemployment", "interest rate", "monetary policy",
        "fiscal", "trade", "export", "import", "currency", "exchange rate",
        "stock", "bond", "investment", "market", "price", "wage", "income",
        "poverty", "inequality", "economic growth", "recession", "banking",
        "federal reserve", "central bank", "treasury", "finance", "macro",
    ],
    "biology": [
        "gene", "genome", "dna", "rna", "protein", "cell", "organism",
        "species", "evolution", "mutation", "expression", "pathway",
        "metabolic", "enzyme", "receptor", "signaling", "transcription",
        "phylogenetic", "ecological", "biodiversity", "population",
    ],
    "medicine": [
        "disease", "patient", "clinical", "treatment", "drug", "therapy",
        "diagnosis", "symptom", "mortality", "morbidity", "trial",
        "randomized", "placebo", "efficacy", "adverse", "dosage",
        "pharmaceutical", "biomarker", "prognosis", "epidemiology",
    ],
    "physics": [
        "quantum", "particle", "energy", "force", "mass", "velocity",
        "acceleration", "electromagnetic", "wave", "photon", "electron",
        "atom", "nuclear", "thermodynamic", "entropy", "relativity",
        "gravitational", "magnetic", "electric", "field", "momentum",
    ],
    "psychology": [
        "cognitive", "behavior", "perception", "memory", "attention",
        "emotion", "motivation", "personality", "development", "social",
        "clinical", "neuropsychology", "therapy", "disorder", "anxiety",
        "depression", "learning", "consciousness", "stimulus", "response",
    ],
    "computer_science": [
        "algorithm", "machine learning", "neural network", "deep learning",
        "artificial intelligence", "data structure", "complexity",
        "optimization", "distributed", "parallel", "database", "software",
        "programming", "computation", "network", "security", "cryptography",
    ],
    "environmental": [
        "climate", "carbon", "emission", "temperature", "ecosystem",
        "pollution", "sustainability", "renewable", "biodiversity",
        "conservation", "deforestation", "ocean", "atmosphere", "weather",
        "drought", "flood", "sea level", "greenhouse", "ozone",
    ],
    "social_science": [
        "demographic", "population", "survey", "ethnography", "qualitative",
        "quantitative", "interview", "observation", "social", "cultural",
        "political", "institution", "governance", "policy", "inequality",
        "migration", "urbanization", "education", "gender", "race",
    ],
}


def detect_domain(objective: str) -> str:
    """
    Detect research domain from objective text.
    
    Args:
        objective: Research objective text
        
    Returns:
        Domain name (e.g., 'economics', 'biology') or 'general'
    """
    text_lower = objective.lower()
    
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            # Normalize by keyword count
            scores[domain] = matches / len(keywords)
    
    if not scores:
        return "general"
    
    best_domain = max(scores, key=scores.get)
    best_score = scores[best_domain]
    
    # Require minimum confidence
    if best_score < 0.05:  # At least ~2-3 keyword matches
        return "general"
    
    return best_domain


def get_domain_skill(domain: str) -> str:
    """
    Get the skill prompt for a specific domain.
    
    Args:
        domain: Domain name from detect_domain()
        
    Returns:
        Skill prompt string to inject into LLM prompts
    """
    skills = {
        "economics": ECONOMICS_SKILL,
        "biology": BIOLOGY_SKILL,
        "medicine": MEDICINE_SKILL,
        "physics": PHYSICS_SKILL,
        "psychology": PSYCHOLOGY_SKILL,
        "computer_science": COMPUTER_SCIENCE_SKILL,
        "environmental": ENVIRONMENTAL_SKILL,
        "social_science": SOCIAL_SCIENCE_SKILL,
        "general": GENERAL_SKILL,
    }
    return skills.get(domain, GENERAL_SKILL)


def get_domain_skill_for_objective(objective: str) -> Tuple[str, str]:
    """
    Convenience function: detect domain and return skill.
    
    Returns:
        Tuple of (domain_name, skill_prompt)
    """
    domain = detect_domain(objective)
    skill = get_domain_skill(domain)
    return domain, skill


# =============================================================================
# DOMAIN SKILL PROMPTS
# =============================================================================

GENERAL_SKILL = """## Research Quality Guidelines

When analyzing data or literature, ensure:
1. Claims are specific with quantitative evidence where possible
2. Statistical significance is noted (p-values, confidence intervals)
3. Effect sizes and practical significance are distinguished
4. Limitations and caveats are acknowledged
5. Sources are properly attributed

Avoid:
- Overclaiming based on weak evidence
- Confusing correlation with causation
- Cherry-picking supportive findings
- Ignoring contradictory evidence"""


ECONOMICS_SKILL = """## Economics Domain Expertise

### Key Methodological Standards
- **Causal Inference**: Distinguish correlation from causation. Look for:
  - Instrumental variables (IV)
  - Difference-in-differences (DiD)
  - Regression discontinuity (RD)
  - Natural experiments
- **Endogeneity**: Watch for omitted variable bias, reverse causality, measurement error
- **Panel Data**: Fixed effects vs. random effects; Hausman test justification

### Statistical Conventions
- Standard errors should be robust (heteroskedasticity-consistent)
- Clustering by entity (firm, country) when appropriate
- Report R² but recognize its limitations
- Economic significance ≠ statistical significance

### Domain Terminology
- GDP, GNP, PPP-adjusted values
- Real vs. nominal values (inflation-adjusted)
- Elasticity (price, income, cross-price)
- Marginal vs. average effects
- Short-run vs. long-run dynamics

### Quality Indicators
- Published in recognized economics journals (AER, QJE, Econometrica, JPE, REStud)
- Robustness checks with alternative specifications
- Sensitivity analysis for key assumptions
- Clear identification strategy

### Common Pitfalls
- Ecological fallacy (aggregate vs. individual)
- Simpson's paradox in grouped data
- Unit roots in time series (spurious regression)
- Publication bias toward significant results"""


BIOLOGY_SKILL = """## Biology Domain Expertise

### Key Methodological Standards
- **Experimental Design**: Controls, replication, randomization, blinding
- **Sample Size**: Power analysis for detecting meaningful effects
- **Reproducibility**: Methods described in sufficient detail for replication
- **Biological Replicates**: Distinguish technical from biological replicates

### Statistical Conventions
- Multiple testing correction (Bonferroni, FDR/Benjamini-Hochberg)
- Effect sizes with confidence intervals, not just p-values
- Non-parametric tests when assumptions violated
- Appropriate normalization methods

### Domain Terminology
- Gene nomenclature (italics for genes, regular for proteins)
- Pathways and networks (KEGG, GO terms)
- Model organisms (in vivo, in vitro, ex vivo)
- -omics data (transcriptomics, proteomics, metabolomics)

### Quality Indicators
- Peer-reviewed in field journals (Nature, Cell, Science, PNAS, PLoS Biology)
- Deposited raw data (GEO, SRA, PRIDE)
- ARRIVE guidelines for animal studies
- CONSORT for clinical trials

### Common Pitfalls
- Batch effects in high-throughput data
- Overfitting in machine learning on biological data
- Circular analysis (double-dipping)
- Inappropriate statistical tests for non-normal data
- Confusing fold change with statistical significance"""


MEDICINE_SKILL = """## Medicine Domain Expertise

### Key Methodological Standards
- **Study Hierarchy**: Meta-analyses > RCTs > Cohort > Case-control > Case series
- **RCT Quality**: Randomization, allocation concealment, intention-to-treat
- **GRADE Evidence**: Quality ratings for clinical recommendations
- **Bias Assessment**: Cochrane Risk of Bias tool

### Statistical Conventions
- Hazard ratios, odds ratios, relative risk with 95% CI
- Number needed to treat (NNT) / Number needed to harm (NNH)
- Kaplan-Meier survival curves, log-rank tests
- Pre-specified primary and secondary endpoints

### Domain Terminology
- ICD codes, MeSH terms
- PICO framework (Population, Intervention, Comparison, Outcome)
- ITT (intention-to-treat) vs. per-protocol analysis
- Adverse events vs. serious adverse events

### Quality Indicators
- Registered trials (ClinicalTrials.gov)
- CONSORT statement compliance
- Published in medical journals (NEJM, Lancet, JAMA, BMJ)
- Cochrane systematic reviews

### Common Pitfalls
- Confounding by indication
- Immortal time bias
- Lead-time and length-time bias in screening
- Surrogate endpoints vs. patient-centered outcomes
- Industry funding bias"""


PHYSICS_SKILL = """## Physics Domain Expertise

### Key Methodological Standards
- **Uncertainty Quantification**: Systematic vs. statistical uncertainties
- **Reproducibility**: Independent verification of results
- **Calibration**: Detector/instrument calibration procedures
- **Theoretical Predictions**: Comparison with established theory

### Statistical Conventions
- Gaussian error propagation
- Chi-squared goodness of fit
- 5-sigma discovery threshold (particle physics)
- Bayesian vs. frequentist approaches

### Domain Terminology
- SI units with appropriate prefixes
- Dimensionless quantities and natural units
- Standard Model nomenclature
- Order of magnitude estimates

### Quality Indicators
- Physical Review journals (PRL, PRA-E)
- arXiv preprints with peer review follow-up
- Reproducible by independent groups
- Consistency with conservation laws

### Common Pitfalls
- Significant figures beyond measurement precision
- Ignoring systematic uncertainties
- Confirmation bias in data selection
- Overinterpreting numerical coincidences"""


PSYCHOLOGY_SKILL = """## Psychology Domain Expertise

### Key Methodological Standards
- **Replication Crisis Awareness**: Effect sizes often smaller than originally reported
- **Pre-registration**: Hypotheses and analysis plans registered before data collection
- **Power Analysis**: Most psychology studies historically underpowered
- **Open Science**: Data and materials sharing

### Statistical Conventions
- Cohen's d, η², r for effect sizes
- Bayesian alternatives to NHST
- Mixed-effects models for nested data
- Correction for multiple comparisons

### Domain Terminology
- DSM-5/ICD-11 diagnostic criteria
- Validated scales and questionnaires
- Within-subjects vs. between-subjects designs
- Ecological validity vs. internal validity

### Quality Indicators
- Pre-registered studies (OSF, AsPredicted)
- Published in APA journals or equivalent
- Open data and materials
- Direct replications

### Common Pitfalls
- WEIRD samples (Western, Educated, Industrialized, Rich, Democratic)
- Demand characteristics
- Social desirability bias
- p-hacking and HARKing (Hypothesizing After Results Known)
- Small sample sizes in cognitive neuroscience"""



COMPUTER_SCIENCE_SKILL = """## Computer Science Domain Expertise

### Key Methodological Standards
- **Benchmarking**: Standard datasets, fair comparisons, controlled experiments
- **Reproducibility**: Code release, environment specifications, random seeds
- **Ablation Studies**: Isolating contribution of each component
- **Statistical Significance**: Multiple runs with different seeds

### Statistical Conventions
- Mean ± standard deviation over N runs
- Confidence intervals or significance tests for comparisons
- Appropriate baselines (not just weak ones)
- Cross-validation for model selection

### Domain Terminology
- Big-O complexity notation
- Precision, recall, F1, AUC-ROC
- Training/validation/test splits
- Hyperparameter tuning vs. architecture search

### Quality Indicators
- Top venues (NeurIPS, ICML, ICLR, ACL, CVPR, SIGIR)
- Released code with documentation
- Reproducible results
- Fair comparison with SOTA

### Common Pitfalls
- Test set contamination / data leakage
- Overfitting to benchmarks
- Cherry-picking results or seeds
- Inadequate baselines
- Ignoring computational cost
- Not reporting variance"""


ENVIRONMENTAL_SKILL = """## Environmental Science Domain Expertise

### Key Methodological Standards
- **Temporal Coverage**: Long-term data series for climate/ecology
- **Spatial Scale**: Local vs. regional vs. global patterns
- **Uncertainty Quantification**: Model ensembles, confidence intervals
- **Attribution**: Natural variability vs. anthropogenic forcing

### Statistical Conventions
- Trend analysis with appropriate tests (Mann-Kendall)
- Spatial autocorrelation (Moran's I)
- Time series decomposition (seasonal, trend, residual)
- Ensemble statistics for projections

### Domain Terminology
- IPCC scenarios (RCPs, SSPs)
- Carbon cycle (sinks, sources, fluxes)
- Ecosystem services
- Tipping points, feedback loops

### Quality Indicators
- IPCC-cited literature
- Nature Climate Change, Global Change Biology
- Observational validation of models
- Long-term monitoring datasets (NOAA, NASA)

### Common Pitfalls
- Confusing weather with climate
- Extrapolating beyond model validity
- Ignoring spatial heterogeneity
- Publication bias toward alarming results
- Cherry-picking time periods"""


SOCIAL_SCIENCE_SKILL = """## Social Science Domain Expertise

### Key Methodological Standards
- **Mixed Methods**: Quantitative + qualitative triangulation
- **Sampling**: Representative samples, stratification
- **Validity**: Construct, internal, external validity
- **Reflexivity**: Researcher positionality acknowledged

### Statistical Conventions
- Survey weights for population inference
- Structural equation modeling (SEM)
- Multilevel/hierarchical models for nested data
- Qualitative coding reliability (inter-rater)

### Domain Terminology
- Operationalization of constructs
- Latent vs. observed variables
- Thick description (qualitative)
- Generalizability vs. transferability

### Quality Indicators
- Peer-reviewed social science journals
- IRB/ethics approval
- Transparent sampling methodology
- Pre-registration for confirmatory research

### Common Pitfalls
- Selection bias in sampling
- Social desirability in self-reports
- Ecological fallacy
- WEIRD population bias
- Researcher bias in qualitative work
- Overgeneralization from case studies"""
