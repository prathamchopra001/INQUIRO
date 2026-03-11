# -*- coding: utf-8 -*-
"""Prompt template for objective analysis — determines the best run mode."""

OBJECTIVE_ANALYSIS_PROMPT = """You are a research planning assistant. Analyze the following research objective and determine what type of research pipeline is needed.

## Research Objective
{objective}

## Available Dataset
{dataset_info}

## Seed Papers Loaded
{seed_info}

## Available Modes
1. **literature** — Literature survey only. Search papers, extract findings, synthesize knowledge. No data analysis or dataset searching. Best for: systematic reviews, literature surveys, theoretical research, when the user explicitly wants papers only.

2. **data** — Data analysis only. Analyze the provided dataset with code execution. No literature search. Best for: pure data analysis tasks, statistical modeling on a known dataset, when papers are not needed.

3. **full** — Both literature search AND data analysis. Search for datasets if none provided. Best for: empirical research that needs both data analysis and literature context, replication studies, any research that benefits from both code and papers.

## Instructions
Based on the objective, determine which mode is most appropriate.

Consider:
- Does the objective mention "literature review", "survey", "review papers"? → literature
- Does the objective mention analyzing a specific dataset or running experiments on data? → data or full
- Does the objective require both understanding prior work AND running analysis? → full
- Is a dataset already provided? If yes and the objective needs both, → full. If yes and no papers needed, → data.
- If no dataset and the objective is about finding/reviewing knowledge → literature

Respond with ONLY a JSON object:
{{"mode": "literature", "reasoning": "Brief explanation of why this mode", "confidence": 0.9}}

The mode MUST be one of: "literature", "data", "full"
Confidence should be 0.0-1.0 (how sure you are about this choice)."""
