# -*- coding: utf-8 -*-
"""Prompt templates for public dataset search."""

DATASET_QUERY_PROMPT = """You are a research data scientist looking for publicly available datasets.

## Research Objective
{objective}

## Context from Reference Papers
{seed_context}

## Instructions
Generate {num_queries} search queries to find relevant datasets on HuggingFace Datasets Hub.
Think about:
- What kind of data would be needed to investigate this objective?
- What domain/field does this research belong to?
- What specific variables or measurements would be useful?
- What geographic or temporal scope is relevant?
- If the reference papers above mention specific datasets, databases, or statistical agencies (e.g., ISTAT, Eurostat, FRED), prioritize searching for those exact resources.

Keep queries short (2-5 words) for best API results.
Focus on DATA topics, not paper titles.

Return ONLY a JSON array of query strings, nothing else.
Example: ["italian economy firms", "macroeconomic quarterly data", "firm pricing panel"]"""

DATASET_RELEVANCE_PROMPT = """You are a research data scientist evaluating whether a dataset is suitable for a research objective.

## Research Objective
{objective}

## Dataset Candidate
- Name: {dataset_name}
- Description: {dataset_description}
- Tags: {dataset_tags}
- Downloads: {dataset_downloads}
- Size: {dataset_size}

## Instructions
Score this dataset's relevance from 0.0 to 1.0 for the research objective.

Consider:
- Does it contain the types of variables needed? (0.3 weight)
- Is it in the right domain/field? (0.3 weight)
- Is the data format usable (tabular CSV/parquet preferred)? (0.2 weight)
- Is it large enough to be meaningful? (0.1 weight)
- Is it well-documented and maintained? (0.1 weight)

Respond with ONLY a JSON object:
{{"score": 0.75, "reason": "Brief explanation of why this score"}}"""