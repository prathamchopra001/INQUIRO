"""
Report Generator for Inquiro.

Takes the top discoveries from the World Model and uses the LLM
to write a human-readable scientific report with proper citations.

Think of this as the "pen" — everything else built the knowledge,
this component writes it up as something a researcher can read.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils.llm_client import LLMClient
from src.world_model.world_model import WorldModel
from src.synthesis.cross_finding import CrossFindingSynthesizer
from src.reports.latex_compiler import LatexCompiler, LatexConfig, check_latex_installation
from config.prompts.academic_paper import (
    ABSTRACT_PROMPT,
    INTRODUCTION_PROMPT,
    METHODS_PROMPT,
    RESULTS_INTRO_PROMPT,
    LIMITATIONS_PROMPT,
)

logger = logging.getLogger(__name__)


def _clean_objective_text(objective: str) -> str:
    """
    Clean formatting artifacts from objective text.
    
    Removes common issues like:
    - `+4`, `+1`, `+2` annotations from source documents
    - Excessive whitespace and blank lines
    - Leading/trailing whitespace on lines
    - "Title:" and "Objective:" prefixes (for cleaner display)
    
    Args:
        objective: Raw objective text
        
    Returns:
        Cleaned objective text suitable for report display
    """
    if not objective:
        return objective
    
    # Remove +N annotations (e.g., "+4", "+1", "+2" that appear in source docs)
    # Multiple patterns to catch various formats:
    # - "+4" at end of line
    # - "+1 " with trailing space
    # - "(+2)" in parentheses
    # - Multiple consecutive like "+4+1"
    cleaned = re.sub(r'\s*\+\d{1,2}(?=\s|$|\+)', '', objective)
    cleaned = re.sub(r'\(\+\d{1,2}\)', '', cleaned)
    
    # Remove other common artifacts
    # - Bullet markers that got garbled: "•", "○", "■"
    cleaned = re.sub(r'^\s*[•○■►▪]\s*', '', cleaned, flags=re.MULTILINE)
    
    # Remove "Title:" and "Objective:" prefixes if they exist on their own lines
    # This makes the display cleaner - we already have section headers
    lines = cleaned.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just "Title:" or "Objective:" labels
        if stripped.lower() in ['title:', 'objective:']:
            continue
        # Remove "Title: " or "Objective: " prefix from the start of a line
        if stripped.lower().startswith('title:'):
            stripped = stripped[6:].strip()
        elif stripped.lower().startswith('objective:'):
            stripped = stripped[10:].strip()
        cleaned_lines.append(stripped)
    cleaned = '\n'.join(cleaned_lines)
    
    # Normalize whitespace: collapse multiple spaces to single space
    cleaned = re.sub(r' +', ' ', cleaned)
    
    # Normalize newlines: collapse 3+ newlines to 2
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in cleaned.split('\n')]
    cleaned = '\n'.join(lines)
    
    # Remove leading/trailing whitespace from entire text
    cleaned = cleaned.strip()
    
    # Remove empty lines at start/end
    while cleaned.startswith('\n'):
        cleaned = cleaned[1:]
    while cleaned.endswith('\n'):
        cleaned = cleaned[:-1]
    
    return cleaned


REPORT_PROMPT = """You are a scientific writer synthesizing research findings into a clear discovery report.

## Research Objective
{objective}

## Finding Type
{finding_type}

## Top Discovery
{finding_claim}

## Supporting Evidence Chain
{evidence_chain}

## Related Findings
{related_findings}

## Instructions
Write a focused scientific narrative (3-5 paragraphs) for this discovery that:
1. States the key finding clearly in the opening sentence
2. Explains the evidence that supports it
3. Notes any caveats or confidence levels
4. Connects it to the broader research objective
5. Uses precise, scientific language

CRITICAL FRAMING RULES based on Finding Type:
- If Finding Type is "data_analysis": This is INQUIRO's own discovery from analyzing the dataset.
  Write as "Our analysis revealed..." or "The data shows..."
- If Finding Type is "literature": This is a finding from a published paper, NOT our own discovery.
  Write as "Prior work by [authors] demonstrated..." or "[Paper title] reported that..."
  NEVER present literature findings as if INQUIRO discovered them.
  NEVER say "This work represents..." or "We found..." for literature findings.

Do NOT use bullet points. Write in flowing paragraphs.
Do NOT include a title — just the narrative text."""


EXECUTIVE_SUMMARY_PROMPT = """You are a scientific writer creating an executive summary for a research report.

## Research Objective
{objective}

## Key Discoveries (ranked by evidence strength)
{discoveries_summary}

## Statistics
- Cycles completed: {cycles_completed}
- Total findings: {total_findings}
- Total relationships: {total_relationships}

## Instructions
Write a concise executive summary (2-3 paragraphs) that:
1. States the research objective in one sentence
2. Summarizes the most significant discoveries
3. Notes the overall confidence and completeness of the investigation
4. Highlights any gaps or areas needing further research

Write in flowing paragraphs. No bullet points. Be precise and scientific."""


CONCLUSION_PROMPT = """You are a scientific writer creating the conclusion for a research report.

## Research Objective
{objective}

## Key Discoveries
{discoveries_summary}

## Literature Findings
{literature_summary}

## Data Analysis Available
{has_data_analysis}

## Instructions
Write a conclusion (2-3 paragraphs) that:
1. Restates the main findings in context of the objective
2. Notes limitations of this autonomous analysis
3. Suggests concrete next steps for future research

CRITICAL FRAMING RULES:
- If Data Analysis Available is "No": This was a LITERATURE REVIEW ONLY.
  Do NOT claim that experiments were run, results were verified, or that
  "replacing pricing rules improved forecasting." Only summarize what
  published papers reported. Say "Prior literature suggests..." not "We found..."
- If Data Analysis Available is "Yes": You may describe INQUIRO's own data
  analysis results using "Our analysis showed..." language.
- NEVER include a references list in the conclusion — there is a separate References section.

Be honest about what was and wasn't addressed. No bullet points.""" 


class ReportGenerator:
    """
    Generates human-readable scientific reports from World Model discoveries.

    Usage:
        generator = ReportGenerator(llm, world_model, output_dir="./outputs")
        report_path = generator.generate_report(objective, cycles_completed=5)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        world_model: WorldModel,
        output_dir: str = "./outputs",
        top_n_discoveries: int = 5,
        top_n_literature: int = 3,
    ):
        self.llm = llm_client
        self.world_model = world_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_n_discoveries = top_n_discoveries
        self.top_n_literature = top_n_literature

    # =========================================================================
    # LITERATURE FINDINGS HELPER
    # =========================================================================

    def _get_top_literature_findings(self, n: int = 3, max_per_source: int = 1) -> list:
        """
        Get the top N literature findings from DIVERSE sources.
        
        This ensures the Literature Context section cites multiple papers,
        not just multiple findings from one paper. Each source paper can
        contribute at most `max_per_source` findings.
        
        Args:
            n: Number of literature findings to return
            max_per_source: Maximum findings from same paper (default 1)
            
        Returns:
            List of dicts with 'finding' and 'score' keys
        """
        all_findings = self.world_model.get_all_findings()
        
        # Filter to literature findings only
        lit_findings = [
            f for f in all_findings
            if getattr(f.finding_type, "value", str(f.finding_type)) == "literature"
        ]
        
        if not lit_findings:
            return []
        
        # Score by confidence (literature findings rarely have support relationships)
        scored = []
        for finding in lit_findings:
            # Count supporting relationships for this finding
            support_count = sum(
                1 for _, _, d in self.world_model.graph.in_edges(finding.id, data=True)
                if d.get("relationship_type") == "supports"
            )
            score = support_count + finding.confidence
            
            # Extract source key for deduplication
            source_key = self._extract_source_key(finding)
            
            scored.append({
                "finding": finding, 
                "score": score,
                "source_key": source_key,
            })
        
        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        # Select top N with source diversity (max_per_source from each paper)
        selected = []
        source_counts = {}  # Track how many from each source
        
        for item in scored:
            source_key = item["source_key"]
            current_count = source_counts.get(source_key, 0)
            
            if current_count < max_per_source:
                selected.append({"finding": item["finding"], "score": item["score"]})
                source_counts[source_key] = current_count + 1
                
                if len(selected) >= n:
                    break
        
        # Log diversity stats
        unique_sources = len(source_counts)
        logger.info(f"  Literature diversity: {len(selected)} findings from {unique_sources} unique sources")
        
        return selected
    
    def _extract_source_key(self, finding) -> str:
        """
        Extract a normalized key from a finding's source for deduplication.
        
        For papers: uses DOI if available, otherwise title
        For notebooks: uses notebook filename
        """
        source = finding.source
        if source is None:
            return "unknown"
        
        # Handle dict-style source
        if isinstance(source, dict):
            source_type = source.get("type", "")
            if source_type == "paper":
                # Prefer DOI, fall back to title
                doi = source.get("doi", "")
                if doi:
                    return f"doi:{doi.lower()}"
                title = source.get("title", "")
                return f"title:{title.lower().strip()[:50]}"
            elif source_type == "notebook":
                path = source.get("path", "")
                return f"notebook:{Path(path).name}"
            return str(source).lower()[:50]
        
        # Handle object-style source (Pydantic model)
        source_type = getattr(source, "type", "")
        if source_type == "paper":
            doi = getattr(source, "doi", "")
            if doi:
                return f"doi:{doi.lower()}"
            title = getattr(source, "title", "")
            return f"title:{title.lower().strip()[:50]}"
        elif source_type == "notebook":
            path = getattr(source, "path", "")
            return f"notebook:{Path(path).name}"
        
        return str(source).lower()[:50]

    # =========================================================================
    # COMPREHENSIVE REFERENCES COLLECTION
    # =========================================================================

    def _collect_all_references(self) -> list:
        """
        Collect ALL paper references from ALL literature findings.
        
        This ensures the References section includes every paper that
        contributed to any finding, not just the top discoveries.
        
        Returns:
            List of formatted citation strings, deduplicated by title
        """
        all_findings = self.world_model.get_all_findings()
        
        # Filter to literature findings
        lit_findings = [
            f for f in all_findings
            if getattr(f.finding_type, "value", str(f.finding_type)) == "literature"
        ]
        
        # Extract citations from each finding's source
        seen_titles = {}  # title_key -> full citation
        
        for finding in lit_findings:
            source = finding.source
            if not source:
                continue
            
            # Handle dict-style source
            if isinstance(source, dict):
                source_type = source.get("type", "")
                if source_type == "paper":
                    title = source.get("title", "Unknown Paper")
                    doi = source.get("doi", "")
                    authors = source.get("authors", "")
                    year = source.get("year", "")
                    
                    # Build citation
                    cite = f"[Literature] {title}"
                    if authors:
                        cite = f"[Literature] {authors}. {title}"
                    if year:
                        cite += f" ({year})"
                    if doi:
                        cite += f" DOI: {doi}"
                    
                    title_key = title.lower().strip()[:50]
                    if title_key not in seen_titles or len(cite) > len(seen_titles[title_key]):
                        seen_titles[title_key] = cite
            
            # Handle object-style source (Pydantic model)
            elif hasattr(source, "type"):
                source_type = getattr(source, "type", "")
                if source_type == "paper":
                    title = getattr(source, "title", "Unknown Paper")
                    doi = getattr(source, "doi", "")
                    authors = getattr(source, "authors", "")
                    year = getattr(source, "year", "")
                    
                    cite = f"[Literature] {title}"
                    if authors:
                        cite = f"[Literature] {authors}. {title}"
                    if year:
                        cite += f" ({year})"
                    if doi:
                        cite += f" DOI: {doi}"
                    
                    title_key = title.lower().strip()[:50]
                    if title_key not in seen_titles or len(cite) > len(seen_titles[title_key]):
                        seen_titles[title_key] = cite
        
        # Return sorted by title
        return sorted(seen_titles.values())

    def _generate_findings_summary_table(self, include_literature: bool = True) -> list:
        """
        Generate a summary table of all findings.
        
        Returns:
            List of markdown lines for the table
        """
        all_findings = self.world_model.get_all_findings()
        
        if not all_findings:
            return []
        
        lines = [
            "### Findings Summary",
            "",
            "| # | Type | Claim | Confidence | Cycle |",
            "|---|------|-------|------------|-------|",
        ]
        
        # Sort by cycle then confidence
        sorted_findings = sorted(
            all_findings,
            key=lambda f: (f.cycle, -f.confidence),
        )
        
        for i, f in enumerate(sorted_findings[:20], 1):  # Limit to top 20
            ftype = getattr(f.finding_type, "value", str(f.finding_type))
            type_icon = "📊" if ftype == "data_analysis" else "📚"
            claim_short = f.claim[:60] + "..." if len(f.claim) > 60 else f.claim
            # Escape pipe characters in claim
            claim_short = claim_short.replace("|", "\\|")
            conf = f"{f.confidence:.0%}"
            lines.append(f"| {i} | {type_icon} {ftype} | {claim_short} | {conf} | {f.cycle} |")
        
        if len(sorted_findings) > 20:
            lines.append(f"| ... | ... | *({len(sorted_findings) - 20} more findings)* | ... | ... |")
        
        lines.append("")
        return lines

    # =========================================================================
    # PUBLIC ENTRY POINT
    # =========================================================================

    def generate_report(self, objective: str, cycles_completed: int,
                        is_synthetic_data: bool = False,
                        data_source_info: str = "",
                        paper_format: str = "academic",
                        generate_latex: bool = False,
                        latex_template: str = "plain") -> Union[str, Dict[str, str]]:
        """
        Generate a full discovery report and save it to disk.

        Args:
            objective:         The research question that was investigated
            cycles_completed:  How many cycles ran (for the report header)
            is_synthetic_data: Whether synthetic data was used
            data_source_info:  Description of data sources
            paper_format:      Report format - "academic" (default) or "discovery"
            generate_latex:    If True, also compile to LaTeX/PDF
            latex_template:    LaTeX template ("plain", "arxiv", "neurips", "ieee")

        Returns:
            If generate_latex is False: Path to the saved markdown report file
            If generate_latex is True: Dict with {"markdown": path, "latex": path, "pdf": path or None}
        """
        logger.info("Generating discovery report...")
        
        # Clean the objective text ONCE at the start to remove formatting artifacts
        # like +4, +1, +2 annotations that sometimes appear from source documents.
        # This ensures all prompts and displays receive the cleaned version.
        objective = _clean_objective_text(objective)

        # 1. Get top discoveries from world model
        stats = self.world_model.get_statistics()
        
        # Get top findings (will be mostly data_analysis due to higher scores)
        top_findings = self.world_model.get_top_findings(n=self.top_n_discoveries)
        
        # Diversify: limit to max 2 findings from the same source
        if top_findings:
            seen_sources = {}
            diversified = []
            for item in top_findings:
                finding = item["finding"]
                source_key = ""
                if hasattr(finding, 'source'):
                    src = finding.source
                    source_key = (
                        getattr(src, 'title', '') or 
                        getattr(src, 'path', '') or 
                        str(src)
                    ).lower().strip()
                
                count = seen_sources.get(source_key, 0)
                if count < 2 or not source_key:
                    diversified.append(item)
                    seen_sources[source_key] = count + 1
            
            # Fill remaining slots if needed
            if len(diversified) < self.top_n_discoveries:
                remaining = [i for i in top_findings if i not in diversified]
                diversified.extend(remaining[:self.top_n_discoveries - len(diversified)])
            
            top_findings = diversified[:self.top_n_discoveries]

        # 2. SEPARATELY get top literature findings (guaranteed representation)
        #    This ensures literature context appears even when data findings dominate scores
        top_lit_findings = self._get_top_literature_findings(n=self.top_n_literature)
        
        # Remove any literature findings that are already in top_findings (avoid duplicates)
        existing_ids = {item["finding"].id for item in top_findings}
        top_lit_findings = [item for item in top_lit_findings if item["finding"].id not in existing_ids]
        
        logger.info(f"  Found {len(top_findings)} top findings, {len(top_lit_findings)} additional literature findings")

        if not top_findings and not top_lit_findings:
            logger.warning("No findings in world model — generating empty report")
            return self._save_report(
                content=self._empty_report(objective),
                objective=objective,
            )

        # 3. Build report sections for top findings (mostly data analysis)
        sections = []
        
        for rank, item in enumerate(top_findings, start=1):
            finding = item["finding"]
            logger.info(f"  Writing narrative for discovery {rank}/{len(top_findings)}...")

            # Get evidence chain for this finding
            evidence_chain = self.world_model.get_evidence_chain(finding.id)

            # Get related findings
            related = (
                self.world_model.get_supporting_findings(finding.id)
                + self.world_model.get_contradicting_findings(finding.id)
            )

            # Generate narrative
            narrative = self._generate_narrative(
                objective=objective,
                finding=finding,
                evidence_chain=evidence_chain,
                related_findings=related,
            )

            # Build citations for this finding
            citations = self._format_citations(finding, evidence_chain)

            sections.append({
                "rank": rank,
                "finding": finding,
                "narrative": narrative,
                "citations": citations,
                "score": item["score"],
            })

        # 4. Build sections for LITERATURE findings (guaranteed separate section)
        lit_sections_extra = []
        
        for rank, item in enumerate(top_lit_findings, start=1):
            finding = item["finding"]
            logger.info(f"  Writing narrative for literature finding {rank}/{len(top_lit_findings)}...")

            evidence_chain = self.world_model.get_evidence_chain(finding.id)
            related = (
                self.world_model.get_supporting_findings(finding.id)
                + self.world_model.get_contradicting_findings(finding.id)
            )

            narrative = self._generate_narrative(
                objective=objective,
                finding=finding,
                evidence_chain=evidence_chain,
                related_findings=related,
            )

            citations = self._format_citations(finding, evidence_chain)

            lit_sections_extra.append({
                "rank": rank,
                "finding": finding,
                "narrative": narrative,
                "citations": citations,
                "score": item["score"],
            })

        # 5. Get research questions for the questions section
        questions = []
        try:
            questions = self.world_model.get_all_questions()
            if questions:
                logger.info(f"  Including {len(questions)} research questions in report")
        except Exception as e:
            logger.debug(f"Could not fetch questions for report: {e}")

        # 6. Cross-finding synthesis for Discussion section
        discussion_text = None
        try:
            all_findings = self.world_model.get_all_findings()
            if len(all_findings) >= 3:
                logger.info(f"  Running cross-finding synthesis on {len(all_findings)} findings...")
                synthesizer = CrossFindingSynthesizer(self.llm)
                
                # Convert Finding objects to dicts for the synthesizer
                findings_for_synthesis = []
                for f in all_findings:
                    findings_for_synthesis.append({
                        "id": f.id,
                        "claim": f.claim,
                        "confidence": f.confidence,
                        "finding_type": getattr(f.finding_type, "value", str(f.finding_type)),
                        "evidence": getattr(f, "evidence", ""),
                        "source": f.source if isinstance(f.source, dict) else {"type": "unknown"},
                    })
                
                result = synthesizer.synthesize_and_narrate(
                    findings=findings_for_synthesis,
                    objective=objective,
                    max_themes=5,
                )
                discussion_text = result.get("discussion_text")
                if discussion_text:
                    logger.info(f"  ✅ Generated Discussion with {len(result.get('themes', []))} themes")
            else:
                logger.info("  Skipping synthesis (fewer than 3 findings)")
        except Exception as e:
            logger.warning(f"Cross-finding synthesis failed: {e}")
            discussion_text = None

        # 7. Assemble full report based on format
        if paper_format == "academic":
            logger.info("  Using academic paper format...")
            report_content = self._assemble_academic_report(
                objective=objective,
                sections=sections,
                lit_sections_extra=lit_sections_extra,
                stats=stats,
                cycles_completed=cycles_completed,
                is_synthetic_data=is_synthetic_data,
                questions=questions,
                discussion_text=discussion_text,
                data_source_info=data_source_info,
            )
        else:
            # Original discovery format
            report_content = self._assemble_report(
                objective=objective,
                sections=sections,
                lit_sections_extra=lit_sections_extra,
                stats=stats,
                cycles_completed=cycles_completed,
                is_synthetic_data=is_synthetic_data,
                questions=questions,
                discussion_text=discussion_text,
            )

        # 8. Save markdown to disk
        markdown_path = self._save_report(content=report_content, objective=objective)
        
        # 9. Optionally compile to LaTeX/PDF
        if not generate_latex:
            return markdown_path
        
        # Compile to LaTeX
        result = {
            "markdown": markdown_path,
            "latex": None,
            "pdf": None,
        }
        
        try:
            latex_config = LatexConfig(
                template=latex_template,
                compile_pdf=True,
                author="INQUIRO Autonomous Research System",
            )
            compiler = LatexCompiler(latex_config)
            
            # Create latex output directory
            latex_dir = self.output_dir / "latex"
            latex_dir.mkdir(parents=True, exist_ok=True)
            
            # Compile
            output_path = compiler.compile_report(
                markdown_path=markdown_path,
                output_dir=str(latex_dir),
                title=objective[:100]
            )
            
            if output_path:
                if output_path.endswith('.pdf'):
                    result["pdf"] = output_path
                    # The .tex file is in the same directory
                    result["latex"] = output_path.replace('.pdf', '.tex')
                else:
                    result["latex"] = output_path
                    
                logger.info(f"LaTeX output: {output_path}")
        except Exception as e:
            logger.warning(f"LaTeX compilation failed: {e}. Markdown report is still available.")
        
        return result

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _generate_narrative(
        self, objective: str, finding, evidence_chain: list, related_findings: list
    ) -> str:
        """Ask the LLM to write a scientific narrative for one discovery.
        
        Falls back to a structured plain-text summary if the LLM is unavailable
        (e.g. quota exhausted), so the report always completes.
        """

        # Format evidence chain for prompt
        if evidence_chain:
            chain_text = "\n".join(
                f"  [{e['depth']}] {e['claim']} "
                f"(confidence: {e['confidence']:.2f}, source: {e['source'].get('type', 'unknown')})"
                for e in evidence_chain
            )
        else:
            chain_text = "No direct supporting evidence chain available."

        # Format related findings
        if related_findings:
            related_text = "\n".join(
                f"  - {f.claim} (confidence: {f.confidence:.2f})"
                for f in related_findings[:5]
            )
        else:
            related_text = "No directly related findings."

        # Get finding_type as string (it may be an enum)
        finding_type_str = (
            getattr(finding.finding_type, "value", str(finding.finding_type))
            if hasattr(finding, 'finding_type') 
            else "unknown"
        )
        
        prompt = REPORT_PROMPT.format(
            objective=objective,
            finding_type=finding_type_str,
            finding_claim=finding.claim,
            evidence_chain=chain_text,
            related_findings=related_text,
        )

        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="report_narrative",
                system="You are a precise scientific writer. Write clearly and concisely.",
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(
                f"LLM narrative generation failed (quota or error): {e}. "
                f"Using structured fallback."
            )
            return self._fallback_narrative(finding, evidence_chain, related_findings)

    def _fallback_narrative(
        self, finding, evidence_chain: list, related_findings: list
    ) -> str:
        """Plain-text narrative used when LLM is unavailable."""
        parts = [finding.claim, ""]

        if finding.evidence:
            parts += [f"Evidence: {finding.evidence}", ""]

        if evidence_chain and len(evidence_chain) > 1:
            parts.append("Supporting evidence:")
            for e in evidence_chain[1:4]:
                parts.append(f"  - {e['claim']} (confidence: {e['confidence']:.2f})")
            parts.append("")

        if related_findings:
            parts.append("Related findings:")
            for f in related_findings[:3]:
                parts.append(f"  - {f.claim}")
            parts.append("")

        finding_type_str = getattr(finding.finding_type, "value", str(finding.finding_type))
        parts.append(
            f"Source type: {finding_type_str} | "
            f"Confidence: {finding.confidence:.0%} | "
            f"Cycle: {finding.cycle}"
        )
        return "\n".join(parts)

    def _format_citations(self, finding, evidence_chain: list) -> List[str]:
        """
        Build a list of citation strings for a finding.

        Data findings cite notebooks, literature findings cite papers.
        """
        citations = []

        # Citation for the finding itself
        source = finding.source
        source_type = source.type if hasattr(source, "type") else source.get("type", "unknown")

        if source_type == "notebook":
            path = source.path if hasattr(source, "path") else source.get("path", "")
            cell = source.cell if hasattr(source, "cell") else source.get("cell", "")
            citations.append(f"[Data] Notebook: `{Path(path).name}`, Cell {cell}")

        elif source_type == "paper":
            title = source.title if hasattr(source, "title") else source.get("title", "Unknown")
            doi = source.doi if hasattr(source, "doi") else source.get("doi", "")
            doi_str = f" DOI: {doi}" if doi else ""
            citations.append(f"[Literature] {title}{doi_str}")

        # Citations from evidence chain (deduplicated)
        seen = set()
        for item in evidence_chain[1:]:  # Skip index 0 (the finding itself)
            src = item.get("source", {})
            src_type = src.get("type", "")
            if src_type == "paper":
                title = src.get("title", "Unknown paper")
                if title not in seen:
                    citations.append(f"[Literature] {title}")
                    seen.add(title)
            elif src_type == "notebook":
                nb_path = src.get("path", "")
                nb_name = Path(nb_path).name if nb_path else "unknown notebook"
                if nb_name not in seen:
                    citations.append(f"[Data] Notebook: `{nb_name}`")
                    seen.add(nb_name)

        return citations

    def _generate_executive_summary(
        self, objective: str, sections: list, stats: dict, cycles_completed: int
    ) -> str:
        """Generate an LLM-written executive summary."""
        discoveries_summary = "\n".join(
            f"  {s['rank']}. {s['finding'].claim[:120]} "
            f"(confidence: {s['finding'].confidence:.0%}, score: {s['score']:.2f})"
            for s in sections
        )
        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            objective=objective,
            discoveries_summary=discoveries_summary,
            cycles_completed=cycles_completed,
            total_findings=stats.get("total_findings", 0),
            total_relationships=stats.get("total_relationships", 0),
        )
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="executive_summary",
                system="You are a precise scientific writer.",
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Executive summary generation failed: {e}")
            return (
                f"This report presents the top {len(sections)} discoveries from "
                f"{cycles_completed} autonomous research cycles analyzing the "
                f"objective. A total of {stats.get('total_findings', 0)} findings "
                f"and {stats.get('total_relationships', 0)} relationships were "
                f"identified."
            )

    def _generate_conclusion(
        self, objective: str, sections: list
    ) -> str:
        """Generate an LLM-written conclusion."""
        discoveries_summary = "\n".join(
            f"  - {s['finding'].claim[:150]}" for s in sections
        )
        # Collect literature findings
        lit_findings = [
            s for s in sections
            if getattr(s["finding"].finding_type, "value", s["finding"].finding_type) == "literature"
        ]
        literature_summary = "\n".join(
            f"  - {s['finding'].claim[:150]}" for s in lit_findings
        ) if lit_findings else "No literature-based findings in this report."

        has_data = "Yes" if any(
            getattr(s["finding"].finding_type, "value", s["finding"].finding_type) == "data_analysis" 
            for s in sections
        ) else "No"

        prompt = CONCLUSION_PROMPT.format(
            objective=objective,
            discoveries_summary=discoveries_summary,
            literature_summary=literature_summary,
            has_data_analysis=has_data,
        )
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="conclusion",
                system="You are a precise scientific writer.",
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Conclusion generation failed: {e}")
            return (
                "This autonomous analysis identified several findings relevant "
                "to the research objective. Further investigation by domain "
                "experts is recommended to validate these results and explore "
                "areas that were not fully addressed."
            )

    def _assemble_report(
        self, objective: str, sections: list, stats: dict,
        cycles_completed: int, is_synthetic_data: bool = False,
        lit_sections_extra: list = None,
        questions: list = None,
        discussion_text: str = None,
    ) -> str:
        """Combine all sections into a properly structured markdown report.
        
        Args:
            sections: Top findings (ranked by score, usually data_analysis)
            lit_sections_extra: Additional literature findings fetched separately
                               to guarantee literature representation
            questions: List of ResearchQuestion objects for the questions section
            discussion_text: Cross-finding synthesis text for Discussion section
        """

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if lit_sections_extra is None:
            lit_sections_extra = []

        # Separate data and literature findings from the main sections
        # Note: finding_type is a FindingType enum, so we compare to .value
        data_sections = [
            s for s in sections 
            if getattr(s["finding"].finding_type, "value", s["finding"].finding_type) == "data_analysis"
        ]
        # Literature from main sections (if any made it by score)
        lit_sections_from_main = [
            s for s in sections 
            if getattr(s["finding"].finding_type, "value", s["finding"].finding_type) == "literature"
        ]
        
        # Combine: literature from main ranking + extra literature findings
        lit_sections = lit_sections_from_main + lit_sections_extra

        # Collect all citations for a references section
        all_citations = []
        for section in sections:
            all_citations.extend(section.get("citations", []))
        for section in lit_sections_extra:
            all_citations.extend(section.get("citations", []))
        
        # Combined sections for executive summary
        all_sections = sections + lit_sections_extra

        # Deduplicate: same paper may appear with and without DOI.
        # Key by the title portion, keep the most informative version.
        seen_titles = {}
        for cite in all_citations:
            # Extract title: strip "[Literature] " or "[Data] Notebook: " prefix
            # and any trailing " DOI: ..." suffix to get the core title
            title_key = cite.split("] ", 1)[-1]  # after "[Literature] " or "[Data] "
            title_key = title_key.split(" DOI:")[0].strip()
            title_key = title_key.lower()

            if title_key not in seen_titles:
                seen_titles[title_key] = cite
            else:
                # Keep the version with more info (longer = has DOI)
                if len(cite) > len(seen_titles[title_key]):
                    seen_titles[title_key] = cite

        unique_citations = list(seen_titles.values())

        lines = []

        # ── Title & Metadata ─────────────────────────────────────────────
        # Create a clean title from the objective
        title = objective.split("\n")[0].strip().rstrip(".")
        if len(title) > 100:
            title = title[:100] + "..."

        lines += [
            f"# {title}",
            "",
            "*Generated by INQUIRO — Autonomous Scientific Research System*",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Generated | {timestamp} |",
            f"| Cycles Completed | {cycles_completed} |",
            f"| Total Findings | {stats.get('total_findings', 0)} |",
            f"| Total Relationships | {stats.get('total_relationships', 0)} |",
            f"| Discoveries in Report | {len(all_sections)} |",
            "",
        ]

        # Synthetic data warning banner
        if is_synthetic_data:
            lines += [
                "> **⚠️ SYNTHETIC DATA NOTICE**",
                "> ",
                "> This analysis was conducted on a **synthetically generated dataset** ",
                "> created by INQUIRO to match the research objective. The dataset was ",
                "> designed to contain realistic distributions and relationships, but ",
                "> **findings reflect model assumptions, not real-world observations.**",
                "> ",
                "> Results should be interpreted as methodological demonstrations — ",
                "> showing *how* the analysis would work with real data, not as ",
                "> empirical discoveries.",
                "",
            ]

        lines += [
            "---",
            "",
        ]

        # ── 1. Executive Summary ──────────────────────────────────────────
        logger.info("  Writing executive summary...")
        exec_summary = self._generate_executive_summary(
            objective, all_sections, stats, cycles_completed
        )
        lines += [
            "## 1. Executive Summary",
            "",
            exec_summary,
            "",
            "---",
            "",
        ]

        # ── 2. Introduction ───────────────────────────────────────────────
        # Note: objective is already cleaned at the start of generate_report()
        
        # Format objective nicely - if multi-line, use blockquote; if single line, use emphasis
        objective_lines = [line for line in objective.split('\n') if line.strip()]
        if len(objective_lines) == 1:
            # Single line objective - use emphasis
            formatted_objective = f"**{objective_lines[0]}**"
        else:
            # Multi-line objective - use blockquote
            formatted_objective = '\n'.join(f"> {line}" for line in objective_lines)
        
        lines += [
            "## 2. Introduction",
            "",
            "### 2.1 Research Objective",
            "",
            formatted_objective,
            "",
            "### 2.2 Methodology",
            "",
            f"This research was conducted autonomously by INQUIRO over "
            f"{cycles_completed} iterative research cycles. Each cycle "
            f"consisted of task generation by an orchestrator agent, parallel "
            f"execution by data analysis and literature search agents, and "
            f"knowledge integration into a structured world model. ",
            "",
            f"The system generated {stats.get('total_findings', 0)} total "
            f"findings and identified {stats.get('total_relationships', 0)} "
            f"relationships between them. The top {len(all_sections)} discoveries "
            f"are presented below, ranked by evidence strength.",
            "",
            "---",
            "",
        ]

        # ── Dynamic section numbering ─────────────────────────────────────
        sec = 3  # sections 1 (exec summary) and 2 (intro) already written

        if data_sections:
            lines += [
                f"## {sec}. Key Discoveries",
                "",
            ]
            for i, section in enumerate(data_sections, 1):
                finding = section["finding"]
                score = section["score"]
                
                # Calculate supporting relationships (score - confidence)
                support_count = max(0, int(score - finding.confidence))

                lines += [
                    f"### {sec}.{i} {finding.claim[:100]}{'...' if len(finding.claim) > 100 else ''}",
                    "",
                    f"**Confidence:** {finding.confidence:.0%} &nbsp;|&nbsp; "
                    f"**Supporting Relationships:** {support_count} &nbsp;|&nbsp; "
                    f"**Cycle:** {finding.cycle}",
                    "",
                    section["narrative"],
                    "",
                ]
                
                # Include figures if present (data analysis findings)
                if hasattr(finding, 'figures') and finding.figures:
                    lines.append("**Figures:**")
                    lines.append("")
                    for fig_path in finding.figures:
                        fig_name = Path(fig_path).name
                        lines.append(f"![{fig_name}]({fig_path})")
                        lines.append("")
                        lines.append(f"*Figure: {fig_name}*")
                        lines.append("")

                if section["citations"]:
                    lines.append("**Sources:**")
                    for cite in section["citations"]:
                        lines.append(f"- {cite}")
                    lines.append("")

                lines.append("")

            lines += ["---", ""]
            sec += 1

        if lit_sections:
            lines += [
                f"## {sec}. Literature Context",
                "",
            ]
            for i, section in enumerate(lit_sections, 1):
                finding = section["finding"]
                score = section["score"]
                
                # Calculate supporting relationships (score - confidence)
                support_count = max(0, int(score - finding.confidence))

                lines += [
                    f"### {sec}.{i} {finding.claim[:100]}{'...' if len(finding.claim) > 100 else ''}",
                    "",
                    f"**Confidence:** {finding.confidence:.0%} &nbsp;|&nbsp; "
                    f"**Supporting Relationships:** {support_count} &nbsp;|&nbsp; "
                    f"**Cycle:** {finding.cycle}",
                    "",
                    section["narrative"],
                    "",
                ]

                if section["citations"]:
                    lines.append("**Sources:**")
                    for cite in section["citations"]:
                        lines.append(f"- {cite}")
                    lines.append("")

                lines.append("")

            lines += ["---", ""]
            sec += 1

        if not data_sections and not lit_sections:
            lines += [
                f"## {sec}. Discoveries",
                "",
                "No significant discoveries met the reporting threshold.",
                "",
                "---",
                "",
            ]
            sec += 1

        # ── Research Questions Section ────────────────────────────────────
        if questions:
            question_lines = self._generate_question_section(questions)
            if question_lines:
                # Update section number in the generated content
                question_lines[0] = f"## {sec}. Research Questions Addressed"
                lines.extend(question_lines)
                sec += 1

        # ── Discussion (Cross-Finding Synthesis) ──────────────────────────
        if discussion_text:
            lines += [
                f"## {sec}. Discussion",
                "",
                "The following themes emerged from synthesizing the individual findings:",
                "",
                discussion_text,
                "",
                "---",
                "",
            ]
            sec += 1

        # ── Conclusion ────────────────────────────────────────────────────
        logger.info("  Writing conclusion...")
        conclusion = self._generate_conclusion(objective, all_sections)
        lines += [
            f"## {sec}. Conclusion",
            "",
            conclusion,
            "",
            "---",
            "",
        ]
        sec += 1

        # ── References ────────────────────────────────────────────────────
        # Use comprehensive references from ALL literature findings
        all_references = self._collect_all_references()
        
        # Merge with any citations we found in the report sections
        for cite in unique_citations:
            title_key = cite.split("] ", 1)[-1].split(" DOI:")[0].strip().lower()[:50]
            if not any(title_key in ref.lower() for ref in all_references):
                all_references.append(cite)
        
        if all_references:
            lines += [
                f"## {sec}. References",
                "",
            ]
            for i, cite in enumerate(all_references, 1):
                lines.append(f"{i}. {cite}")
            lines.append("")
            lines += ["---", ""]

        # ── 7. Appendix: Methodology Details ─────────────────────────────
        lines += [
            "## Appendix: System Details",
            "",
            "This report was generated by INQUIRO, an autonomous scientific "
            "research system inspired by the Inquiro paper (arXiv:2511.02824). "
            "Each discovery was identified through iterative cycles of data "
            "analysis and literature search, coordinated by an orchestrator agent.",
            "",
            "Confidence scores reflect the strength of supporting evidence. "
            "All data analysis findings are traceable to specific Jupyter "
            "notebook cells. Literature findings include paper citations.",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Research Cycles | {cycles_completed} |",
            f"| Total Findings | {stats.get('total_findings', 0)} |",
            f"| Total Relationships | {stats.get('total_relationships', 0)} |",
            f"| Data Discoveries | {len(data_sections)} |",
            f"| Literature Discoveries | {len(lit_sections)} |",
        ]

        return "\n".join(lines)

    def _assemble_academic_report(
        self, objective: str, sections: list, stats: dict,
        cycles_completed: int, is_synthetic_data: bool = False,
        lit_sections_extra: list = None,
        questions: list = None,
        discussion_text: str = None,
        data_source_info: str = "",
    ) -> str:
        """Assemble report in academic research paper format.
        
        Structure:
        - Abstract
        - 1. Introduction
        - 2. Methods
        - 3. Results
        - 4. Discussion
        - 5. Conclusion
        - References
        - Appendix
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if lit_sections_extra is None:
            lit_sections_extra = []

        # Separate data and literature findings
        data_sections = [
            s for s in sections 
            if getattr(s["finding"].finding_type, "value", s["finding"].finding_type) == "data_analysis"
        ]
        lit_sections_from_main = [
            s for s in sections 
            if getattr(s["finding"].finding_type, "value", s["finding"].finding_type) == "literature"
        ]
        lit_sections = lit_sections_from_main + lit_sections_extra
        all_sections = sections + lit_sections_extra

        # Collect citations
        all_citations = []
        for section in sections + lit_sections_extra:
            all_citations.extend(section.get("citations", []))
        
        # Deduplicate citations
        seen_titles = {}
        for cite in all_citations:
            title_key = cite.split("] ", 1)[-1].split(" DOI:")[0].strip().lower()
            if title_key not in seen_titles or len(cite) > len(seen_titles[title_key]):
                seen_titles[title_key] = cite
        unique_citations = list(seen_titles.values())

        # Count questions
        questions_answered = sum(1 for q in (questions or []) 
                                  if getattr(q.status, 'value', str(q.status)) == 'answered')
        questions_total = len(questions or [])

        lines = []

        # ── Title ─────────────────────────────────────────────────────────
        title = objective.split("\n")[0].strip().rstrip(".")
        if len(title) > 100:
            title = title[:100] + "..."

        lines += [
            f"# {title}",
            "",
            "*Generated by INQUIRO — Autonomous Scientific Research System*",
            "",
            f"**Generated:** {timestamp} | **Cycles:** {cycles_completed} | "
            f"**Findings:** {stats.get('total_findings', 0)} | "
            f"**Questions Addressed:** {questions_answered}/{questions_total}",
            "",
        ]

        # Synthetic data warning
        if is_synthetic_data:
            lines += [
                "> **⚠️ Note:** This analysis used synthetically generated data. "
                "Findings reflect model assumptions, not real-world observations.",
                "",
            ]

        lines += ["---", ""]

        # ── Abstract ──────────────────────────────────────────────────────
        logger.info("  Writing abstract...")
        abstract = self._generate_abstract(
            objective, all_sections, stats, cycles_completed, 
            questions_answered, questions_total
        )
        lines += [
            "## Abstract",
            "",
            abstract,
            "",
            "---",
            "",
        ]

        # ── 1. Introduction ───────────────────────────────────────────────
        logger.info("  Writing introduction...")
        introduction = self._generate_introduction(
            objective, all_sections, bool(lit_sections)
        )
        lines += [
            "## 1. Introduction",
            "",
            introduction,
            "",
            "---",
            "",
        ]

        # ── 2. Methods ────────────────────────────────────────────────────
        logger.info("  Writing methods...")
        methods = self._generate_methods(
            stats, cycles_completed, len(data_sections), 
            len(lit_sections), data_source_info
        )
        lines += [
            "## 2. Methods",
            "",
            methods,
            "",
            "---",
            "",
        ]

        # ── 3. Results ────────────────────────────────────────────────────
        lines += [
            "## 3. Results",
            "",
        ]

        # Results intro
        results_intro = self._generate_results_intro(
            stats, len(data_sections), len(lit_sections),
            questions_answered, questions_total
        )
        lines += [results_intro, ""]

        # Add findings summary table
        summary_table = self._generate_findings_summary_table()
        if summary_table:
            lines.extend(summary_table)

        # 3.1 Key Discoveries (Data Analysis)
        if data_sections:
            lines += [
                "### 3.1 Key Discoveries",
                "",
            ]
            for i, section in enumerate(data_sections, 1):
                finding = section["finding"]
                lines += [
                    f"**Discovery {i}: {finding.claim[:80]}{'...' if len(finding.claim) > 80 else ''}**",
                    "",
                    f"*Confidence: {finding.confidence:.0%} | Cycle: {finding.cycle}*",
                    "",
                    section["narrative"],
                    "",
                ]
                
                # Include figures if present
                if hasattr(finding, 'figures') and finding.figures:
                    lines.append("**Figures:**")
                    lines.append("")
                    for fig_path in finding.figures:
                        fig_name = Path(fig_path).name
                        # Use relative path for markdown image embedding
                        lines.append(f"![{fig_name}]({fig_path})")
                        lines.append("")
                        lines.append(f"*Figure: {fig_name}*")
                        lines.append("")

        # 3.2 Literature Findings
        if lit_sections:
            subsec = "3.2" if data_sections else "3.1"
            lines += [
                f"### {subsec} Literature Findings",
                "",
            ]
            for i, section in enumerate(lit_sections, 1):
                finding = section["finding"]
                lines += [
                    f"**Finding {i}: {finding.claim[:80]}{'...' if len(finding.claim) > 80 else ''}**",
                    "",
                    f"*Confidence: {finding.confidence:.0%} | Cycle: {finding.cycle}*",
                    "",
                    section["narrative"],
                    "",
                ]

        # 3.3 Research Question Coverage
        if questions:
            subsec_num = "3.3" if data_sections and lit_sections else ("3.2" if data_sections or lit_sections else "3.1")
            question_summary = self._generate_question_summary(questions)
            if question_summary:
                lines += [
                    f"### {subsec_num} Research Question Coverage",
                    "",
                    question_summary,
                    "",
                ]

        lines += ["---", ""]

        # ── 4. Discussion ─────────────────────────────────────────────────
        lines += [
            "## 4. Discussion",
            "",
        ]

        if discussion_text:
            lines += [
                "### 4.1 Key Themes",
                "",
                discussion_text,
                "",
            ]

        # Limitations
        logger.info("  Writing limitations...")
        limitations = self._generate_limitations(
            is_synthetic_data, bool(data_source_info),
            bool(data_sections), not data_sections
        )
        subsec = "4.2" if discussion_text else "4.1"
        lines += [
            f"### {subsec} Limitations",
            "",
            limitations,
            "",
            "---",
            "",
        ]

        # ── 5. Conclusion ─────────────────────────────────────────────────
        logger.info("  Writing conclusion...")
        conclusion = self._generate_conclusion(objective, all_sections)
        lines += [
            "## 5. Conclusion",
            "",
            conclusion,
            "",
            "---",
            "",
        ]

        # ── References ────────────────────────────────────────────────────
        # Use comprehensive references from ALL literature findings
        all_references = self._collect_all_references()
        
        # Merge with any citations we found in the report sections
        # (to catch any notebooks or other sources)
        for cite in unique_citations:
            title_key = cite.split("] ", 1)[-1].split(" DOI:")[0].strip().lower()[:50]
            if not any(title_key in ref.lower() for ref in all_references):
                all_references.append(cite)
        
        if all_references:
            lines += [
                "## References",
                "",
            ]
            for i, cite in enumerate(all_references, 1):
                lines.append(f"{i}. {cite}")
            lines += ["", "---", ""]

        # ── Appendix ──────────────────────────────────────────────────────
        lines += [
            "## Appendix: Research Details",
            "",
            "### A.1 System Configuration",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Research Cycles | {cycles_completed} |",
            f"| Total Findings | {stats.get('total_findings', 0)} |",
            f"| Total Relationships | {stats.get('total_relationships', 0)} |",
            f"| Data Analysis Findings | {len(data_sections)} |",
            f"| Literature Findings | {len(lit_sections)} |",
            f"| Research Questions | {questions_total} |",
            f"| Questions Answered | {questions_answered} |",
            "",
        ]

        # Full question details in appendix
        if questions:
            lines += [
                "### A.2 Research Questions Detail",
                "",
            ]
            for i, q in enumerate(questions, 1):
                status = getattr(q.status, 'value', str(q.status))
                status_emoji = "✅" if status == "answered" else ("🔶" if status == "partial" else "❓")
                lines += [
                    f"**Q{i}.** {q.question_text}",
                    f"*Status: {status_emoji} {status.title()}*",
                    "",
                ]

        lines += [
            "### A.3 Methodology",
            "",
            "This research was conducted using INQUIRO, an autonomous AI scientist system "
            "inspired by arXiv:2511.02824. The system iteratively generates research tasks, "
            "executes data analysis in sandboxed environments, searches scientific literature "
            "across multiple databases (Semantic Scholar, ArXiv, PubMed, OpenAlex), and "
            "synthesizes findings into a structured knowledge graph.",
            "",
            "All findings were validated using ScholarEval quality scoring and deduplicated "
            "using Jaccard similarity. Cross-finding synthesis identified overarching themes "
            "using semantic analysis of the complete finding corpus.",
        ]

        return "\n".join(lines)

    def _generate_abstract(self, objective: str, sections: list, stats: dict,
                           cycles_completed: int, questions_answered: int, 
                           questions_total: int) -> str:
        """Generate academic-style abstract."""
        findings_summary = "\n".join(
            f"- {s['finding'].claim[:100]} (confidence: {s['finding'].confidence:.0%})"
            for s in sections[:5]
        )
        
        prompt = ABSTRACT_PROMPT.format(
            objective=objective,
            findings_summary=findings_summary,
            cycles_completed=cycles_completed,
            total_findings=stats.get('total_findings', 0),
            questions_answered=questions_answered,
            questions_total=questions_total,
        )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="report_writing",
                system="You are a scientific writer creating a research paper abstract.",
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Abstract generation failed: {e}")
            return (
                f"This study investigated {objective.split('.')[0].lower()}. "
                f"Using an autonomous AI research system, {stats.get('total_findings', 0)} "
                f"findings were generated across {cycles_completed} research cycles. "
                f"The analysis addressed {questions_answered} of {questions_total} research questions."
            )

    def _generate_introduction(self, objective: str, sections: list, 
                                has_literature: bool) -> str:
        """Generate academic-style introduction."""
        findings_preview = "\n".join(
            f"- {s['finding'].claim[:80]}"
            for s in sections[:3]
        )
        
        prompt = INTRODUCTION_PROMPT.format(
            objective=objective,
            findings_preview=findings_preview,
            has_literature="Yes" if has_literature else "No",
        )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="report_writing",
                system="You are a scientific writer creating a paper introduction.",
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Introduction generation failed: {e}")
            return (
                f"This research addresses: {objective}\n\n"
                f"The investigation was conducted using INQUIRO, an autonomous AI research "
                f"system that iteratively analyzes data and literature to generate scientific insights."
            )

    def _generate_methods(self, stats: dict, cycles_completed: int,
                          data_count: int, lit_count: int, 
                          data_source_info: str) -> str:
        """Generate academic-style methods section."""
        data_sources = (
            "Literature databases: Semantic Scholar, ArXiv, PubMed, OpenAlex, CrossRef, CORE, Dimensions. "
            "PDF processing via PyMuPDF with semantic chunking. "
            "Vector embeddings via sentence-transformers (all-MiniLM-L6-v2). "
            "RAG retrieval via ChromaDB."
        )
        
        if data_source_info:
            data_sources = f"Dataset: {data_source_info}. " + data_sources
        
        prompt = METHODS_PROMPT.format(
            cycles_completed=cycles_completed,
            total_findings=stats.get('total_findings', 0),
            data_findings_count=data_count,
            lit_findings_count=lit_count,
            total_relationships=stats.get('total_relationships', 0),
            dataset_info=data_source_info or "Literature-only mode (no dataset provided)",
            data_sources=data_sources,
        )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="report_writing",
                system="You are a scientific writer describing research methodology.",
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Methods generation failed: {e}")
            return (
                f"This research was conducted using INQUIRO over {cycles_completed} iterative cycles. "
                f"The system generated {stats.get('total_findings', 0)} findings "
                f"({data_count} from data analysis, {lit_count} from literature review) "
                f"and identified {stats.get('total_relationships', 0)} relationships between them."
            )

    def _generate_results_intro(self, stats: dict, data_count: int, 
                                 lit_count: int, questions_answered: int,
                                 questions_total: int) -> str:
        """Generate brief results section introduction."""
        prompt = RESULTS_INTRO_PROMPT.format(
            total_findings=stats.get('total_findings', 0),
            data_count=data_count,
            lit_count=lit_count,
            questions_answered=questions_answered,
            questions_total=questions_total,
        )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="report_writing",
                max_tokens=300,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Results intro generation failed: {e}")
            return (
                f"The autonomous analysis produced {stats.get('total_findings', 0)} validated findings, "
                f"comprising {data_count} data analysis discoveries and {lit_count} literature findings. "
                f"Overall, {questions_answered} of {questions_total} research questions were addressed."
            )

    def _generate_limitations(self, is_synthetic: bool, has_dataset: bool,
                               has_data_analysis: bool, literature_only: bool) -> str:
        """Generate limitations discussion."""
        prompt = LIMITATIONS_PROMPT.format(
            is_synthetic="Yes" if is_synthetic else "No",
            has_dataset="Yes" if has_dataset else "No",
            has_data_analysis="Yes" if has_data_analysis else "No",
            literature_only="Yes" if literature_only else "No",
        )
        
        try:
            response = self.llm.complete_for_role(
                prompt=prompt,
                role="report_writing",
                max_tokens=500,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Limitations generation failed: {e}")
            limitations = []
            if is_synthetic:
                limitations.append("findings are based on synthetic data and reflect model assumptions")
            if literature_only:
                limitations.append("this analysis relied solely on literature review without original data analysis")
            limitations.append("autonomous extraction may miss nuances a human researcher would capture")
            return "This analysis has several limitations: " + "; ".join(limitations) + "."

    def _generate_question_summary(self, questions: list) -> str:
        """Generate detailed question coverage summary for Results section."""
        if not questions:
            return ""
        
        answered = sum(1 for q in questions 
                       if getattr(q.status, 'value', str(q.status)) == 'answered')
        partial = sum(1 for q in questions 
                      if getattr(q.status, 'value', str(q.status)) == 'partial')
        unanswered = len(questions) - answered - partial
        
        lines = [
            f"The research objective was decomposed into {len(questions)} specific questions. "
            f"Of these, {answered} were fully answered, {partial} were partially addressed, "
            f"and {unanswered} remain open for future investigation.",
            "",
            "| # | Question | Priority | Status | Evidence |",
            "|---|----------|----------|--------|----------|",
        ]
        
        # Build table rows
        for i, q in enumerate(questions, 1):
            status = getattr(q.status, 'value', str(q.status))
            priority = getattr(q.priority, 'value', str(q.priority))
            
            # Emoji indicators
            status_emoji = "✅" if status == "answered" else ("🔶" if status == "partial" else "❓")
            priority_emoji = "🔴" if priority == "high" else ("🟡" if priority == "medium" else "🟢")
            
            # Truncate question text
            q_text = q.question_text[:60] + "..." if len(q.question_text) > 60 else q.question_text
            q_text = q_text.replace("|", "\\|")  # Escape pipes
            
            evidence = f"{q.evidence_count} findings" if q.evidence_count else "—"
            
            lines.append(f"| {i} | {q_text} | {priority_emoji} {priority} | {status_emoji} {status} | {evidence} |")
        
        lines.append("")
        
        # Show answered questions with brief summaries
        answered_qs = [q for q in questions 
                       if getattr(q.status, 'value', str(q.status)) == 'answered']
        if answered_qs:
            lines.append("**Answered questions:**")
            for q in answered_qs[:5]:  # Limit to top 5
                lines.append(f"- {q.question_text[:100]}{'...' if len(q.question_text) > 100 else ''}")
            lines.append("")
        
        return "\n".join(lines)

    def _generate_question_section(self, questions: list) -> list:
        """
        Generate the "Research Questions Addressed" section for the report.
        
        Shows which questions were answered, partially answered, or remain open.
        This provides transparency about research coverage.
        
        Args:
            questions: List of ResearchQuestion objects from world model
            
        Returns:
            List of markdown lines for this section
        """
        if not questions:
            return []
        
        lines = [
            "## Research Questions Addressed",
            "",
            "The research objective was decomposed into the following questions. "
            "Status indicates how well each was addressed by the findings.",
            "",
        ]
        
        # Group by status
        answered = []
        partial = []
        unanswered = []
        
        for q in questions:
            status = q.status.value if hasattr(q.status, 'value') else str(q.status)
            priority = q.priority.value if hasattr(q.priority, 'value') else str(q.priority)
            
            item = {
                "question": q.question_text,
                "priority": priority,
                "confidence": q.confidence_score,
                "evidence_count": q.evidence_count,
                "answer": q.answer_summary,
            }
            
            if status == "answered":
                answered.append(item)
            elif status == "partial":
                partial.append(item)
            else:
                unanswered.append(item)
        
        # Summary stats
        total = len(questions)
        lines.append(f"**Coverage:** {len(answered)}/{total} fully answered, "
                     f"{len(partial)} partially answered, {len(unanswered)} open")
        lines.append("")
        
        # Answered questions
        if answered:
            lines.append("### ✅ Answered")
            lines.append("")
            for item in answered:
                priority_badge = "🔴" if item["priority"] == "high" else "🟡" if item["priority"] == "medium" else "🟢"
                lines.append(f"**{priority_badge} {item['question']}**")
                if item["answer"]:
                    lines.append(f"> {item['answer'][:300]}{'...' if len(item['answer'] or '') > 300 else ''}")
                lines.append(f"*Confidence: {item['confidence']:.0%} | Evidence: {item['evidence_count']} findings*")
                lines.append("")
        
        # Partially answered
        if partial:
            lines.append("### 🟡 Partially Answered")
            lines.append("")
            for item in partial:
                priority_badge = "🔴" if item["priority"] == "high" else "🟡" if item["priority"] == "medium" else "🟢"
                lines.append(f"**{priority_badge} {item['question']}**")
                if item["answer"]:
                    lines.append(f"> {item['answer'][:200]}...")
                lines.append(f"*Confidence: {item['confidence']:.0%} | Evidence: {item['evidence_count']} findings*")
                lines.append("")
        
        # Unanswered questions (gaps)
        if unanswered:
            lines.append("### ❌ Open Questions (Future Research)")
            lines.append("")
            for item in unanswered:
                priority_badge = "🔴" if item["priority"] == "high" else "🟡" if item["priority"] == "medium" else "🟢"
                lines.append(f"- {priority_badge} {item['question']}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines

    def _empty_report(self, objective: str) -> str:
        """Fallback report when no findings exist."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"# INQUIRO Discovery Report\n\n"
            f"**Objective:** {objective}\n\n"
            f"**Generated:** {timestamp}\n\n"
            f"---\n\n"
            f"No findings were recorded during this research run. "
            f"This may indicate issues with data access, API connectivity, "
            f"or an insufficient number of cycles.\n"
        )

    def _save_report(self, content: str, objective: str) -> str:
        """Save report to disk and return the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean objective for filename
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in objective[:40]
        ).strip("_")
        filename = f"report_{safe_name}_{timestamp}.md"
        report_path = self.output_dir / filename

        report_path.write_text(content, encoding="utf-8")
        logger.info(f"Report saved: {report_path}")
        return str(report_path)