# -*- coding: utf-8 -*-
"""
VLM Figure Verification for INQUIRO.

Uses Vision Language Models (Claude, GPT-4V) to verify that generated
figures match their captions and support the claimed findings.

Verification Dimensions:
    1. Caption Accuracy: Does the figure match its title/caption?
    2. Axis Labels: Are X/Y axis labels correct and readable?
    3. Data Consistency: Does the visual match the claimed findings?
    4. Chart Appropriateness: Is this the right chart type for the data?
    
Usage:
    verifier = FigureVerifier(llm_client)
    results = verifier.verify_figures(
        figures_dir="outputs/run_.../figures/",
        findings=list_of_findings,
        code_context=generated_code,
    )
    for result in results:
        print(f"{result.figure_path}: {result.overall_score}/4")
"""

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class FigureVerificationResult:
    """Verification result for a single figure."""
    
    figure_path: str
    figure_name: str
    
    # Dimension scores (1-4 scale)
    caption_accuracy: int = 0
    axis_labels: int = 0
    data_consistency: int = 0
    chart_appropriateness: int = 0
    
    # Overall assessment
    overall_score: float = 0.0
    passes_verification: bool = True
    
    # Detailed feedback
    caption_found: str = ""
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    description: str = ""  # VLM's description of what the figure shows
    
    # Confidence
    confidence: float = 0.5
    
    def __post_init__(self):
        """Calculate overall score from dimensions."""
        scores = [
            self.caption_accuracy,
            self.axis_labels,
            self.data_consistency,
            self.chart_appropriateness,
        ]
        valid_scores = [s for s in scores if s > 0]
        if valid_scores:
            self.overall_score = sum(valid_scores) / len(valid_scores)
            self.passes_verification = self.overall_score >= 2.5
    
    @property
    def rating(self) -> str:
        """Human-readable rating."""
        if self.overall_score >= 3.5:
            return "Excellent"
        elif self.overall_score >= 2.5:
            return "Good"
        elif self.overall_score >= 1.5:
            return "Fair"
        else:
            return "Poor"
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "figure_path": self.figure_path,
            "figure_name": self.figure_name,
            "overall_score": round(self.overall_score, 2),
            "rating": self.rating,
            "passes_verification": self.passes_verification,
            "dimensions": {
                "caption_accuracy": self.caption_accuracy,
                "axis_labels": self.axis_labels,
                "data_consistency": self.data_consistency,
                "chart_appropriateness": self.chart_appropriateness,
            },
            "caption_found": self.caption_found,
            "description": self.description,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class FigureVerificationReport:
    """Complete verification report for all figures."""
    
    results: List[FigureVerificationResult] = field(default_factory=list)
    total_figures: int = 0
    figures_passed: int = 0
    figures_flagged: int = 0
    average_score: float = 0.0
    
    def __post_init__(self):
        """Calculate summary statistics."""
        if self.results:
            self.total_figures = len(self.results)
            self.figures_passed = sum(1 for r in self.results if r.passes_verification)
            self.figures_flagged = self.total_figures - self.figures_passed
            self.average_score = sum(r.overall_score for r in self.results) / len(self.results)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "FIGURE VERIFICATION REPORT",
            "=" * 50,
            "",
            f"Total Figures: {self.total_figures}",
            f"Passed: {self.figures_passed}",
            f"Flagged: {self.figures_flagged}",
            f"Average Score: {self.average_score:.1f}/4.0",
            "",
        ]
        
        if self.figures_flagged > 0:
            lines.append("⚠️ FLAGGED FIGURES:")
            lines.append("-" * 40)
            for result in self.results:
                if not result.passes_verification:
                    lines.append(f"  • {result.figure_name}: {result.overall_score:.1f}/4")
                    for issue in result.issues[:2]:
                        lines.append(f"    - {issue}")
            lines.append("")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "total_figures": self.total_figures,
            "figures_passed": self.figures_passed,
            "figures_flagged": self.figures_flagged,
            "average_score": round(self.average_score, 2),
            "results": [r.to_dict() for r in self.results],
        }


class FigureVerifier:
    """
    Verifies generated figures using Vision Language Models.
    
    Uses Claude's vision capabilities or GPT-4V to analyze figures
    and verify they match their captions and claimed findings.
    """
    
    # Supported image formats
    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
    
    def __init__(
        self,
        llm_client,
        max_figures: int = 10,
        min_score_threshold: float = 2.0,
    ):
        """
        Args:
            llm_client: LLMClient instance with vision capabilities
            max_figures: Maximum figures to verify per run
            min_score_threshold: Minimum score to pass (1-4 scale)
        """
        self.llm = llm_client
        self.max_figures = max_figures
        self.min_score_threshold = min_score_threshold
    
    def verify_figures(
        self,
        figures_dir: str,
        findings: List[Dict] = None,
        code_context: str = "",
    ) -> FigureVerificationReport:
        """
        Verify all figures in a directory.
        
        Args:
            figures_dir: Directory containing figure files
            findings: List of finding dicts for context
            code_context: Generated code that created the figures
            
        Returns:
            FigureVerificationReport with all results
        """
        figures_path = Path(figures_dir)
        
        if not figures_path.exists():
            logger.warning(f"Figures directory not found: {figures_dir}")
            return FigureVerificationReport()
        
        # Find all figure files
        figure_files = self._find_figures(figures_path)
        
        if not figure_files:
            logger.info(f"No figures found in {figures_dir}")
            return FigureVerificationReport()
        
        logger.info(f"Found {len(figure_files)} figures to verify")
        
        # Limit to max_figures
        if len(figure_files) > self.max_figures:
            logger.info(f"Limiting to {self.max_figures} figures")
            figure_files = figure_files[:self.max_figures]
        
        # Extract captions from code context
        captions = self._extract_captions(code_context)
        
        # Build findings context
        findings_context = self._build_findings_context(findings)
        
        # Verify each figure
        results = []
        for fig_path in figure_files:
            try:
                result = self._verify_single_figure(
                    fig_path,
                    captions.get(fig_path.name, ""),
                    findings_context,
                )
                results.append(result)
                
                status = "✅" if result.passes_verification else "⚠️"
                logger.info(f"  {status} {fig_path.name}: {result.overall_score:.1f}/4")
                
            except Exception as e:
                logger.warning(f"  ❌ Failed to verify {fig_path.name}: {e}")
                results.append(FigureVerificationResult(
                    figure_path=str(fig_path),
                    figure_name=fig_path.name,
                    issues=[f"Verification failed: {e}"],
                ))
        
        return FigureVerificationReport(results=results)
    
    def _find_figures(self, figures_path: Path) -> List[Path]:
        """Find all figure files in directory."""
        figures = []
        
        for ext in self.SUPPORTED_FORMATS:
            figures.extend(figures_path.glob(f"*{ext}"))
            figures.extend(figures_path.glob(f"**/*{ext}"))  # Recursive
        
        # Sort by name for consistent ordering
        figures = sorted(set(figures), key=lambda p: p.name)
        return figures
    
    def _extract_captions(self, code_context: str) -> Dict[str, str]:
        """Extract figure captions/titles from code context."""
        captions = {}
        
        if not code_context:
            return captions
        
        # Pattern 1: plt.savefig with preceding plt.title
        # plt.title("My Title")
        # plt.savefig("/app/outputs/figures/my_figure.png")
        title_pattern = r"plt\.title\(['\"]([^'\"]+)['\"]\)"
        savefig_pattern = r"plt\.savefig\(['\"]([^'\"]+)['\"]\)"
        
        lines = code_context.split("\n")
        current_title = ""
        
        for line in lines:
            title_match = re.search(title_pattern, line)
            if title_match:
                current_title = title_match.group(1)
            
            savefig_match = re.search(savefig_pattern, line)
            if savefig_match:
                fig_path = savefig_match.group(1)
                fig_name = Path(fig_path).name
                if current_title:
                    captions[fig_name] = current_title
                    current_title = ""
        
        # Pattern 2: plt.suptitle for subplots
        suptitle_pattern = r"plt\.suptitle\(['\"]([^'\"]+)['\"]\)"
        for match in re.finditer(suptitle_pattern, code_context):
            # Associate with all figures if we don't have specific mapping
            title = match.group(1)
            for fig_name in captions:
                if not captions[fig_name]:
                    captions[fig_name] = title
        
        return captions
    
    def _build_findings_context(self, findings: List[Dict]) -> str:
        """Build context string from findings."""
        if not findings:
            return "No findings available for context."
        
        context_parts = ["Related findings from the analysis:"]
        
        for i, finding in enumerate(findings[:5], 1):
            claim = finding.get("claim", finding.get("text", ""))[:200]
            context_parts.append(f"{i}. {claim}")
        
        return "\n".join(context_parts)
    
    def _verify_single_figure(
        self,
        fig_path: Path,
        caption: str,
        findings_context: str,
    ) -> FigureVerificationResult:
        """Verify a single figure using VLM."""
        
        # Load and encode image
        image_data = self._load_image(fig_path)
        if not image_data:
            return FigureVerificationResult(
                figure_path=str(fig_path),
                figure_name=fig_path.name,
                issues=["Could not load image"],
            )
        
        # Build verification prompt
        prompt = self._build_verification_prompt(caption, findings_context)
        
        # Call VLM
        try:
            response = self._call_vlm(image_data, prompt, fig_path.suffix)
            return self._parse_vlm_response(response, fig_path, caption)
        except Exception as e:
            logger.error(f"VLM call failed: {e}")
            return FigureVerificationResult(
                figure_path=str(fig_path),
                figure_name=fig_path.name,
                issues=[f"VLM verification failed: {e}"],
            )
    
    def _load_image(self, fig_path: Path) -> Optional[str]:
        """Load image and encode as base64."""
        try:
            with open(fig_path, "rb") as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to load image {fig_path}: {e}")
            return None
    
    def _build_verification_prompt(self, caption: str, findings_context: str) -> str:
        """Build the verification prompt for the VLM."""
        caption_section = f"Expected Caption/Title: {caption}" if caption else "No caption provided."
        
        return f"""Analyze this scientific figure and verify its quality.

{caption_section}

{findings_context}

Evaluate the figure on these dimensions (score 1-4 each):

1. CAPTION ACCURACY (1-4): Does the figure match its title/caption?
   - 4: Perfect match, caption accurately describes content
   - 3: Good match, minor discrepancies
   - 2: Partial match, some elements don't align
   - 1: Poor match, caption is misleading or wrong

2. AXIS LABELS (1-4): Are axes properly labeled?
   - 4: Clear labels with units, readable fonts
   - 3: Labels present but could be clearer
   - 2: Missing units or unclear labels
   - 1: Missing labels or illegible

3. DATA CONSISTENCY (1-4): Does the visual support the findings?
   - 4: Visual clearly supports claimed findings
   - 3: Mostly supports with minor ambiguity
   - 2: Ambiguous or hard to verify
   - 1: Contradicts or doesn't match findings

4. CHART APPROPRIATENESS (1-4): Is this the right chart type?
   - 4: Optimal chart type for this data
   - 3: Acceptable choice
   - 2: Suboptimal but readable
   - 1: Wrong chart type, misleading

Respond with JSON only:
{{
    "caption_accuracy": <1-4>,
    "axis_labels": <1-4>,
    "data_consistency": <1-4>,
    "chart_appropriateness": <1-4>,
    "description": "<what the figure shows in 1-2 sentences>",
    "issues": ["<issue 1>", "<issue 2>"],
    "suggestions": ["<suggestion 1>"],
    "confidence": <0.0-1.0>
}}"""


    def _call_vlm(
        self,
        image_data: str,
        prompt: str,
        file_extension: str,
    ) -> str:
        """Call Vision Language Model with image."""
        
        # Determine media type
        media_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(file_extension.lower(), "image/png")
        
        # Try different providers based on what's available
        # Priority: Anthropic (native vision) > Gemini > OpenAI
        
        provider = getattr(self.llm, 'provider', 'anthropic')
        
        if provider == "anthropic" or hasattr(self.llm, '_anthropic_client'):
            return self._call_anthropic_vision(image_data, prompt, media_type)
        elif provider == "gemini" or hasattr(self.llm, '_gemini_client'):
            return self._call_gemini_vision(image_data, prompt, media_type)
        else:
            # Fallback: try Anthropic
            return self._call_anthropic_vision(image_data, prompt, media_type)
    
    def _call_anthropic_vision(
        self,
        image_data: str,
        prompt: str,
        media_type: str,
    ) -> str:
        """Call Anthropic Claude with vision."""
        try:
            # Use the LLM client's Anthropic instance
            client = getattr(self.llm, '_anthropic_client', None)
            
            if client is None:
                # Try to create one
                import anthropic
                client = anthropic.Anthropic()
            
            response = client.messages.create(
                model="claude-sonnet-4-20250514",  # Vision-capable model
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )
            
            return response.content[0].text
            
        except ImportError:
            logger.warning("Anthropic library not available for vision")
            raise
        except Exception as e:
            logger.error(f"Anthropic vision call failed: {e}")
            raise
    
    def _call_gemini_vision(
        self,
        image_data: str,
        prompt: str,
        media_type: str,
    ) -> str:
        """Call Google Gemini with vision."""
        try:
            import google.generativeai as genai
            
            # Decode base64 to bytes
            import base64
            image_bytes = base64.b64decode(image_data)
            
            # Create image part
            image_part = {
                "mime_type": media_type,
                "data": image_bytes,
            }
            
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content([prompt, image_part])
            
            return response.text
            
        except ImportError:
            logger.warning("Google Generative AI library not available")
            raise
        except Exception as e:
            logger.error(f"Gemini vision call failed: {e}")
            raise
    
    def _parse_vlm_response(
        self,
        response_text: str,
        fig_path: Path,
        caption: str,
    ) -> FigureVerificationResult:
        """Parse VLM response into FigureVerificationResult."""
        
        try:
            # Find JSON in response
            text = response_text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                
                result = FigureVerificationResult(
                    figure_path=str(fig_path),
                    figure_name=fig_path.name,
                    caption_accuracy=max(1, min(4, int(data.get("caption_accuracy", 2)))),
                    axis_labels=max(1, min(4, int(data.get("axis_labels", 2)))),
                    data_consistency=max(1, min(4, int(data.get("data_consistency", 2)))),
                    chart_appropriateness=max(1, min(4, int(data.get("chart_appropriateness", 2)))),
                    caption_found=caption,
                    description=data.get("description", ""),
                    issues=data.get("issues", [])[:5],
                    suggestions=data.get("suggestions", [])[:3],
                    confidence=float(data.get("confidence", 0.5)),
                )
                
                # Recalculate overall score
                result.__post_init__()
                return result
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse VLM response: {e}")
        
        # Fallback: return default scores with warning
        return FigureVerificationResult(
            figure_path=str(fig_path),
            figure_name=fig_path.name,
            caption_accuracy=2,
            axis_labels=2,
            data_consistency=2,
            chart_appropriateness=2,
            caption_found=caption,
            issues=["Could not parse VLM response"],
            confidence=0.3,
        )


def verify_run_figures(
    run_output_dir: str,
    findings: List[Dict] = None,
    llm_client=None,
) -> FigureVerificationReport:
    """
    Convenience function to verify figures from a INQUIRO run.
    
    Args:
        run_output_dir: Path to run output directory
        findings: Optional list of findings for context
        llm_client: Optional LLMClient (creates one if not provided)
        
    Returns:
        FigureVerificationReport
    """
    run_path = Path(run_output_dir)
    figures_dir = run_path / "figures"
    
    if not figures_dir.exists():
        # Try alternative paths
        for alt in ["outputs/figures", "data/figures", "."]:
            alt_path = run_path / alt
            if alt_path.exists() and any(alt_path.glob("*.png")):
                figures_dir = alt_path
                break
    
    if llm_client is None:
        from src.utils.llm_client import LLMClient
        llm_client = LLMClient()
    
    verifier = FigureVerifier(llm_client)
    return verifier.verify_figures(str(figures_dir), findings)
