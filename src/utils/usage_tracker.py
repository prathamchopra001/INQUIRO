# -*- coding: utf-8 -*-
"""
LLM Usage Tracker for Inquiro.

Tracks token usage and caching efficiency across LLM calls.
Useful for cost optimization and monitoring.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class ProviderUsage:
    """Usage statistics for a single provider."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    call_count: int = 0


@dataclass
class UsageTracker:
    """
    Tracks LLM usage across multiple providers.
    
    Thread-safe for use in concurrent environments.
    
    Usage:
        tracker = UsageTracker()
        tracker.record(response)  # After each LLM call
        print(tracker.summary())  # At end of run
    """
    
    _providers: Dict[str, ProviderUsage] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)
    
    def record(self, response) -> None:
        """Record usage from an LLMResponse."""
        if not response:
            return
            
        provider = getattr(response, 'provider', 'unknown')
        
        with self._lock:
            if provider not in self._providers:
                self._providers[provider] = ProviderUsage()
            
            pu = self._providers[provider]
            pu.call_count += 1
            
            # Extract usage
            usage = getattr(response, 'usage', None)
            if usage and isinstance(usage, dict):
                pu.prompt_tokens += usage.get('prompt_tokens', 0) or 0
                pu.completion_tokens += usage.get('completion_tokens', 0) or 0
                pu.total_tokens += usage.get('total_tokens', 0) or 0
            
            # Extract cache metrics
            pu.cache_creation_tokens += getattr(response, 'cache_creation_input_tokens', 0) or 0
            pu.cache_read_tokens += getattr(response, 'cache_read_input_tokens', 0) or 0
    
    def get_provider_usage(self, provider: str) -> Optional[ProviderUsage]:
        """Get usage for a specific provider."""
        return self._providers.get(provider)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens across all providers."""
        return sum(pu.total_tokens for pu in self._providers.values())
    
    @property
    def total_cache_read(self) -> int:
        """Total cache read tokens across all providers."""
        return sum(pu.cache_read_tokens for pu in self._providers.values())
    
    @property
    def total_calls(self) -> int:
        """Total LLM calls across all providers."""
        return sum(pu.call_count for pu in self._providers.values())
    
    def cache_efficiency(self) -> float:
        """
        Calculate overall cache efficiency (0-1).
        
        Returns the fraction of input tokens served from cache.
        """
        total_input = sum(pu.prompt_tokens for pu in self._providers.values())
        total_cached = sum(pu.cache_read_tokens for pu in self._providers.values())
        
        if total_input == 0:
            return 0.0
        return total_cached / total_input
    
    def estimated_savings(self) -> Dict[str, float]:
        """
        Estimate cost savings from caching.
        
        Returns dict with:
        - cache_read_tokens: tokens served from cache
        - estimated_savings_pct: percentage cost reduction
        - uncached_equivalent: what it would cost without caching
        """
        total_cached = self.total_cache_read
        # Anthropic: cache reads cost 10% of regular (90% savings)
        # Gemini: cache reads cost 25% of regular (75% savings)
        # Use conservative 75% savings estimate
        savings_rate = 0.75
        
        # Calculate uncached equivalent
        total_prompt = sum(pu.prompt_tokens for pu in self._providers.values())
        uncached_equiv = total_prompt + (total_cached * (1 / (1 - savings_rate)))
        
        savings_pct = 0.0
        if uncached_equiv > 0:
            savings_pct = (uncached_equiv - total_prompt) / uncached_equiv * 100
        
        return {
            "cache_read_tokens": total_cached,
            "estimated_savings_pct": round(savings_pct, 1),
            "uncached_equivalent_tokens": int(uncached_equiv),
        }
    
    def summary(self) -> str:
        """Generate a human-readable usage summary."""
        lines = ["=" * 50, "LLM Usage Summary", "=" * 50]
        
        for provider, pu in sorted(self._providers.items()):
            lines.append(f"\n{provider.upper()}:")
            lines.append(f"  Calls: {pu.call_count}")
            lines.append(f"  Prompt tokens: {pu.prompt_tokens:,}")
            lines.append(f"  Completion tokens: {pu.completion_tokens:,}")
            lines.append(f"  Total tokens: {pu.total_tokens:,}")
            
            if pu.cache_read_tokens > 0 or pu.cache_creation_tokens > 0:
                lines.append(f"  Cache created: {pu.cache_creation_tokens:,} tokens")
                lines.append(f"  Cache reads: {pu.cache_read_tokens:,} tokens")
        
        # Overall summary
        lines.append(f"\n{'=' * 50}")
        lines.append("TOTALS:")
        lines.append(f"  Total calls: {self.total_calls}")
        lines.append(f"  Total tokens: {self.total_tokens:,}")
        
        savings = self.estimated_savings()
        if savings["cache_read_tokens"] > 0:
            lines.append(f"  Cache efficiency: {self.cache_efficiency():.1%}")
            lines.append(f"  Est. cost savings: ~{savings['estimated_savings_pct']:.0f}%")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Export usage data as a dictionary."""
        return {
            "providers": {
                provider: {
                    "calls": pu.call_count,
                    "prompt_tokens": pu.prompt_tokens,
                    "completion_tokens": pu.completion_tokens,
                    "total_tokens": pu.total_tokens,
                    "cache_creation_tokens": pu.cache_creation_tokens,
                    "cache_read_tokens": pu.cache_read_tokens,
                }
                for provider, pu in self._providers.items()
            },
            "totals": {
                "calls": self.total_calls,
                "tokens": self.total_tokens,
                "cache_read": self.total_cache_read,
                "cache_efficiency": round(self.cache_efficiency(), 3),
            },
            "savings": self.estimated_savings(),
        }
    
    def reset(self) -> None:
        """Reset all usage counters."""
        with self._lock:
            self._providers.clear()


# Global tracker instance
_global_tracker: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    """Get or create the global usage tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = UsageTracker()
    return _global_tracker


def reset_usage_tracker() -> None:
    """Reset the global usage tracker."""
    global _global_tracker
    if _global_tracker:
        _global_tracker.reset()
