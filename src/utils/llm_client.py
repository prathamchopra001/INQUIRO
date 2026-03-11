# -*- coding: utf-8 -*-
"""
Multi-provider LLM client for Inquiro.

Supports:
- OpenAI (GPT-4o, etc.)
- Anthropic (Claude)
- Ollama (local models)
- LM Studio (local, OpenAI-compatible)
- Google Gemini

All providers use the same interface: client.complete(prompt)
"""

import json
import time
import threading
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import importlib.util

from config.settings import settings
from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from src.utils.usage_tracker import get_usage_tracker

# Lazy import to avoid circular dependencies
_skill_manager = None
_domain_skill_injector = None

def _get_domain_skill_injector():
    """Lazily get the domain skill injector singleton."""
    global _domain_skill_injector
    if _domain_skill_injector is None:
        try:
            from src.skills.domain_skill_injector import get_domain_skill_injector
            _domain_skill_injector = get_domain_skill_injector()
        except ImportError as e:
            logging.getLogger(__name__).debug(f"Domain skills not available: {e}")
            _domain_skill_injector = False  # Mark as unavailable
    return _domain_skill_injector if _domain_skill_injector else None

def _get_skill_manager(llm_client=None):
    """Lazily initialize SkillManager to avoid circular imports.
    
    Args:
        llm_client: Optional LLMClient instance to use for skill generation
    """
    global _skill_manager
    if _skill_manager is None:
        try:
            from src.skills import SkillManager
            skills_cfg = settings.skills
            _skill_manager = SkillManager(
                llm_client=llm_client,
                skills_dir=skills_cfg.skills_dir,
                auto_generate=skills_cfg.auto_generate,
                enabled=skills_cfg.enabled,
            )
        except ImportError as e:
            logging.getLogger(__name__).warning(f"Skills system not available: {e}")
            _skill_manager = False  # Mark as unavailable
    elif _skill_manager and llm_client and _skill_manager.generator:
        # Update LLM client reference if provided and generator exists
        if _skill_manager.generator.llm is None:
            _skill_manager.generator.llm = llm_client
    return _skill_manager if _skill_manager else None

logger = logging.getLogger(__name__)

# Suppress noisy google_genai AFC logging
logging.getLogger("google_genai.models").setLevel(logging.WARNING)


def parse_llm_json(text: str, expect_array: bool = False, repair: bool = True) -> Any:
    """
    Parse JSON from LLM response with repair for common issues.
    
    Local models often produce malformed JSON:
    - Extra text before/after JSON
    - Single quotes instead of double
    - Trailing commas
    - Newlines inside strings
    
    Args:
        text: Raw LLM response text
        expect_array: If True, expect [] result; if False, expect {}
        repair: If True, attempt to repair malformed JSON
        
    Returns:
        Parsed JSON (list or dict), or None if parsing fails
    """
    import re
    
    if not text or not text.strip():
        return None
    
    # Step 1: Find JSON boundaries
    if expect_array:
        if '[' not in text or ']' not in text:
            return None
        start = text.find('[')
        end = text.rfind(']') + 1
    else:
        if '{' not in text or '}' not in text:
            return None
        start = text.find('{')
        end = text.rfind('}') + 1
    
    json_str = text[start:end]
    
    # Step 2: Try direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    if not repair:
        return None
    
    # Step 3: Repair common issues
    repaired = json_str
    
    # Fix single quotes -> double quotes (but not inside strings)
    repaired = repaired.replace("'", '"')
    
    # Fix trailing commas
    repaired = re.sub(r',\s*]', ']', repaired)
    repaired = re.sub(r',\s*}', '}', repaired)
    
    # Fix newlines inside strings
    repaired = re.sub(r'"\s*\n\s*', '" ', repaired)
    
    # Try parsing repaired
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # Step 4: Fallback - extract content via regex
    if expect_array:
        # Extract quoted strings from array
        matches = re.findall(r'"([^"]+)"', json_str)
        if matches:
            return [m.strip() for m in matches if len(m.strip()) > 2]
    
    return None


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    
    # Caching metrics (when available)
    cache_creation_input_tokens: int = 0   # Tokens written to cache
    cache_read_input_tokens: int = 0       # Tokens served from cache


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
        json_mode: bool = False,
        include_tools: bool = True
    ) -> LLMResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    def _is_package_installed(self, package_name: str) -> bool:
        return importlib.util.find_spec(package_name) is not None


# =============================================================================
# OPENAI CLIENT
# =============================================================================

class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""

    def __init__(self, model: str = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or settings.llm.model
        self.api_key = api_key or settings.llm.openai_api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def is_available(self) -> bool:
        return self.api_key is not None and self._is_package_installed("openai")

    def complete(self, prompt: str, system: Optional[str] = None, temperature: float = None,
                 max_tokens: int = None, json_mode: bool = False, include_tools: bool = True) -> LLMResponse:
        client = self._get_client()
        temperature = temperature if temperature is not None else settings.llm.temperature
        max_tokens = max_tokens or settings.llm.max_tokens
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {"model": self.model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider="openai",
            usage={"prompt_tokens": response.usage.prompt_tokens,
                   "completion_tokens": response.usage.completion_tokens,
                   "total_tokens": response.usage.total_tokens},
            raw_response=response
        )


# =============================================================================
# ANTHROPIC CLIENT
# =============================================================================

class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self, model: str = None, api_key: Optional[str] = None):
        self.model = model or settings.llm.model
        self.api_key = api_key or settings.llm.anthropic_api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._client

    def is_available(self) -> bool:
        return self.api_key is not None and self._is_package_installed("anthropic")

    def complete(self, prompt: str, system: Optional[str] = None, temperature: float = None,
                 max_tokens: int = None, json_mode: bool = False, include_tools: bool = True) -> LLMResponse:
        client = self._get_client()
        temperature = temperature if temperature is not None else settings.llm.temperature
        max_tokens = max_tokens or settings.llm.max_tokens
        kwargs = {"model": self.model, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": temperature}
        if system:
            kwargs["system"] = system
        if json_mode and system:
            kwargs["system"] = system + "\n\nYou must respond with valid JSON only."
        elif json_mode:
            kwargs["system"] = "You must respond with valid JSON only."

        # E2: Prompt caching — cache system prompt (25-35% cost reduction)
        # Anthropic caches content marked with cache_control for 5 minutes.
        # Cache reads cost 90% less than regular input tokens.
        if kwargs.get("system") and isinstance(kwargs["system"], str) and len(kwargs["system"]) > 100:
            kwargs["system"] = [
                {"type": "text", "text": kwargs["system"],
                 "cache_control": {"type": "ephemeral"}}
            ]

        response = client.messages.create(**kwargs)
        
        # Extract cache metrics from usage (Anthropic provides these)
        cache_creation = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
        cache_read = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
        
        if cache_read > 0:
            logger.debug(f"💾 Anthropic cache HIT: {cache_read} tokens from cache")
        elif cache_creation > 0:
            logger.debug(f"💾 Anthropic cache WRITE: {cache_creation} tokens cached")
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            provider="anthropic",
            usage={"prompt_tokens": response.usage.input_tokens,
                   "completion_tokens": response.usage.output_tokens,
                   "total_tokens": response.usage.input_tokens + response.usage.output_tokens},
            raw_response=response,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )


# =============================================================================
# OLLAMA CLIENT (Local)
# =============================================================================

class OllamaClient(BaseLLMClient):
    """Client for Ollama (local models) with timeout protection."""

    # Default timeout: 2 minutes (prevents 7+ minute hangs)
    DEFAULT_TIMEOUT = 120

    def __init__(self, model: str = None, base_url: str = None, timeout: int = None):
        self.model = model or settings.llm.model
        self.base_url = (base_url or settings.llm.ollama_base_url).rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                import httpx
                # Set explicit timeout on the HTTP client
                http_client = httpx.Client(timeout=httpx.Timeout(self.timeout))
                self._client = OpenAI(
                    api_key="ollama", 
                    base_url=f"{self.base_url}/v1",
                    http_client=http_client
                )
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def is_available(self) -> bool:
        if not self._is_package_installed("openai"):
            return False
        try:
            import urllib.request
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2)
            return True
        except:
            return False

    def complete(self, prompt: str, system: Optional[str] = None, temperature: float = None,
                 max_tokens: int = None, json_mode: bool = False, include_tools: bool = True,
                 timeout: int = None) -> LLMResponse:
        """Generate completion with timeout protection.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Request JSON output
            include_tools: Include tool definitions
            timeout: Override timeout in seconds (default: 120)
            
        Returns:
            LLMResponse object
            
        Raises:
            TimeoutError: If the request takes longer than timeout
        """
        # Recreate client with new timeout if specified
        if timeout and timeout != self.timeout:
            self.timeout = timeout
            self._client = None  # Force recreation with new timeout
        
        client = self._get_client()
        temperature = temperature if temperature is not None else settings.llm.temperature
        max_tokens = max_tokens or settings.llm.max_tokens
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {"model": self.model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            response = client.chat.completions.create(**kwargs)
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                provider="ollama",
                usage={"prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                       "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                       "total_tokens": getattr(response.usage, 'total_tokens', 0)} if response.usage else None,
                raw_response=response
            )
        except Exception as e:
            # Check if it's a timeout-related error
            error_str = str(e).lower()
            if 'timeout' in error_str or 'timed out' in error_str:
                logger.warning(f"Ollama timeout after {self.timeout}s for model {self.model}")
                raise TimeoutError(f"Ollama call timed out after {self.timeout} seconds") from e
            raise


# =============================================================================
# LM STUDIO CLIENT (Local)
# =============================================================================

class LMStudioClient(BaseLLMClient):
    """Client for LM Studio (local, OpenAI-compatible)."""

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or "local-model"
        self.base_url = (base_url or settings.llm.lm_studio_base_url).rstrip("/")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key="lm-studio", base_url=f"{self.base_url}/v1")
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def is_available(self) -> bool:
        if not self._is_package_installed("openai"):
            return False
        try:
            import urllib.request
            urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=2)
            return True
        except:
            return False

    def complete(self, prompt: str, system: Optional[str] = None, temperature: float = None,
                 max_tokens: int = None, json_mode: bool = False, include_tools: bool = True) -> LLMResponse:
        client = self._get_client()
        temperature = temperature if temperature is not None else settings.llm.temperature
        max_tokens = max_tokens or settings.llm.max_tokens
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {"model": self.model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider="lm_studio",
            usage={"prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                   "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                   "total_tokens": getattr(response.usage, 'total_tokens', 0)} if response.usage else None,
            raw_response=response
        )


# =============================================================================
# GOOGLE GEMINI CLIENT
# =============================================================================

class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""

    _global_lock = threading.Lock()
    _global_last_request_time: float = 0.0
    _min_interval: float = 60.0 / 10  # Conservative: 10 RPM = 6s between calls (safe for free + paid Tier 1)

    def __init__(self, model: str = None, api_key: Optional[str] = None):
        self.model = model or settings.llm.model
        self.api_key = api_key or settings.llm.google_api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError("google-genai package not installed. Run: pip install google-genai")
        return self._client

    def is_available(self) -> bool:
        return self.api_key is not None

    def _wait_for_rate_limit(self):
        with GeminiClient._global_lock:
            elapsed = time.time() - GeminiClient._global_last_request_time
            if elapsed < GeminiClient._min_interval:
                sleep_time = GeminiClient._min_interval - elapsed
                logger.debug(f"Gemini rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            GeminiClient._global_last_request_time = time.time()

    def complete(self, prompt: str, system: Optional[str] = None, temperature: float = None,
                 max_tokens: int = None, json_mode: bool = False, include_tools: bool = True, max_retries: int = 4) -> LLMResponse:
        from google.genai import types
        from google.genai.errors import ClientError

        client = self._get_client()
        temperature = temperature if temperature is not None else settings.llm.temperature
        max_tokens = max_tokens or settings.llm.max_tokens

        full_prompt = prompt
        if json_mode:
            full_prompt += "\n\nRespond with valid JSON only."

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system if system else None,
            tools=[] if not include_tools else None,
        )

        for attempt in range(1, max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = client.models.generate_content(
                    model=self.model, contents=full_prompt, config=config,
                )
                
                # Extract usage metadata if available
                usage = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    um = response.usage_metadata
                    usage = {
                        "prompt_tokens": getattr(um, 'prompt_token_count', 0) or 0,
                        "completion_tokens": getattr(um, 'candidates_token_count', 0) or 0,
                        "total_tokens": getattr(um, 'total_token_count', 0) or 0,
                    }
                    # Log usage for cost tracking
                    cached = getattr(um, 'cached_content_token_count', 0) or 0
                    if cached > 0:
                        logger.debug(f"💾 Gemini cache: {cached} tokens from cache")
                
                return LLMResponse(
                    content=response.text, model=self.model,
                    provider="gemini", usage=usage, raw_response=response,
                )
            except ClientError as e:
                is_rate_limit = e.status_code == 429 if hasattr(e, 'status_code') else '429' in str(e)
                if not is_rate_limit:
                    logger.error(f"Gemini API error (not retryable): {e}")
                    raise
                wait = 15
                try:
                    details = e.args[0] if e.args else {}
                    if isinstance(details, dict):
                        for d in details.get("error", {}).get("details", []):
                            if d.get("@type", "").endswith("RetryInfo"):
                                wait = int(d.get("retryDelay", "15s").replace("s", "")) + 2
                except Exception:
                    pass
                if attempt < max_retries:
                    logger.warning(f"Gemini 429 (attempt {attempt}/{max_retries}). Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error("Gemini rate limit: all retries exhausted.")
                    raise
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Gemini error (attempt {attempt}): {e}. Retrying...")
                    time.sleep(5)
                else:
                    raise
                
# =============================================================================
# GROQ CLIENT (OpenAI-compatible, very fast inference)
# =============================================================================

class GroqClient(BaseLLMClient):
    """Client for Groq API (OpenAI-compatible)."""

    def __init__(self, model: str = None, api_key: Optional[str] = None):
        self.model = model or "llama-3.3-70b-versatile"
        self.api_key = api_key or getattr(settings.llm, 'groq_api_key', None) or settings.llm.openai_api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def is_available(self) -> bool:
        return self.api_key is not None and self._is_package_installed("openai")

    def complete(self, prompt: str, system: Optional[str] = None, temperature: float = None,
                 max_tokens: int = None, json_mode: bool = False, include_tools: bool = True) -> LLMResponse:
        client = self._get_client()
        temperature = temperature if temperature is not None else settings.llm.temperature
        max_tokens = max_tokens or settings.llm.max_tokens
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {"model": self.model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider="groq",
            usage={"prompt_tokens": response.usage.prompt_tokens,
                   "completion_tokens": response.usage.completion_tokens,
                   "total_tokens": response.usage.total_tokens} if response.usage else None,
            raw_response=response
        )


# =============================================================================
# UNIFIED LLM CLIENT (Factory)
# =============================================================================

class LLMClient:
    """
    Unified LLM client that uses config for provider selection.

    Usage:
        client = LLMClient()
        response = client.complete("What is 2+2?")

        # E1: Route by complexity
        response = client.complete_for_task(prompt, task_type="simple")
    """

    PROVIDERS = {
        "openai":    OpenAIClient,
        "anthropic": AnthropicClient,
        "ollama":    OllamaClient,
        "lm_studio": LMStudioClient,
        "gemini":    GeminiClient,
        "groq":      GroqClient,
    }

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
        provider = provider or settings.llm.provider
        model    = model    or settings.llm.model

        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Choose from: {list(self.PROVIDERS.keys())}")

        client_class = self.PROVIDERS[provider]
        kwargs["model"] = model

        try:
            self._client = client_class(**kwargs)
            if not self._client.is_available():
                print(f"⚠️  Provider '{provider}' not available, trying auto-detect...")
                self._client = self._auto_detect_client()
        except Exception as e:
            print(f"⚠️  Error initializing provider '{provider}': {e}. Trying auto-detect...")
            self._client = self._auto_detect_client()

        self.provider = provider
        self.model    = model

        # D1: Circuit breaker — prevents cascade failures
        self._circuit_breaker = CircuitBreaker(
            name=provider,
            failure_threshold=3,
            recovery_timeout=60.0,
        )

        # E1: Model routing — complex vs simple tasks
        self._complex_model = model
        self._simple_model  = self._resolve_simple_model(provider, model)

    # =========================================================================
    # E1: Model routing
    # =========================================================================

    def _resolve_simple_model(self, provider: str, complex_model: str) -> str:
        """Return a faster/cheaper model for simple extraction tasks."""
        simple_models = {
            "anthropic": "claude-haiku-4-5-20251001",
            "openai":    "gpt-4o-mini",
            "gemini":    "gemini-2.0-flash",
            "ollama":    complex_model,
            "lm_studio": complex_model,
        }
        return simple_models.get(provider, complex_model)

    def complete_for_task(
        self,
        prompt: str,
        task_type: str = "complex",
        system: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> LLMResponse:
        """
        Route to appropriate model based on task complexity.

        DEPRECATED: Use complete_for_role() instead for thread-safe routing.
        This method is kept for backward compatibility but now delegates
        to complete_for_role() to avoid mutating shared client state.
        """
        role = "fallback" if task_type == "simple" else "orchestrator"
        return self.complete_for_role(
            prompt=prompt, role=role, system=system,
            temperature=temperature, max_tokens=max_tokens,
        )
            
    # =========================================================================
    # ROLE-BASED MODEL ROUTING
    # =========================================================================

    # Maps task roles to tiers: "strong", "fast", "code", or "local"
    ROLE_TIER_MAP = {
        # Strong tier — critical reasoning tasks
        "orchestrator":        "strong",
        "task_generation":     "strong",
        "completion_check":    "strong",
        "gap_analysis":        "strong",
        "finding_extraction":  "strong",
        "literature_extraction": "strong",  # Literature findings with attribution
        "report_writing":      "strong",
        "relationship_proposal": "strong",
        "objective_analysis":  "strong",
        "report_narrative":    "strong",
        "executive_summary":   "strong",
        "conclusion":          "strong",
        "schema_design":       "strong",  # Synthetic data: requires reasoning about dataset structure
        
        # Code tier — using fast tier (Groq) to avoid Gemini rate limits
        # Groq has Llama 3.3 70B which produces good code and is free
        "code_generation":     "fast",    # Groq: high quality, no rate limit issues
        "code_fix":            "fast",    # Groq: can understand errors properly
        "code_verification":   "fast",    # Groq: good enough for logic checks
        
        # Fast tier — structured extraction, formatting
        "query_generation":    "fast",
        "paper_ranking":       "fast",
        "scoring":             "fast",
        "dataset_search":      "fast",
        "query_classification": "fast",  # Classify user queries as simple/research
        
        # Local tier — cheap/fallback
        "fallback":            "local",
    }

    def _get_routed_client(self, tier: str) -> BaseLLMClient:
        """
        Get or create a client for the specified tier.
        Caches clients so we don't recreate them on every call.
        """
        cache_key = f"_routed_client_{tier}"
        
        if not hasattr(self, cache_key) or getattr(self, cache_key) is None:
            router_cfg = settings.router
            
            if tier == "strong":
                provider = router_cfg.strong_provider
                model = router_cfg.strong_model
            elif tier == "fast":
                provider = router_cfg.fast_provider
                model = router_cfg.fast_model
            elif tier == "code":
                provider = router_cfg.code_provider
                model = router_cfg.code_model
            else:  # local
                provider = router_cfg.local_provider or settings.llm.provider
                model = router_cfg.local_model or settings.llm.model
            
            if provider not in self.PROVIDERS:
                logger.warning(f"Unknown router provider '{provider}' for tier '{tier}' — falling back to default")
                setattr(self, cache_key, self._client)
                return self._client
            
            try:
                # Build kwargs based on provider type
                kwargs = {"model": model}
                if provider == "groq":
                    import os
                    kwargs["api_key"] = os.getenv("GROQ_API_KEY")
                
                client = self.PROVIDERS[provider](**kwargs)
                if client.is_available():
                    setattr(self, cache_key, client)
                    logger.info(f"Router: {tier} tier → {provider}/{model}")
                else:
                    logger.warning(f"Router: {tier} tier ({provider}) not available — using default")
                    setattr(self, cache_key, self._client)
            except Exception as e:
                logger.warning(f"Router: failed to create {tier} client ({provider}): {e} — using default")
                setattr(self, cache_key, self._client)
        
        return getattr(self, cache_key)

    def is_local_tier_for_role(self, role: str) -> bool:
        """
        Check if a given role uses a local model (Ollama/LM Studio).
        
        This helps callers decide whether to use simplified prompts
        that work better with local models.
        
        Args:
            role: Task role (e.g., "query_generation", "paper_ranking")
            
        Returns:
            True if the role routes to a local provider
        """
        if not settings.router.enabled:
            # Single provider mode - check main provider
            provider = settings.llm.provider.lower()
            return provider in ("ollama", "lm_studio", "local")
        
        # Get tier for this role
        tier = self.ROLE_TIER_MAP.get(role, "local")
        
        # Get provider for that tier
        router_cfg = settings.router
        if tier == "strong":
            provider = router_cfg.strong_provider
        elif tier == "fast":
            provider = router_cfg.fast_provider
        elif tier == "code":
            provider = router_cfg.code_provider
        else:  # local
            provider = router_cfg.local_provider or settings.llm.provider
        
        return provider.lower() in ("ollama", "lm_studio", "local")

    def complete_for_role(
        self,
        prompt: str,
        role: str = "fallback",
        system: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
        json_mode: bool = False,
        skip_skill_injection: bool = False,
    ) -> LLMResponse:
        """
        Route to the appropriate model based on the task role.

        This is the primary method callers should use instead of complete().
        When routing is disabled, falls back to the default provider.

        Args:
            prompt:      The prompt text
            role:        Task role (e.g., "orchestrator", "finding_extraction", "query_generation")
            system:      Optional system prompt
            temperature: Override temperature
            max_tokens:  Override max tokens
            json_mode:   Request JSON output format
            skip_skill_injection: If True, skip skill injection (used by SkillGenerator to avoid recursion)

        Returns:
            LLMResponse from the routed provider
        """
        # Option A: Disable tools for report-related roles to save tokens and prevent AFC 429 errors
        include_tools = role not in ["report_narrative", "executive_summary", "conclusion"]

        # =================================================================
        # SKILL INJECTION LAYER
        # =================================================================
        # Inject skill into system prompt if available (and not skipped)
        enhanced_system = system
        if not skip_skill_injection:
            # 1. Domain-specific skill injection (based on research objective)
            if settings.domain_skills.enabled:
                domain_injector = _get_domain_skill_injector()
                if domain_injector and domain_injector.current_skill:
                    # Inject domain expertise for relevant roles
                    domain_relevant_roles = {
                        "finding_extraction", "query_formulation", "query_generation",
                        "task_generation", "paper_ranking", "report_narrative",
                        "report_writing", "scoring", "orchestrator", "research_plan",
                        "literature_search", "data_analysis",
                    }
                    if role in domain_relevant_roles:
                        domain_skill = domain_injector.current_skill
                        if enhanced_system:
                            enhanced_system = f"{domain_skill}\n\n{enhanced_system}"
                        else:
                            enhanced_system = domain_skill
                        logger.debug(f"Domain skill injected for role '{role}' (domain={domain_injector.current_domain})")
            
            # 2. Role-specific skill injection (existing behavior)
            skill_manager = _get_skill_manager(llm_client=self)
            if skill_manager:
                # Try to get or generate skill for this role
                skill = skill_manager.get_skill(role, example_prompt=prompt[:500] if prompt else "")
                if skill:
                    skill_block = skill_manager.loader.format_skill_for_injection(skill)
                    if enhanced_system:
                        enhanced_system = f"{skill_block}\n\n{enhanced_system}"
                    else:
                        enhanced_system = skill_block
                    logger.debug(f"Skill injected for role '{role}' (~{skill.token_estimate} tokens)")

        if not settings.router.enabled:
            response = self.complete(
                prompt=prompt, system=enhanced_system,
                temperature=temperature, max_tokens=max_tokens,
                json_mode=json_mode,
                include_tools=include_tools,
            )
            get_usage_tracker().record(response)
            return response
        
        tier = self.ROLE_TIER_MAP.get(role, "strong")
        client = self._get_routed_client(tier)
        
        logger.debug(f"Router: role='{role}' → tier='{tier}' → {client.model}")
        
        try:
            with self._circuit_breaker:
                response = client.complete(
                    prompt=prompt, system=enhanced_system,
                    temperature=temperature, max_tokens=max_tokens,
                    json_mode=json_mode,
                    include_tools=include_tools,
                )
                get_usage_tracker().record(response)
                return response
        except Exception as e:
            # If routed provider fails, fall back to default
            if client is not self._client:
                logger.warning(f"Router: {tier} tier failed ({e}) — falling back to default")
                response = self.complete(
                    prompt=prompt, system=enhanced_system,
                    temperature=temperature, max_tokens=max_tokens,
                    json_mode=json_mode,
                    include_tools=include_tools,
                )
                get_usage_tracker().record(response)
                return response
            raise

    # =========================================================================
    # Auto-detection
    # =========================================================================

    def _auto_detect_client(self) -> BaseLLMClient:
        """Try providers in order of preference."""
        for client_cls, label in [
            (OllamaClient,    "Ollama (local)"),
            (LMStudioClient,  "LM Studio (local)"),
        ]:
            try:
                c = client_cls()
                if c.is_available():
                    print(f"🟢 Auto-detected: {label}")
                    return c
            except Exception:
                pass

        if settings.llm.anthropic_api_key:
            try:
                c = AnthropicClient()
                if c.is_available():
                    print("🟢 Auto-detected: Anthropic Claude")
                    return c
            except Exception:
                pass

        if settings.llm.openai_api_key:
            try:
                c = OpenAIClient()
                if c.is_available():
                    print("🟢 Auto-detected: OpenAI")
                    return c
            except Exception:
                pass

        if settings.llm.google_api_key:
            print("🟢 Auto-detected: Google Gemini")
            return GeminiClient(model="gemini-2.5-flash")

        raise RuntimeError(
            "No LLM provider available!\n"
            "Please configure in .env file:\n"
            "1. Set LLM_PROVIDER and start Ollama/LM Studio locally, or\n"
            "2. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY"
        )

    # =========================================================================
    # Core interface
    # =========================================================================

    def complete(self, prompt: str, system: Optional[str] = None, temperature: float = None,
                 max_tokens: int = None, json_mode: bool = False, include_tools: bool = True) -> LLMResponse:
        """Generate a completion using the configured provider."""
        try:
            with self._circuit_breaker:
                return self._client.complete(
                    prompt=prompt, system=system,
                    temperature=temperature, max_tokens=max_tokens,
                    json_mode=json_mode,
                    include_tools=include_tools
                )
        except CircuitOpenError as e:
            logger.warning(f"Circuit breaker blocked call: {e}")
            raise
        except Exception:
            raise

    def is_available(self) -> bool:
        return self._client.is_available()

    @classmethod
    def list_available_providers(cls) -> List[str]:
        available = []
        for name, client_class in cls.PROVIDERS.items():
            try:
                if client_class().is_available():
                    available.append(name)
            except:
                pass
        return available
