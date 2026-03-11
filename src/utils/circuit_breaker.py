# -*- coding: utf-8 -*-
"""
Circuit Breaker for LLM API calls.

3-state pattern that prevents cascade failures:

    CLOSED  → normal operation, requests pass through
       ↓ (failure_threshold failures in window)
    OPEN    → requests blocked immediately, no API calls made
       ↓ (recovery_timeout seconds pass)
    HALF_OPEN → one test request allowed through
       ↓ success              ↓ failure
    CLOSED                  OPEN

Why this matters: without a circuit breaker, a rate-limited or
unavailable API causes every agent thread to pile up retrying —
wasting time and exhausting the retry budget. The breaker stops
the bleeding immediately and waits for the API to recover.
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED    = "closed"      # Normal — requests flow through
    OPEN      = "open"        # Tripped — requests blocked
    HALF_OPEN = "half_open"   # Testing — one request allowed


@dataclass
class CircuitStats:
    """Runtime statistics for a circuit breaker."""
    total_calls:     int = 0
    total_failures:  int = 0
    total_blocked:   int = 0
    total_successes: int = 0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0


class CircuitBreaker:
    """
    Thread-safe 3-state circuit breaker.

    Usage:
        breaker = CircuitBreaker(name="gemini", failure_threshold=3)

        try:
            with breaker:
                response = llm_client.complete(prompt)
        except CircuitOpenError:
            # API is down — skip or use fallback
            logger.warning("Circuit open, skipping LLM call")
        except Exception as e:
            # Normal error — circuit breaker records the failure
            raise
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        success_threshold: int = 1,
    ):
        """
        Args:
            name:               Identifier for logging
            failure_threshold:  Failures before opening circuit
            recovery_timeout:   Seconds to wait before trying again
            success_threshold:  Successes in HALF_OPEN to close circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = threading.Lock()
        self._half_open_successes = 0

    # =========================================================================
    # CONTEXT MANAGER INTERFACE
    # =========================================================================

    def __enter__(self):
        self._before_call()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._on_success()
        elif exc_type is not CircuitOpenError:
            self._on_failure(exc_val)
        # Don't suppress exceptions
        return False

    # =========================================================================
    # STATE MACHINE
    # =========================================================================

    def _before_call(self):
        """Check if call should proceed. Raises CircuitOpenError if blocked."""
        with self._lock:
            self._stats.total_calls += 1

            if self._state == CircuitState.CLOSED:
                return  # Allow call

            elif self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                elapsed = time.time() - self._stats.last_failure_time
                if elapsed >= self.recovery_timeout:
                    logger.info(
                        f"Circuit '{self.name}': OPEN → HALF_OPEN "
                        f"(recovery timeout elapsed)"
                    )
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_successes = 0
                    return  # Allow the test call
                else:
                    self._stats.total_blocked += 1
                    wait_remaining = self.recovery_timeout - elapsed
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is OPEN. "
                        f"Retry in {wait_remaining:.0f}s."
                    )

            elif self._state == CircuitState.HALF_OPEN:
                return  # Allow the test call

    def _on_success(self):
        """Record a successful call."""
        with self._lock:
            self._stats.total_successes += 1
            self._stats.consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.success_threshold:
                    logger.info(
                        f"Circuit '{self.name}': HALF_OPEN → CLOSED "
                        f"(recovery confirmed)"
                    )
                    self._state = CircuitState.CLOSED
                    self._half_open_successes = 0

    def _on_failure(self, exc: Exception):
        """Record a failed call."""
        with self._lock:
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Test call failed — reopen the circuit
                logger.warning(
                    f"Circuit '{self.name}': HALF_OPEN → OPEN "
                    f"(test call failed: {exc})"
                )
                self._state = CircuitState.OPEN

            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    logger.error(
                        f"Circuit '{self.name}': CLOSED → OPEN "
                        f"({self._stats.consecutive_failures} consecutive failures)"
                    )
                    self._state = CircuitState.OPEN

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    def get_stats(self) -> dict:
        return {
            "name":                   self.name,
            "state":                  self._state.value,
            "total_calls":            self._stats.total_calls,
            "total_failures":         self._stats.total_failures,
            "total_blocked":          self._stats.total_blocked,
            "total_successes":        self._stats.total_successes,
            "consecutive_failures":   self._stats.consecutive_failures,
            "failure_threshold":      self.failure_threshold,
        }

    def reset(self):
        """Manually reset to CLOSED state (for testing)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._stats = CircuitStats()
            self._half_open_successes = 0
        logger.info(f"Circuit '{self.name}' manually reset to CLOSED")


class CircuitOpenError(Exception):
    """Raised when a call is blocked by an open circuit."""
    pass