"""
Latency Awareness Evaluation

Tracks response times to ensure the application meets acceptable performance
thresholds for the use case. Unlike quality evaluators, this is a pure
measurement and threshold check — no LLM judge, no API calls.

The application under test is responsible for recording timing data and
providing it in the input file. This evaluator reads those pre-recorded
timings and checks them against configured thresholds.

Two timing metrics:

    Total Latency (required)
        Full round-trip time from request sent to complete response received,
        in milliseconds. Must be present in every input sample.
        Passed when latency_ms <= threshold_ms.

    Time to First Token / TTFT (optional)
        Time from request sent to first token received, in milliseconds.
        Only evaluated when present in the input sample — absence means
        the application did not measure it and TTFT is skipped for that sample.
        Passed when ttft_ms <= ttft_threshold_ms.

Overall pass condition:
    ttft_ms absent  : passed = latency_passed
    ttft_ms present : passed = latency_passed AND ttft_passed

Threshold precedence:
    Per-query threshold_ms (if set) overrides global LATENCY_THRESHOLD_MS.
    Per-query ttft_threshold_ms (if set) overrides global LATENCY_TTFT_THRESHOLD_MS.
    If ttft_ms is present but no ttft_threshold_ms is resolvable, TTFT
    evaluation is skipped with a warning.

Usage:
    python run_eval.py latency
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Data model
@dataclass
class LatencySample:
    """
    Input sample for latency evaluation.

    Timing data is recorded by the application under test and provided
    in the input file — this evaluator does not make any LLM calls.

    Attributes:
        query_id:          Unique identifier for tracking in output
        query:             The query that was sent to the application
        generated_response: The response the application produced
        latency_ms:        Total round-trip time recorded by the application (ms)
        ttft_ms:           Time to first token recorded by the application (ms).
                           Optional — absent means the application did not
                           measure TTFT and it will be skipped for this sample.
        threshold_ms:      Optional per-query total latency threshold (ms).
                           Overrides global LATENCY_THRESHOLD_MS when set.
        ttft_threshold_ms: Optional per-query TTFT threshold (ms).
                           Overrides global LATENCY_TTFT_THRESHOLD_MS when set.
        metadata:          Optional additional metadata
    """
    query_id:           int
    query:              str
    generated_response: str
    latency_ms:         float
    ttft_ms:            Optional[float]   = None
    threshold_ms:       Optional[float]   = None
    ttft_threshold_ms:  Optional[float]   = None
    metadata:           Dict[str, Any]    = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatencySample":
        """Create sample from dictionary (for JSON input)."""
        if "latency_ms" not in data:
            raise ValueError(
                f"Sample query_id={data.get('query_id')} is missing required field 'latency_ms'. "
                f"The application must record and provide timing data in the input file."
            )
        return cls(
            query_id=data.get("query_id", 0),
            query=data["query"],
            generated_response=data["generated_response"],
            latency_ms=float(data["latency_ms"]),
            ttft_ms=float(data["ttft_ms"]) if data.get("ttft_ms") is not None else None,
            threshold_ms=float(data["threshold_ms"]) if data.get("threshold_ms") is not None else None,
            ttft_threshold_ms=float(data["ttft_threshold_ms"]) if data.get("ttft_threshold_ms") is not None else None,
            metadata=data.get("metadata", {}),
        )

# Latency Evaluator
class LatencyEvaluator:
    """
    Evaluator for application response latency.

    Reads pre-recorded timing data from input samples and checks them
    against configured thresholds. No LLM calls are made.

    TTFT evaluation is per-sample opt-in — a sample without ttft_ms
    simply skips TTFT evaluation regardless of global config.
    """

    def __init__(
        self,
        threshold_ms:       float,
        ttft_threshold_ms:  Optional[float] = None,
    ):
        """
        Initialize the latency evaluator.

        Args:
            threshold_ms:      Global total latency threshold in ms.
                               Overridden per-sample if threshold_ms is set on sample.
            ttft_threshold_ms: Global TTFT threshold in ms.
                               Only applied when a sample provides ttft_ms.
                               Overridden per-sample if ttft_threshold_ms is set on sample.
        """
        self.threshold_ms      = threshold_ms
        self.ttft_threshold_ms = ttft_threshold_ms

    def evaluate_latency(self, sample: LatencySample) -> Dict[str, Any]:
        """
        Evaluate pre-recorded latency timings against thresholds.

        Args:
            sample: LatencySample with pre-recorded timing data from the application

        Returns:
            Dictionary with:
                - generated_response: Passed through from input for traceability
                - latency_ms:         Total latency from input
                - threshold_ms:       Effective threshold used for total latency
                - latency_passed:     True if latency_ms <= threshold_ms
                - ttft_ms:            TTFT from input, or None if not provided
                - ttft_threshold_ms:  Effective TTFT threshold, or None if not evaluated
                - ttft_passed:        True/False if TTFT evaluated, else None
                - passed:             True if all evaluated checks passed
                - reason:             Human-readable explanation
        """
        # Resolve thresholds — per-sample overrides global
        effective_threshold = sample.threshold_ms if sample.threshold_ms is not None else self.threshold_ms

        # Evaluate total latency
        latency_passed = sample.latency_ms <= effective_threshold

        # Evaluate TTFT — only when sample provides ttft_ms
        ttft_passed           = None
        effective_ttft_threshold = None

        if sample.ttft_ms is not None:
            # Resolve TTFT threshold — per-sample overrides global
            effective_ttft_threshold = (
                sample.ttft_threshold_ms
                if sample.ttft_threshold_ms is not None
                else self.ttft_threshold_ms
            )
            if effective_ttft_threshold is not None:
                ttft_passed = sample.ttft_ms <= effective_ttft_threshold
            # If no threshold resolvable, ttft_passed stays None — skip silently

        # Overall pass — all evaluated checks must pass
        passed = latency_passed and (ttft_passed is None or ttft_passed)

        reason = self._build_reason(
            latency_ms=sample.latency_ms,
            threshold_ms=effective_threshold,
            latency_passed=latency_passed,
            ttft_ms=sample.ttft_ms,
            ttft_threshold_ms=effective_ttft_threshold,
            ttft_passed=ttft_passed,
        )

        return {
            "generated_response": sample.generated_response,
            "latency_ms":         sample.latency_ms,
            "threshold_ms":       effective_threshold,
            "latency_passed":     latency_passed,
            "ttft_ms":            sample.ttft_ms,
            "ttft_threshold_ms":  effective_ttft_threshold,
            "ttft_passed":        ttft_passed,
            "passed":             passed,
            "reason":             reason,
        }

    def _build_reason(
        self,
        latency_ms:          float,
        threshold_ms:        float,
        latency_passed:      bool,
        ttft_ms:             Optional[float],
        ttft_threshold_ms:   Optional[float],
        ttft_passed:         Optional[bool],
    ) -> str:
        """Build a human-readable reason string from timing results."""
        parts = []

        if latency_passed:
            parts.append(
                f"Total latency {latency_ms:.0f}ms is within the {threshold_ms:.0f}ms threshold."
            )
        else:
            parts.append(
                f"Total latency {latency_ms:.0f}ms exceeds the {threshold_ms:.0f}ms threshold "
                f"by {latency_ms - threshold_ms:.0f}ms."
            )

        if ttft_ms is not None and ttft_threshold_ms is not None and ttft_passed is not None:
            if ttft_passed:
                parts.append(
                    f"TTFT {ttft_ms:.0f}ms is within the {ttft_threshold_ms:.0f}ms threshold."
                )
            else:
                parts.append(
                    f"TTFT {ttft_ms:.0f}ms exceeds the {ttft_threshold_ms:.0f}ms threshold "
                    f"by {ttft_ms - ttft_threshold_ms:.0f}ms."
                )
        elif ttft_ms is not None and ttft_threshold_ms is None:
            parts.append(
                f"TTFT {ttft_ms:.0f}ms recorded but no threshold configured — skipped."
            )

        return " ".join(parts)