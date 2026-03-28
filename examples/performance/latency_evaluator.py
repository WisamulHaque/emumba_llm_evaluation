"""
Latency Evaluator

Measures and validates the response time of an LLM application against
a configurable threshold. Wraps any callable (API call, agent pipeline,
RAG chain) and records wall-clock elapsed time.

No external dependencies beyond the Python standard library.
"""

import time
from typing import Dict, Any, Callable, Optional


def evaluate_latency(
    fn: Callable[[], Any],
    threshold_seconds: float = 3.0,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute *fn* and measure its wall-clock latency.

    Args:
        fn:                 A zero-argument callable whose latency will be measured.
                            Typically a lambda or partial wrapping your LLM call.
        threshold_seconds:  Maximum acceptable response time in seconds.
        label:              Optional human-readable label for the test case.

    Returns a dict with:
        - passed:            True if elapsed time <= threshold, False otherwise
        - reason:            Summary of the timing result
        - elapsed_seconds:   Actual measured time (rounded to 3 decimal places)
        - threshold_seconds: The threshold that was applied
        - label:             The label, if provided
        - result:            The return value of *fn* (so you can inspect the response)
    """
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        elapsed = round(time.perf_counter() - start, 3)
        return {
            "passed": False,
            "reason": f"Function raised {type(exc).__name__}: {exc} (after {elapsed}s)",
            "elapsed_seconds": elapsed,
            "threshold_seconds": threshold_seconds,
            "label": label,
            "result": None,
        }
    elapsed = round(time.perf_counter() - start, 3)

    passed = elapsed <= threshold_seconds
    if passed:
        reason = f"{elapsed}s <= {threshold_seconds}s threshold"
    else:
        reason = f"{elapsed}s exceeded {threshold_seconds}s threshold"

    return {
        "passed": passed,
        "reason": reason,
        "elapsed_seconds": elapsed,
        "threshold_seconds": threshold_seconds,
        "label": label,
        "result": result,
    }


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

SAMPLES = [
    {
        # Fast response — well within threshold (expected: PASS)
        "label": "fast_query",
        "sleep": 0.1,
        "threshold": 3.0,
    },
    {
        # Slow response — exceeds threshold (expected: FAIL)
        "label": "slow_query",
        "sleep": 4.0,
        "threshold": 3.0,
    },
    {
        # Borderline — just under threshold (expected: PASS)
        "label": "borderline_query",
        "sleep": 2.9,
        "threshold": 3.0,
    },
]


def main():
    import functools

    def simulate_llm_call(sleep_time: float) -> str:
        """Simulate an LLM call that takes *sleep_time* seconds."""
        time.sleep(sleep_time)
        return "Simulated LLM response"

    for sample in SAMPLES:
        fn = functools.partial(simulate_llm_call, sample["sleep"])
        result = evaluate_latency(
            fn=fn,
            threshold_seconds=sample["threshold"],
            label=sample["label"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['label']}] {status}")
        print(f"  Elapsed   : {result['elapsed_seconds']}s")
        print(f"  Threshold : {result['threshold_seconds']}s")
        print(f"  Reason    : {result['reason']}")
    print()


if __name__ == "__main__":
    main()
