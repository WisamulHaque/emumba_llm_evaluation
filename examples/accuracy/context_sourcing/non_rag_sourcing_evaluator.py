"""
Non-RAG Sourcing Evaluator

Beyond RAG, systems often fetch data from APIs, databases, or programmatic
tools.  The exact evaluation approach depends on your architecture, but three
common failure modes are:

  1. API / Endpoint Selection  — Did the system pick the right endpoint?
  2. Parameter Accuracy        — Did it pass the right query params / filters?
  3. Query Generation          — Did it produce a correct database query?

This file provides one evaluator for each area as a starting example.

  • API Selection and Parameter Accuracy use simple ground-truth comparison
    (no LLM needed).  For more complex systems — e.g. ambiguous routing
    logic, multiple valid endpoints, or dynamic schemas — you can substitute
    an LLM-as-judge approach instead.
  • Query Generation uses LLM-as-judge because SQL equivalence cannot be
    reliably determined with string matching alone.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Dict, Any, List
from examples.common.llm_judge import LLMJudge


# ====================================================================
# 1. API Selection  (simple ground-truth comparison — no LLM)
# ====================================================================

def evaluate_api_selection(
    selected_api: str,
    ground_truth_api: str,
) -> Dict[str, Any]:
    """
    Check whether the system selected the correct API endpoint.

    This is a simple string comparison.  For complex systems where multiple
    endpoints could be valid, replace this with an LLM-as-judge call.

    Args:
        selected_api:     The endpoint the system chose
        ground_truth_api: The known-correct endpoint

    Returns:
        passed, reason
    """
    passed = selected_api == ground_truth_api
    reason = (
        f"Correct endpoint selected: '{selected_api}'."
        if passed
        else f"Wrong endpoint selected: got '{selected_api}', expected '{ground_truth_api}'."
    )
    return {"passed": passed, "reason": reason}


# ====================================================================
# 2. Parameter Accuracy  (simple ground-truth comparison — no LLM)
# ====================================================================

def evaluate_parameter_accuracy(
    generated_params: Dict[str, Any],
    ground_truth_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check whether the generated API parameters match the ground truth.

    Compares every key-value pair.  Extra keys not in the ground truth are
    flagged as issues, and missing keys are flagged as issues.

    For complex cases (e.g. date resolution from "next Friday" to an actual
    date), replace this with an LLM-as-judge call.

    Args:
        generated_params:    The params the system produced
        ground_truth_params: The known-correct params

    Returns:
        passed, reason, issues (list of mismatches)
    """
    issues: List[str] = []

    # Keys in ground truth but missing from generated
    for key in ground_truth_params:
        if key not in generated_params:
            issues.append(f"Missing param '{key}': expected {ground_truth_params[key]!r}")

    # Keys in generated but not in ground truth (extra / invented params)
    for key in generated_params:
        if key not in ground_truth_params:
            issues.append(f"Extra param '{key}': {generated_params[key]!r} (not expected)")

    # Keys present in both but with different values
    for key in ground_truth_params:
        if key in generated_params and generated_params[key] != ground_truth_params[key]:
            issues.append(
                f"Wrong value for '{key}': got {generated_params[key]!r}, "
                f"expected {ground_truth_params[key]!r}"
            )

    passed = len(issues) == 0
    reason = (
        "All parameters match the ground truth."
        if passed
        else f"Found {len(issues)} parameter mismatch(es)."
    )
    return {"passed": passed, "reason": reason, "issues": issues}


# ====================================================================
# 3. Query Generation  (LLM-as-judge)
# ====================================================================

_QUERY_GEN_PROMPT = """Purpose:
Evaluate whether the generated database query correctly captures the user's
natural language request.

Inputs:
- User Query: {query}
- Generated Query:
{generated_query}

Evaluation Rules:
1. The Generated Query must be syntactically valid for its query language
2. It must retrieve exactly what the natural language query asks for — wrong
   columns, wrong table, wrong filters, or wrong aggregation is a failure
3. It must not silently drop required clauses (WHERE, ORDER BY, GROUP BY,
   LIMIT) that the natural language query implies
4. It must not contain destructive statements (DROP, DELETE, TRUNCATE, UPDATE)
   unless the user explicitly requested a destructive operation

Output Format:
Respond with ONLY a JSON object:
{{
    "passed": <true or false>,
    "reason": "<one or two sentences>",
    "issues": ["<problems found, or empty list>"]
}}"""


def evaluate_query_generation(
    judge: LLMJudge,
    query: str,
    generated_query: str,
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to evaluate whether a generated database query (SQL /
    NoSQL) correctly answers the user's natural language request.

    Args:
        judge:           Configured LLMJudge instance
        query:           The user's natural language request
        generated_query: The SQL/NoSQL query the system produced

    Returns:
        passed, reason, issues
    """
    if not generated_query or not generated_query.strip():
        return {
            "passed": False,
            "reason": "No query was generated.",
            "issues": ["Empty query."],
        }

    prompt = _QUERY_GEN_PROMPT.format(
        query=query,
        generated_query=generated_query,
    )
    result = judge.judge(prompt)

    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"

    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
        "issues": result.get("issues", []) if isinstance(result.get("issues"), list) else [],
    }


# ====================================================================
# Test data & main
# ====================================================================

SAMPLES_API_SELECTION = [
    {
        # Correct endpoint selected (expected: PASS)
        "query_id": "api-1",
        "query": "Find flights from Karachi to Dubai next Friday.",
        "selected_api": "flight_search",
        "ground_truth_api": "flight_search",
    },
    {
        # Wrong endpoint — weather instead of flights (expected: FAIL)
        "query_id": "api-2",
        "query": "Find flights from Karachi to Dubai next Friday.",
        "selected_api": "weather",
        "ground_truth_api": "flight_search",
    },
]

SAMPLES_PARAMETERS = [
    {
        # All params correct (expected: PASS)
        "query_id": "param-1",
        "query": "Economy flights KHI to DXB on March 28 for 2 passengers.",
        "generated_params": {
            "origin": "KHI", "destination": "DXB",
            "departure_date": "2026-03-28", "passengers": 2, "cabin_class": "economy",
        },
        "ground_truth_params": {
            "origin": "KHI", "destination": "DXB",
            "departure_date": "2026-03-28", "passengers": 2, "cabin_class": "economy",
        },
    },
    {
        # Wrong passenger count and invented cabin_class (expected: FAIL)
        "query_id": "param-2",
        "query": "Flights LHE to IST for 3 passengers on April 5.",
        "generated_params": {
            "origin": "LHE", "destination": "IST",
            "departure_date": "2026-04-05", "passengers": 1, "cabin_class": "business",
        },
        "ground_truth_params": {
            "origin": "LHE", "destination": "IST",
            "departure_date": "2026-04-05", "passengers": 3,
        },
    },
]

SAMPLES_QUERY_GEN = [
    {
        # Correct SQL with both filters (expected: PASS)
        "query_id": "sql-1",
        "query": "Show all pending orders from customers in Pakistan.",
        "generated_query": (
            "SELECT o.order_id, o.order_date, o.total_amount "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE c.country = 'Pakistan' AND o.status = 'pending';"
        ),
    },
    {
        # Missing status filter — returns all Pakistan orders, not just pending (expected: FAIL)
        "query_id": "sql-2",
        "query": "Show all pending orders from customers in Pakistan.",
        "generated_query": (
            "SELECT o.order_id, o.order_date, o.total_amount "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE c.country = 'Pakistan';"
        ),
    },
]


def main():
    from dotenv import load_dotenv
    load_dotenv()

    judge = LLMJudge(
        provider=os.environ.get("LLM_JUDGE_PROVIDER", "groq"),
        model=os.environ.get("LLM_JUDGE_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        api_key=os.environ["LLM_JUDGE_API_KEY"],
    )

    # --- 1. API Selection (simple comparison) ---
    print("=" * 60)
    print("1. API Selection  (ground-truth comparison)")
    print("=" * 60)
    for s in SAMPLES_API_SELECTION:
        r = evaluate_api_selection(s["selected_api"], s["ground_truth_api"])
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{s['query_id']}] {status}")
        print(f"  Query    : {s['query']}")
        print(f"  Selected : {s['selected_api']}")
        print(f"  Reason   : {r['reason']}")

    # --- 2. Parameter Accuracy (simple comparison) ---
    print("\n" + "=" * 60)
    print("2. Parameter Accuracy  (ground-truth comparison)")
    print("=" * 60)
    for s in SAMPLES_PARAMETERS:
        r = evaluate_parameter_accuracy(s["generated_params"], s["ground_truth_params"])
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{s['query_id']}] {status}")
        print(f"  Query  : {s['query']}")
        print(f"  Params : {s['generated_params']}")
        print(f"  Reason : {r['reason']}")
        for issue in r.get("issues", []):
            print(f"  Issue  : {issue}")

    # --- 3. Query Generation (LLM-as-judge) ---
    print("\n" + "=" * 60)
    print("3. Query Generation  (LLM-as-judge)")
    print("=" * 60)
    for s in SAMPLES_QUERY_GEN:
        r = evaluate_query_generation(judge, s["query"], s["generated_query"])
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{s['query_id']}] {status}")
        print(f"  Query  : {s['query']}")
        print(f"  Reason : {r['reason']}")
        for issue in r.get("issues", []):
            print(f"  Issue  : {issue}")

    print()


if __name__ == "__main__":
    main()
