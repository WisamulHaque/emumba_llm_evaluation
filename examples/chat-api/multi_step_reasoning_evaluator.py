"""
Multi-Step Reasoning Evaluator

Validates that the application correctly handles queries that require multiple
sequential API calls.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether intermediate results from each API call were passed forward
correctly between steps and whether the final response accurately synthesizes
the outputs from all steps. The sequence of API calls has already been verified
as correct before reaching this evaluation.

Following inputs will be provided:

Inputs:
- Query: The original user query that required multiple API calls to resolve
- API Call Chain: The ordered list of API calls made, each showing the step
  number, the API called, the parameters used, and the response received
- Final Response: The final answer shown to the user after all steps completed

Query:
{query}

API Call Chain:
{chain_block}

Final Response:
{final_response}

Based on the above-provided inputs, use the rules below to evaluate multi-step reasoning:

Evaluation Rules:
1. Intermediate result passing; any value produced by a step that is needed
   as input by a later step must be the actual value from that step's response;
   using a hardcoded, fabricated, or stale value instead of the real output
   is a failure
2. No data invention between steps; a step must not use a parameter value
   that was neither in the original query nor in a prior step's response;
   inventing IDs, codes, or quantities mid-chain is a failure
3. Final response completeness; the final response must incorporate relevant
   information from all steps in the chain; silently ignoring the output of
   any step when its data was relevant to the user's query is a failure
4. Final response accuracy; any fact stated in the final response must be
   traceable to the query or one of the step responses; fabricating or
   contradicting the API outputs in the final answer is a failure
5. Step dependency order; if a later step's parameters depend on an earlier
   step's result, evaluate only whether the actual values match; do not re-
   verify whether the sequence order itself was correct, that was already
   checked before this evaluation

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if intermediate passing and final synthesis are both correct, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<description of each reasoning failure found, or empty list if none>"]
}}"""


def evaluate_multi_step_reasoning(
    judge: LLMJudge,
    query: str,
    expected_steps: List[str],
    api_call_chain: List[Dict[str, Any]],
    final_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the application correctly executed and synthesized a
    multi-step API call sequence.

    Args:
        judge:          Configured LLMJudge instance
        query:          The original user query
        expected_steps: Ordered list of API names that should have been called
                        (ground truth), e.g. ["flight_search", "baggage_policy"]
        api_call_chain: Actual execution trace as a list of step dicts, each with:
                          - step (int): 1-based step index
                          - api (str): name of the API called
                          - params (dict): parameters passed to the API
                          - response (dict): data returned by the API
        final_response: The final answer shown to the user

    Returns a dict with:
        - passed: True if all checks pass, False otherwise
        - reason: Summary of the verdict
        - issues: List of identified problems (empty if passed)
        - layer: "structural" if failed in Layer 1 (wrong sequence),
                 "semantic" if failed in Layer 2 (bad passing or synthesis),
                 "none" if fully passed
    """

    # --- Layer 1: Structural Validation (deterministic, no LLM) ---
    # Check that the actual sequence matches the expected sequence exactly.

    structural_issues = []
    actual_steps = [step["api"] for step in api_call_chain]

    if len(actual_steps) != len(expected_steps):
        structural_issues.append(
            f"Expected {len(expected_steps)} API call(s) "
            f"({', '.join(expected_steps)}) but got {len(actual_steps)} "
            f"({', '.join(actual_steps) if actual_steps else 'none'})."
        )
    else:
        for i, (actual, expected) in enumerate(zip(actual_steps, expected_steps), start=1):
            if actual != expected:
                structural_issues.append(
                    f"Step {i}: expected API '{expected}' but got '{actual}'."
                )

    if structural_issues:
        return {
            "passed": False,
            "reason": f"{len(structural_issues)} sequence issue(s) found in the API call chain.",
            "issues": structural_issues,
            "layer": "structural",
        }

    # --- Layer 2: Semantic Validation (LLM-as-judge) ---
    # Sequence is correct; check intermediate passing and final synthesis.

    chain_block = ""
    for step in api_call_chain:
        params_str = "\n".join(
            f"      {k}: {v!r}" for k, v in step.get("params", {}).items()
        )
        response_str = "\n".join(
            f"      {k}: {v!r}" for k, v in step.get("response", {}).items()
        )
        chain_block += (
            f"  Step {step['step']}: {step['api']}\n"
            f"    Parameters:\n{params_str}\n"
            f"    Response:\n{response_str}\n\n"
        )

    prompt = EVAL_PROMPT.format(
        query=query,
        chain_block=chain_block.strip(),
        final_response=final_response,
    )
    result = judge.judge(prompt)

    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"

    issues = result.get("issues", [])

    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
        "issues": issues if isinstance(issues, list) else [],
        "layer": "none" if passed else "semantic",
    }


# Test data
SAMPLES = [
    {
        # Query requires: search for a flight, then fetch baggage policy for
        # that airline. Both steps executed correctly, intermediate airline code
        # passed forward properly, final response covers both.
        # (expected: PASS)
        "query_id": 1,
        "query": (
            "Find me the cheapest flight from Lahore to Dubai on April 10th, 2026, "
            "and tell me the baggage allowance for that airline."
        ),
        "expected_steps": ["flight_search", "baggage_policy"],
        "api_call_chain": [
            {
                "step": 1,
                "api": "flight_search",
                "params": {
                    "origin": "LHE",
                    "destination": "DXB",
                    "date": "2026-04-10",
                },
                "response": {
                    "flight_number": "PK201",
                    "airline_code": "PK",
                    "price": "$180",
                    "departure": "08:30",
                    "arrival": "10:45",
                },
            },
            {
                "step": 2,
                "api": "baggage_policy",
                "params": {
                    "airline_code": "PK",
                },
                "response": {
                    "airline": "Pakistan International Airlines",
                    "carry_on": "7 kg",
                    "checked_baggage": "23 kg",
                    "excess_fee": "$30 per additional kg",
                },
            },
        ],
        "final_response": (
            "The cheapest flight from Lahore to Dubai on April 10th, 2026 is "
            "PK201 at $180, departing at 08:30 and arriving at 10:45.\n\n"
            "Baggage allowance for Pakistan International Airlines:\n"
            "- Carry-on: 7 kg\n"
            "- Checked baggage: 23 kg\n"
            "- Excess fee: $30 per additional kg"
        ),
    },
    {
        # Query requires: search flight, then fetch baggage policy. But only
        # the flight search was executed — the baggage step was skipped.
        # (expected: FAIL — structural, missing step)
        "query_id": 2,
        "query": (
            "Book the cheapest flight from Karachi to London on May 5th, 2026 "
            "and show me the baggage policy for that airline."
        ),
        "expected_steps": ["flight_search", "baggage_policy"],
        "api_call_chain": [
            {
                "step": 1,
                "api": "flight_search",
                "params": {
                    "origin": "KHI",
                    "destination": "LHR",
                    "date": "2026-05-05",
                },
                "response": {
                    "flight_number": "EK601",
                    "airline_code": "EK",
                    "price": "$620",
                    "departure": "03:20",
                    "arrival": "07:45",
                },
            },
        ],
        "final_response": (
            "The cheapest flight from Karachi to London on May 5th, 2026 is "
            "EK601 at $620, departing at 03:20 and arriving at 07:45."
        ),
    },
    {
        # Query requires: get hotel details, then check room availability for
        # those dates. The hotel_id from step 1 was NOT passed to step 2;
        # a hardcoded wrong ID was used instead.
        # (expected: FAIL — semantic, bad intermediate passing)
        "query_id": 3,
        "query": (
            "Check if Hotel Serena in Islamabad has rooms available "
            "from June 12th to June 15th, 2026."
        ),
        "expected_steps": ["hotel_search", "room_availability"],
        "api_call_chain": [
            {
                "step": 1,
                "api": "hotel_search",
                "params": {
                    "name": "Hotel Serena",
                    "city": "Islamabad",
                },
                "response": {
                    "hotel_id": "HTL_ISB_0042",
                    "name": "Hotel Serena Islamabad",
                    "rating": 5,
                    "address": "Khayaban-e-Suhrawardy, Islamabad",
                },
            },
            {
                "step": 2,
                "api": "room_availability",
                "params": {
                    # Wrong: used a fabricated hotel_id instead of HTL_ISB_0042
                    "hotel_id": "HTL_ISB_0099",
                    "check_in": "2026-06-12",
                    "check_out": "2026-06-15",
                },
                "response": {
                    "available": True,
                    "rooms": [
                        {"type": "Deluxe", "price": "$210/night"},
                        {"type": "Suite", "price": "$380/night"},
                    ],
                },
            },
        ],
        "final_response": (
            "Hotel Serena Islamabad has rooms available from June 12th to 15th, 2026:\n"
            "- Deluxe: $210/night\n"
            "- Suite: $380/night"
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

    for sample in SAMPLES:
        result = evaluate_multi_step_reasoning(
            judge,
            query=sample["query"],
            expected_steps=sample["expected_steps"],
            api_call_chain=sample["api_call_chain"],
            final_response=sample["final_response"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}  (layer: {result['layer']})")
        print(f"  Query  : {sample['query']}")
        print(f"  Reason : {result['reason']}")
        if result["issues"]:
            print(f"  Issues:")
            for i, issue in enumerate(result["issues"], 1):
                print(f"    {i}. {issue}")
    print()


if __name__ == "__main__":
    main()
