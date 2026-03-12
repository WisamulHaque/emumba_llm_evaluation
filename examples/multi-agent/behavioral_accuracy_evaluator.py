"""
Behavioral Accuracy Evaluator (Journey Validation)

Validates that the pathway taken by the LLM through the agent graph is correct.
Given a user query, the sequence of agents actually invoked should follow the
expected logical flow. Uses LLM-as-judge to allow for valid alternative orderings
rather than requiring a rigid exact match.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge

EVAL_PROMPT = """Purpose:
Evaluate whether the sequence of agents invoked during a multi-agent execution
correctly resolves the user query according to the expected agent path.

Following inputs will be provided:

Inputs:
- Query: The original user query the agent system was asked to resolve
- Expected Path: The ordered list of agents that should have been invoked
- Actual Path: The ordered list of agents that were actually invoked

Query:
{query}

Expected Path:
{expected_path}

Actual Path:
{actual_path}

Based on the above-provided inputs, use the rules below to evaluate behavioral accuracy:

Evaluation Rules:
1. The Actual Path must invoke all agents from the Expected Path that are essential
   to resolving the Query — missing a critical agent is a failure
2. The Actual Path must not invoke agents that are entirely unrelated to the Query
   or that contradict the Expected Path
3. Minor reordering is acceptable only if the swapped agents are independent of each
   other and the logical outcome is equivalent
4. Inserting additional agents beyond the Expected Path is acceptable only if they
   serve a clear supporting role; unnecessary extra agents are a failure
5. If the Actual Path skips an expected agent, the reason must be that the agent was
   genuinely unnecessary for the Query — otherwise it is a failure

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the actual path correctly resolves the query, false otherwise>,
    "reason": "<one or two sentences explaining the verdict>",
    "deviations": ["<describe each deviation from the expected path, or empty list if none>"]
}}"""


def evaluate_behavioral_accuracy(
    judge: LLMJudge,
    query: str,
    expected_path: List[str],
    actual_path: List[str],
) -> Dict[str, Any]:
    expected_path_str = " → ".join(expected_path)
    actual_path_str = " → ".join(actual_path)

    prompt = EVAL_PROMPT.format(
        query=query,
        expected_path=expected_path_str,
        actual_path=actual_path_str,
    )
    result = judge.judge(prompt)
    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"
    deviations = result.get("deviations", [])
    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
        "deviations": deviations if isinstance(deviations, list) else [],
    }


# Test data
SAMPLES = [
    {
        # Actual path matches expected exactly (expected: PASS)
        "query_id": 1,
        "query": "Book a flight from Karachi to Dubai for two passengers next Friday.",
        "expected_path": ["intent_classifier", "flight_search", "seat_selector", "booking_agent", "confirmation_agent"],
        "actual_path":   ["intent_classifier", "flight_search", "seat_selector", "booking_agent", "confirmation_agent"],
    },
    {
        # Critical agent skipped — payment_agent missing before confirmation (expected: FAIL)
        "query_id": 2,
        "query": "Purchase the annual Pro plan and send me a receipt.",
        "expected_path": ["intent_classifier", "plan_selector", "payment_agent", "receipt_agent"],
        "actual_path":   ["intent_classifier", "plan_selector", "confirmation_agent"],
    },
    {
        # Valid minor reorder — availability_checker and price_fetcher are independent (expected: PASS)
        "query_id": 3,
        "query": "Find the cheapest available hotel in Istanbul for this weekend.",
        "expected_path": ["intent_classifier", "availability_checker", "price_fetcher", "ranking_agent", "response_formatter"],
        "actual_path":   ["intent_classifier", "price_fetcher", "availability_checker", "ranking_agent", "response_formatter"],
    },
    {
        # Irrelevant agent injected into the path (expected: FAIL)
        "query_id": 4,
        "query": "Summarize the latest support ticket submitted by user ID 4821.",
        "expected_path": ["intent_classifier", "ticket_fetcher", "summarizer"],
        "actual_path":   ["intent_classifier", "ticket_fetcher", "payment_agent", "summarizer"],
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
        result = evaluate_behavioral_accuracy(
            judge,
            query=sample["query"],
            expected_path=sample["expected_path"],
            actual_path=sample["actual_path"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query    : {sample['query']}")
        print(f"  Reason   : {result['reason']}")
        if result["deviations"]:
            for d in result["deviations"]:
                print(f"  Deviation: {d}")
    print()


if __name__ == "__main__":
    main()