"""
Response Formatting Evaluator

Validates that the data retrieved from an API is presented to the user in a
clear, readable, and well-structured format appropriate to the data type and
the user's request. Uses LLM-as-judge to reason about format appropriateness,
readability, data fidelity, and absence of raw API internals in the response.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the formatted response presents the API data to the user in a
clear, readable, and appropriate format given the user's query and the data type.

Following inputs will be provided:

Inputs:
- Query: The original user query submitted to the application
- API Response: The raw data returned by the API before any formatting
- Formatted Response: The final response shown to the user after formatting

Query:
{query}

API Response:
{api_response}

Formatted Response:
{formatted_response}

Based on the above-provided inputs, use the rules below to evaluate response formatting:

Evaluation Rules:
1. The format must match the nature of the data and the user's query intent,
   a list of results should be presented as a structured list, a single value
   as a direct inline answer, a comparison as a table or side-by-side view;
   using a format that does not suit the data type or query intent is a failure
2. The formatted response must accurately represent the API response without
   distorting, rounding, or omitting values, any factual discrepancy between
   the API response and the formatted response is a failure
3. Raw API internals must not be exposed to the user, dumping raw JSON, HTTP
   status codes, field names as-is, or technical metadata directly in the
   response is a failure
4. The response must be readable and clearly structured, unlabeled data,
   wall-of-text output for structured data, or missing units and context that
   leave the user unable to interpret the response is a failure
5. The response must not add information absent from the API response, padding,
   invented values, or commentary not grounded in the returned data is a failure

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the response is well-formatted and appropriate, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<describe each identified formatting problem, or empty list if none>"]
}}"""


def evaluate_response_formatting(
    judge: LLMJudge,
    query: str,
    api_response: str,
    formatted_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the formatted response presents API data clearly and
    appropriately for the user's query.

    Args:
        judge:              Configured LLMJudge instance
        query:              The original user query
        api_response:       The raw data returned by the API (as a string or
                            stringified representation)
        formatted_response: The final response shown to the user

    Returns a dict with:
        - passed: True if the formatting is appropriate and accurate, False otherwise
        - reason: Summary of the verdict
        - issues: List of identified formatting problems (empty if passed)
    """
    prompt = EVAL_PROMPT.format(
        query=query,
        api_response=api_response,
        formatted_response=formatted_response,
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
    }


# Test data
SAMPLES = [
    {
        # List of results correctly formatted as a labeled structured list, values
        # match the API response exactly, no raw internals, clearly readable (expected: PASS)
        "query_id": 1,
        "query": "Show me available flights from Karachi to Dubai on March 13th, 2026.",
        "api_response": (
            '[{"flight": "EK601", "departure": "08:00", "arrival": "10:15", "price": "$210"}, '
            '{"flight": "PK221", "departure": "14:30", "arrival": "16:45", "price": "$175"}]'
        ),
        "formatted_response": (
            "Here are the available flights from Karachi to Dubai on March 13th, 2026:\n\n"
            "1. EK601 — Departs 08:00, Arrives 10:15 | Price: $210\n"
            "2. PK221 — Departs 14:30, Arrives 16:45 | Price: $175"
        ),
    },
        {
        # Raw JSON dumped as-is including technical metadata (timestamp) — no
        # human-readable transformation applied (Rule 3: raw API internals exposed)
        # (expected: FAIL)
        "query_id": 2,
        "query": "What is the current exchange rate from USD to EUR?",
        "api_response": '{"base": "USD", "target": "EUR", "rate": 0.9183, "timestamp": 1741651200}',
        "formatted_response": (
            '{"base": "USD", "target": "EUR", "rate": 0.9183, "timestamp": 1741651200}'
        ),
    },
    {
        # Response is properly formatted as prose but the rate is rounded from
        # 0.9183 to 0.92, distorting the actual value (Rule 2: data fidelity failure)
        # (expected: FAIL)
        "query_id": 3,
        "query": "What is the current exchange rate from USD to EUR?",
        "api_response": '{"base": "USD", "target": "EUR", "rate": 0.9183, "timestamp": 1741651200}',
        "formatted_response": "The current exchange rate from USD to EUR is 0.92.",
    },
    {
        # Comparison query answered as unstructured prose instead of a table or
        # labeled list (Rule 1: format does not suit a comparison query); response
        # also adds invented advice not present in the API data (Rule 5: invented
        # content) (expected: FAIL)
        "query_id": 4,
        "query": "Compare the prices of available hotels in Istanbul for 2 nights.",
        "api_response": (
            '[{"hotel": "Grand Bosphorus", "price_per_night": "$120", "rating": 4.5}, '
            '{"hotel": "Istanbul Central", "price_per_night": "$85", "rating": 4.1}, '
            '{"hotel": "Blue Mosque Inn", "price_per_night": "$65", "rating": 3.8}]'
        ),
        "formatted_response": (
            "There are three hotels available. The Grand Bosphorus costs $120 per night and has a "
            "4.5 rating. Istanbul Central is $85 per night with a 4.1 rating. Blue Mosque Inn is "
            "the cheapest at $65 per night but only has a 3.8 rating. We recommend the Grand "
            "Bosphorus for a premium experience given its superior location near the waterfront."
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
        result = evaluate_response_formatting(
            judge,
            query=sample["query"],
            api_response=sample["api_response"],
            formatted_response=sample["formatted_response"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query  : {sample['query']}")
        print(f"  Reason : {result['reason']}")
        if result["issues"]:
            print(f"  Issues:")
            for i, issue in enumerate(result["issues"], 1):
                print(f"    {i}. {issue}")
    print()


if __name__ == "__main__":
    main()
