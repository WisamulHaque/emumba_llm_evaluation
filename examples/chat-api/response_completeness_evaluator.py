"""
Response Completeness Evaluator

Validates that all relevant data returned from an API is surfaced to the user
and no critical information is dropped or omitted during the response generation
step. Uses LLM-as-judge to reason about which fields in the API response are
relevant to the query and whether all of them appear in the formatted response.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether all relevant data from the API response has been surfaced to
the user in the formatted response. Determine which fields in the API response
are relevant to the user's query and verify that none of them are silently
omitted.

Following inputs will be provided:

Inputs:
- Query: The original user query submitted to the application
- API Response: The complete raw data returned by the API
- Formatted Response: The final response shown to the user

Query:
{query}

API Response:
{api_response}

Formatted Response:
{formatted_response}

Based on the above-provided inputs, use the rules below to evaluate response completeness:

Evaluation Rules:
1. Identify which fields in the API response are relevant to the user's query,
   a field is relevant if its value would help the user understand or act on
   the answer; purely technical metadata such as internal IDs, raw timestamps,
   or system status codes with no user-facing meaning are not relevant
2. Every relevant field identified in Rule 1 must be present in the formatted
   response, silently dropping any relevant field is a failure
3. When the API response contains multiple items (e.g. a list of flights or
   hotels), all items must be represented in the formatted response; showing
   only a subset without indicating that more results exist is a failure
4. A relevant field may be paraphrased or reformatted (e.g. a timestamp
   converted to a readable date), what matters is that its information is
   conveyed, not that the exact field name or raw value appears verbatim
5. Omitting a field that is irrelevant to the query is acceptable and must
   not be flagged as a completeness failure

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if all relevant fields are present in the formatted response, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "missing_fields": ["<name or description of each relevant field omitted from the response, or empty list if none>"]
}}"""


def evaluate_response_completeness(
    judge: LLMJudge,
    query: str,
    api_response: str,
    formatted_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether all relevant API data is surfaced in the formatted response.

    Args:
        judge:              Configured LLMJudge instance
        query:              The original user query
        api_response:       The complete raw data returned by the API (as a
                            string or stringified representation)
        formatted_response: The final response shown to the user

    Returns a dict with:
        - passed:         True if no relevant fields are omitted, False otherwise
        - reason:         Summary of the verdict
        - missing_fields: List of relevant fields omitted from the response
                          (empty if passed)
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

    missing_fields = result.get("missing_fields", [])

    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
        "missing_fields": missing_fields if isinstance(missing_fields, list) else [],
    }


# Test data
SAMPLES = [
    {
        # All relevant fields (flight number, departure, arrival, price) are
        # present; internal booking_ref field correctly omitted as irrelevant
        # to the query (expected: PASS)
        "query_id": 1,
        "query": "Show me available flights from Karachi to Dubai on March 13th, 2026.",
        "api_response": (
            '[{"flight": "EK601", "departure": "08:00", "arrival": "10:15", "price": "$210", "booking_ref": "BK9921"}, '
            '{"flight": "PK221", "departure": "14:30", "arrival": "16:45", "price": "$175", "booking_ref": "BK9922"}]'
        ),
        "formatted_response": (
            "Here are the available flights from Karachi to Dubai on March 13th, 2026:\n\n"
            "1. EK601 — Departs 08:00, Arrives 10:15 | Price: $210\n"
            "2. PK221 — Departs 14:30, Arrives 16:45 | Price: $175"
        ),
    },
    {
        # Response only shows 1 of 3 hotels without indicating more exist —
        # the other two results are silently dropped (expected: FAIL)
        "query_id": 2,
        "query": "Show me available hotels in Istanbul for 2 nights starting July 20th, 2026.",
        "api_response": (
            '[{"hotel": "Grand Bosphorus", "price_per_night": "$120", "rating": 4.5, "availability": "Available"}, '
            '{"hotel": "Istanbul Central", "price_per_night": "$85", "rating": 4.1, "availability": "Available"}, '
            '{"hotel": "Blue Mosque Inn", "price_per_night": "$65", "rating": 3.8, "availability": "Available"}]'
        ),
        "formatted_response": (
            "Here is a hotel available in Istanbul for your dates:\n\n"
            "Grand Bosphorus — $120/night | Rating: 4.5 | Available"
        ),
    },
    {
        # Response omits the baggage_allowance and cancellation_policy fields
        # which are directly relevant to a booking decision query (expected: FAIL)
        "query_id": 3,
        "query": "What are the details for flight EK601 from Karachi to Dubai on March 13th, 2026?",
        "api_response": (
            '{"flight": "EK601", "departure": "KHI 08:00", "arrival": "DXB 10:15", '
            '"price": "$210", "baggage_allowance": "30kg", "cancellation_policy": "Non-refundable"}'
        ),
        "formatted_response": (
            "Flight EK601 departs Karachi at 08:00 and arrives in Dubai at 10:15. "
            "The ticket price is $210."
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
        result = evaluate_response_completeness(
            judge,
            query=sample["query"],
            api_response=sample["api_response"],
            formatted_response=sample["formatted_response"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query  : {sample['query']}")
        print(f"  Reason : {result['reason']}")
        if result["missing_fields"]:
            print(f"  Missing Fields:")
            for i, field in enumerate(result["missing_fields"], 1):
                print(f"    {i}. {field}")
    print()


if __name__ == "__main__":
    main()
