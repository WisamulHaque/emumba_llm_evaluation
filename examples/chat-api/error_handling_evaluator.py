"""
Error Handling Evaluator

Validates that the application responds gracefully when an API call fails.
Covers API errors (4xx/5xx), timeouts, rate limits, and malformed responses.
The user should receive a meaningful, context-appropriate message, never a raw
error dump, a silent failure, or fabricated data presented as real.
Uses LLM-as-judge to reason about whether the application response meets those
standards given what the API actually returned.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the application handled an API error gracefully. Given the raw
error signal returned by the API and the response shown to the user, determine
whether the application responded appropriately or exposed, fabricated, or
silently ignored the failure.

Following inputs will be provided:

Inputs:
- Query: The original user query submitted to the application
- API Error: The raw error or failure signal returned by the API
- Application Response: The response shown to the user after the API call failed

Query:
{query}

API Error:
{api_error}

Application Response:
{application_response}

Based on the above-provided inputs, use the rules below to evaluate error handling:

Evaluation Rules:
1. No raw technical details must appear in the application response; HTTP status
   codes (e.g. "500"), exception class names, stack traces, or internal error
   codes shown to the user are a failure
2. The application must not silently ignore the failure; a response that proceeds
   as if the API call succeeded (e.g. "Here are your results: ..." with no data)
   or that gives no indication anything went wrong is a failure
3. The application must not fabricate results; if the API failed, the response
   must not invent data and present it as genuine API output
4. The failure message must be contextually appropriate for the error type:
   transient errors (timeout, rate limit) should suggest retrying; not-found
   errors should indicate the requested resource was unavailable; server errors
   should acknowledge a service problem without exposing internals
5. Where possible the response should be actionable, telling the user what they
   can do next (retry, refine the query, try different dates, contact support);
   this is a soft guideline, absence of actionable advice alone is not a
   failure but should be noted in the reason

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the application handled the error gracefully, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<description of each violation found, or empty list if none>"]
}}"""


def evaluate_error_handling(
    judge: LLMJudge,
    query: str,
    api_error: str,
    application_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the application handled an API error gracefully.

    Args:
        judge:                Configured LLMJudge instance
        query:                The original user query
        api_error:            The raw error or failure signal returned by the API
                              (e.g. HTTP status + body, timeout message, rate-limit
                              response, as a string)
        application_response: The final response shown to the user after the failure

    Returns a dict with:
        - passed: True if the error was handled gracefully, False otherwise
        - reason: Summary of the verdict
        - issues: List of violations found (empty if passed)
    """
    prompt = EVAL_PROMPT.format(
        query=query,
        api_error=api_error,
        application_response=application_response,
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
        # API returns 404 (no flights found for those dates); application
        # responds with a user-friendly message and suggests alternatives
        # (expected: PASS)
        "query_id": 1,
        "query": "Show me flights from Karachi to Dubai on March 13th, 2026.",
        "api_error": (
            '{"status": 404, "error": "NOT_FOUND", '
            '"message": "No flights available for the requested route and date."}'
        ),
        "application_response": (
            "Sorry, no flights are available from Karachi to Dubai on March 13th, 2026. "
            "You may want to try a different date or check nearby departure airports."
        ),
    },
    {
        # API returns HTTP 500 with a stack trace; application echoes the raw
        # status code and exception name to the user (expected: FAIL)
        "query_id": 2,
        "query": "What hotels are available in Istanbul from July 20th to July 22nd, 2026?",
        "api_error": (
            '{"status": 500, "error": "InternalServerError", '
            '"trace": "NullPointerException at HotelService.java:42"}'
        ),
        "application_response": (
            "Error 500: InternalServerError — NullPointerException at HotelService.java:42. "
            "Please contact the administrator."
        ),
    },
    {
        # API returns HTTP 429 rate-limit response; application silently
        # fabricates hotel listings and presents them as real results
        # (expected: FAIL)
        "query_id": 3,
        "query": "Find me available hotels in Paris for next weekend.",
        "api_error": (
            '{"status": 429, "error": "RATE_LIMIT_EXCEEDED", '
            '"message": "Too many requests. Retry after 60 seconds."}'
        ),
        "application_response": (
            "Here are some hotels available in Paris for next weekend:\n\n"
            "1. Hotel Le Marais — $150/night | Rating: 4.6 | Available\n"
            "2. Paris Central Inn — $95/night | Rating: 4.2 | Available"
        ),
    },
    {
        # API call times out after the deadline; application correctly
        # acknowledges the temporary outage and advises the user to retry
        # (expected: PASS)
        "query_id": 4,
        "query": "What is the current exchange rate from USD to EUR?",
        "api_error": (
            '{"status": "timeout", "error": "REQUEST_TIMEOUT", '
            '"message": "The upstream service did not respond within 10 seconds."}'
        ),
        "application_response": (
            "The exchange rate service is temporarily unavailable. "
            "Please try again in a few moments."
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
        result = evaluate_error_handling(
            judge,
            query=sample["query"],
            api_error=sample["api_error"],
            application_response=sample["application_response"],
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
