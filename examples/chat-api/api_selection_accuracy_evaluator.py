"""
API Selection Accuracy Evaluator

Validates that the correct API endpoint is selected by the LLM based on the
user's query, given a catalog of available APIs. Uses LLM-as-judge to reason
about whether the selection is the most appropriate match for the query intent.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the selected API endpoint is the most appropriate choice to
resolve the user's query, given the catalog of available APIs.

Following inputs will be provided:

Inputs:
- Query: The original user query submitted to the application
- Available APIs: A catalog of API endpoints, each with a name and description
- Selected API: The API endpoint chosen by the application to handle the query

Query:
{query}

Available APIs:
{available_apis_block}

Selected API:
{selected_api}

Based on the above-provided inputs, use the rules below to evaluate API selection accuracy:

Evaluation Rules:
1. If the query cannot reasonably be served by any Available APIs in the catalog,
   the correct behavior is to make no selection, forcing a selection onto an
   unrelated API just to produce output is a failure
2. The Selected API must be the most semantically appropriate match for the query
   intent, a technically valid but clearly suboptimal choice when a more specific
   API exists in the catalog is a failure
3. If two or more APIs in the catalog are equally valid for the query, selecting
   any one of them is acceptable and should pass
4. The evaluation must be based solely on the query and the provided catalog,
   do not use outside knowledge about what the API might do beyond its description

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the selected API is the most appropriate choice, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<describe each identified problem with the selection, or empty list if none>"]
}}"""


def evaluate_api_selection_accuracy(
    judge: LLMJudge,
    query: str,
    available_apis: List[Dict[str, str]],
    selected_api: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the correct API endpoint was selected for the user's query.

    Args:
        judge:          Configured LLMJudge instance
        query:          The original user query
        available_apis: List of dicts with 'name' and 'description' keys
        selected_api:   The name of the API endpoint selected by the application

    Returns a dict with:
        - passed: True if the selection is appropriate, False otherwise
        - reason: Summary of the verdict
        - issues: List of identified problems (empty if passed)
    """
    catalog_names = {api["name"] for api in available_apis}
    if selected_api not in catalog_names:
        return {
            "passed": False,
            "reason": f"'{selected_api}' does not exist in the provided API catalog.",
            "issues": [f"Selected API '{selected_api}' was not found in the available APIs list."],
        }
    
    available_apis_block = "\n".join(
        f"  - [{api['name']}]: {api['description']}"
        for api in available_apis
    )

    prompt = EVAL_PROMPT.format(
        query=query,
        available_apis_block=available_apis_block,
        selected_api=selected_api,
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
        # Correct and unambiguous selection from a clear catalog (expected: PASS)
        "query_id": 1,
        "query": "What is the current weather in Karachi?",
        "available_apis": [
            {"name": "weather_api",        "description": "Returns current weather conditions and forecasts for a given city or coordinates"},
            {"name": "flight_search_api",  "description": "Searches for available flights between two airports on a specified date"},
            {"name": "hotel_search_api",   "description": "Lists available hotels in a city for given check-in and check-out dates"},
            {"name": "currency_rates_api", "description": "Returns current foreign exchange rates between two currencies"},
        ],
        "selected_api": "weather_api",
    },
    {
        # Generic search API selected when a specific flights API exists in the catalog (expected: FAIL)
        "query_id": 2,
        "query": "Find me available flights from London to New York next Monday.",
        "available_apis": [
            {"name": "flight_search_api",  "description": "Searches for available flights between two airports on a specified date"},
            {"name": "generic_search_api", "description": "Performs a broad web search across multiple domains and data sources"},
            {"name": "hotel_search_api",   "description": "Lists available hotels in a city for given check-in and check-out dates"},
        ],
        "selected_api": "generic_search_api",
    },
    {
        # Selected API does not exist in the provided catalog (expected: FAIL)
        "query_id": 3,
        "query": "Convert 500 US dollars to Euros.",
        "available_apis": [
            {"name": "weather_api",       "description": "Returns current weather conditions and forecasts for a given city"},
            {"name": "flight_search_api", "description": "Searches for available flights between two airports on a specified date"},
            {"name": "hotel_search_api",  "description": "Lists available hotels in a city for given check-in and check-out dates"},
        ],
        "selected_api": "currency_rates_api",
    },
    {
        # Two APIs are equally valid; selecting either is acceptable (expected: PASS)
        "query_id": 4,
        "query": "Get me the latest news headlines.",
        "available_apis": [
            {"name": "news_feed_api",       "description": "Returns the latest top news headlines from major global publishers"},
            {"name": "news_search_api",     "description": "Searches for recent news articles by keyword, topic, or publication date"},
            {"name": "social_trending_api", "description": "Returns currently trending topics and hashtags on social media platforms"},
        ],
        "selected_api": "news_feed_api",
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
        result = evaluate_api_selection_accuracy(
            judge,
            query=sample["query"],
            available_apis=sample["available_apis"],
            selected_api=sample["selected_api"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query        : {sample['query']}")
        print(f"  Selected API : {sample['selected_api']}")
        print(f"  Reason       : {result['reason']}")
        if result["issues"]:
            print(f"  Issues:")
            for i, issue in enumerate(result["issues"], 1):
                print(f"    {i}. {issue}")
    print()


if __name__ == "__main__":
    main()
