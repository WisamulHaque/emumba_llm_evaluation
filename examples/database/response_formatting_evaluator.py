"""
Response Formatting Evaluator (Database)

Validates that raw database query results are transformed into a user-friendly
format (tables, summaries, natural language) appropriate to the query type.
Uses LLM-as-judge to reason about format appropriateness, readability
and absence of raw result internals in the formatted response.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the formatted response presents the raw database query results
to the user in a clear, readable, and appropriate format given the user's query
and the nature of the returned data.

Following inputs will be provided:

Inputs:
- Natural Language Query: The original request submitted by the user
- Raw Results: The raw rows or documents returned by executing the database query
- Formatted Response: The final response shown to the user after formatting

Natural Language Query:
{query}

Raw Results:
{raw_results}

Formatted Response:
{formatted_response}

Based on the above-provided inputs, use the rules below to evaluate response formatting:

Evaluation Rules:
1. Format appropriateness; the format must match the nature of the data and
   the query intent; a multi-row result set should be presented as a structured
   table or labeled list; an aggregation or single-value result should be a
   direct inline answer; a comparison should use a table or side-by-side layout;
   using a format that does not suit the data or query intent is a failure
2. Data fidelity; the Formatted Response must accurately represent the Raw
   Results without distorting, rounding, or omitting values; any factual
   discrepancy between the Raw Results and the Formatted Response is a failure
3. Absence of raw internals; Raw Result dumps (JSON arrays, CSV rows,
   internal column names exposed without context, or unprocessed
   result sets) must not be shown directly to the user; they must be
   transformed into a human-readable format
4. Readability; the output must be clearly structured and interpretable by a
   non-technical user; unlabeled columns, wall-of-text output for tabular data,
   or missing units and context that leave the user unable to interpret the
   response are failures
5. Completeness; the Formatted Response must surface all relevant rows or
   fields from the Raw Results; silently dropping records or columns that are
   relevant to the query is a failure
6. No invented content; the Formatted Response must not add information absent
   from the Raw Results; padding, invented values, or commentary not grounded
   in the returned data is a failure

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
    raw_results: str,
    formatted_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the formatted response presents database query results
    clearly and appropriately for the user's query.

    Args:
        judge:              Configured LLMJudge instance
        query:              The original natural language user request
        raw_results:        The raw rows/documents returned by the database query
                            (as a string; JSON, CSV, or plain text)
        formatted_response: The final response shown to the user

    Returns a dict with:
        - passed: True if the formatting is appropriate and accurate, False otherwise
        - reason: LLM judge explanation
        - issues: List of identified formatting problems (empty list if passed)
    """
    if not formatted_response or not formatted_response.strip():
        return {
            "passed": False,
            "reason": "No formatted response was provided.",
            "issues": ["Formatted response is empty."],
        }
    if not raw_results or not raw_results.strip():
        return {
            "passed": False,
            "reason": "No raw results were provided.",
            "issues": ["Raw results are required to evaluate response formatting."],
        }

    prompt = EVAL_PROMPT.format(
        query=query,
        raw_results=raw_results,
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
        # Multi-row result correctly presented as a labeled table with all
        # values accurate and no Raw Result internals exposed (expected: PASS)
        "query_id": 1,
        "query": "List the names and salaries of all employees in the Engineering department, ordered by salary descending.",
        "raw_results": (
            '[{"name": "Alice Johnson", "salary": 95000.00}, '
            '{"name": "Bob Smith", "salary": 87000.00}, '
            '{"name": "Carol Lee", "salary": 82000.00}]'
        ),
        "formatted_response": (
            "Here are the Engineering department employees ranked by salary:\n\n"
            "| Name          | Salary      |\n"
            "|---------------|-------------|\n"
            "| Alice Johnson | $95,000.00  |\n"
            "| Bob Smith     | $87,000.00  |\n"
            "| Carol Lee     | $82,000.00  |"
        ),
    },
    {
        # Raw JSON dumped directly to the user with no transformation applied
        # (Rule 3: Raw Result internals exposed) (expected: FAIL)
        "query_id": 2,
        "query": "What is the total salary expenditure per department?",
        "raw_results": (
            '[{"dept_name": "Engineering", "total_salary": 264000.00}, '
            '{"dept_name": "Marketing", "total_salary": 198000.00}, '
            '{"dept_name": "HR", "total_salary": 142000.00}]'
        ),
        "formatted_response": (
            '[{"dept_name": "Engineering", "total_salary": 264000.00}, '
            '{"dept_name": "Marketing", "total_salary": 198000.00}, '
            '{"dept_name": "HR", "total_salary": 142000.00}]'
        ),
    },
    {
        # NoSQL aggregation result correctly presented as an inline summary
        # with accurate value and appropriate context (expected: PASS)
        "query_id": 3,
        "query": "How many Electronics products are currently in stock?",
        "raw_results": '[{"count": 7}]',
        "formatted_response": (
            "There are currently 7 Electronics products in stock."
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
            raw_results=sample["raw_results"],
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
