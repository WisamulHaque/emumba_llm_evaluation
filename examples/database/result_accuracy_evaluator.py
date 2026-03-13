"""
Result Accuracy Evaluator

Validates that the query results returned to the user match the expected
ground truth output for the given natural language query. Evaluates value
correctness, aggregation correctness, filter correctness, sorting, and
completeness of the returned data. Uses LLM-as-judge to handle format
variance, rounding differences, and unordered result sets without requiring
a live database connection.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the actual query results correctly match the expected ground
truth results for the given natural language query.

Following inputs will be provided:

Inputs:
- Natural Language Query: The original request submitted by the user
- Ground Truth Results: The expected correct output for the query
- Actual Results: The data returned by the application after executing the query

Natural Language Query:
{query}

Ground Truth Results:
{ground_truth_results}

Actual Results:
{actual_results}

Based on the above-provided inputs, use the rules below to evaluate result accuracy:

Evaluation Rules:
1. Value correctness; every value in the Actual Results must match the
   corresponding value in the Ground Truth Results; incorrect, swapped, or
   fabricated values are a failure
2. Aggregation correctness; computed values such as SUM, COUNT, AVG, MAX, or
   MIN must match the Ground Truth Results; small floating-point rounding
   differences within two decimal places are acceptable
3. Filter correctness; the Actual Results must contain only the rows or
   documents that satisfy the query's filter conditions as demonstrated by the
   Ground Truth Results; extra rows that should have been filtered out, or
   missing rows that should have been included, are a failure
4. Sorting correctness; if the Ground Truth Results are ordered, the Actual
   Results must follow the same ordering; an unordered result set when the
   query expects ORDER BY is a failure
5. Completeness; the Actual Results must include all rows or documents present
   in the Ground Truth Results with no missing records; returning a partial
   subset when the full set is expected is a failure
6. Extra records; returning additional rows or documents beyond what the
   Ground Truth Results show is a failure unless the Natural Language Query
   implies no row limit
7. Empty results; if the Actual Results are empty but the Ground Truth Results
   are not, that is a failure; if both are empty, that is a pass

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the actual results are correct, false otherwise>,
    "reason": "<one or two sentences summarising the overall verdict>",
    "issues": ["<describe each discrepancy between actual and ground truth results, or empty list if none>"]
}}"""


def evaluate_result_accuracy(
    judge: LLMJudge,
    query: str,
    actual_results: str,
    ground_truth_results: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the actual query results match the expected ground truth.

    Args:
        judge:                 Configured LLMJudge instance
        query:                 The original natural language user request
        actual_results:        The data returned by the application (string —
                               JSON, table, or plain text)
        ground_truth_results:  The expected correct output in the same format

    Returns a dict with:
        - passed: True if the actual results are correct, False otherwise
        - reason: LLM judge explanation
        - issues: List of specific discrepancies (empty list if passed)
    """
    if not actual_results or not actual_results.strip():
        return {
            "passed": False,
            "reason": "No results were returned by the application.",
            "issues": ["Actual results are empty."],
        }
    if not ground_truth_results or not ground_truth_results.strip():
        return {
            "passed": False,
            "reason": "No ground truth results were provided.",
            "issues": ["Ground truth results are required to evaluate result accuracy."],
        }

    prompt = EVAL_PROMPT.format(
        query=query,
        ground_truth_results=ground_truth_results,
        actual_results=actual_results,
    )
    result = judge.judge(prompt)

    passed = result.get("passed", 0)
    if isinstance(passed, str):
        passed = int(passed) if passed.isdigit() else 0

    issues = result.get("issues", [])

    return {
        "passed": int(passed) == 1,
        "reason": result.get("reason", ""),
        "issues": issues if isinstance(issues, list) else [],
    }


# Test data
SAMPLES = [
    {
        # SQL — actual results exactly match the ground truth (expected: PASS)
        "query_id": 1,
        "query": "List the names and salaries of all employees in the Engineering department, ordered by salary descending.",
        "ground_truth_results": """\
name            | salary
----------------|----------
Alice Johnson   | 95000.00
Bob Smith       | 87000.00
Carol Lee       | 82000.00""",
        "actual_results": """\
name            | salary
----------------|----------
Alice Johnson   | 95000.00
Bob Smith       | 87000.00
Carol Lee       | 82000.00""",
    },
    {
        # SQL — aggregation result has a wrong total and is missing a row (expected: FAIL)
        "query_id": 2,
        "query": "Show the total salary expenditure per department.",
        "ground_truth_results": """\
dept_name       | total_salary
----------------|-------------
Engineering     | 264000.00
Marketing       | 198000.00
HR              | 142000.00""",
        "actual_results": """\
dept_name       | total_salary
----------------|-------------
Engineering     | 264000.00
Marketing       | 210000.00""",
    },
    {
        # NoSQL (MongoDB) — actual results include an out-of-stock product that should have been filtered out (expected: FAIL)
        "query_id": 3,
        "query": "Find all products in the 'Electronics' category that are currently in stock.",
        "ground_truth_results": """\
[
  { "_id": "p1", "name": "Wireless Headphones", "category": "Electronics", "price": 79.99, "in_stock": true },
  { "_id": "p3", "name": "Bluetooth Speaker",   "category": "Electronics", "price": 49.99, "in_stock": true }
]""",
        "actual_results": """\
[
  { "_id": "p1", "name": "Wireless Headphones", "category": "Electronics", "price": 79.99, "in_stock": true },
  { "_id": "p2", "name": "Smart TV",            "category": "Electronics", "price": 499.99, "in_stock": false },
  { "_id": "p3", "name": "Bluetooth Speaker",   "category": "Electronics", "price": 49.99, "in_stock": true }
]""",
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
        result = evaluate_result_accuracy(
            judge,
            query=sample["query"],
            actual_results=sample["actual_results"],
            ground_truth_results=sample["ground_truth_results"],
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
