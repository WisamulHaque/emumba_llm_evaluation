"""
Completeness Evaluator

Checks whether an LLM-generated response covers all key points present in the
ground truth answer for a given query. Uses LLM-as-judge to identify omitted points.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge

EVAL_PROMPT = """Purpose:
You are an expert evaluator assessing whether an AI-generated response fully covers
all key points present in the ground truth answer for the user's query.

Following inputs will be provided:

Inputs:
- Query: The user's original question defining what needs to be answered
- Ground Truth: The complete reference answer containing all key points the response must cover
- Generated Response: The AI-generated response to evaluate

Query: {query}

Ground Truth: {ground_truth}

Generated Response: {generated_response}

Based on the above-provided inputs, use the rules below to evaluate completeness:

Evaluation Rules:
- The Generated Response must cover every key point present in the Ground Truth that is relevant to the Query
- Any key point from the Ground Truth that is absent from the Generated Response is a missing point
- The Generated Response fails if it omits any information from the Ground Truth that the Query requires
- Additional correct information in the Generated Response beyond the Ground Truth does not affect this evaluation
- Minor paraphrasing is acceptable as long as all key points from the Ground Truth are present in the Generated Response

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": true or false,
    "reason": "<one or two sentences confirming full coverage or identifying what key points are missing>",
    "missing_points": ["<list any key points from the ground truth omitted by the response, or empty list if none>"]
}}"""


def evaluate_completeness(judge: LLMJudge, query: str, generated_response: str, ground_truth: str) -> Dict[str, Any]:
    """
    Evaluate whether the generated response covers all key points in the ground truth.

    Returns a dict with:
        - passed: True if response is complete, False otherwise
        - reason: LLM judge explanation
        - missing_points: List of key points from the ground truth omitted by the response
    """
    if not generated_response:
        return {"passed": False, "reason": "No generated response provided", "missing_points": []}
    if not ground_truth:
        return {"passed": False, "reason": "No ground truth provided", "missing_points": []}

    prompt = EVAL_PROMPT.format(
        query=query,
        ground_truth=ground_truth,
        generated_response=generated_response
    )
    result = judge.judge(prompt)
    if "passed" not in result or "reason" not in result:
        raise ValueError(f"Completeness evaluation returned invalid JSON: {result}")
    return {
        "passed": result["passed"],
        "reason": result["reason"],
        "missing_points": result.get("missing_points", [])
    }


# Test data
SAMPLES = [
    {
        # Response covers all key points from the ground truth (expected: PASS)
        "query_id": 1,
        "query": "What are Python decorators and how are they used?",
        "generated_response": (
            "Python decorators use the @syntax to modify function behavior. "
            "They are commonly applied for logging, authentication, and caching. "
            "Python also provides built-in decorators such as @staticmethod, @classmethod, and @property."
        ),
        "ground_truth": (
            "Python decorators modify function behavior using the @syntax. "
            "They are used for logging, authentication, and caching. "
            "Built-in decorators include @staticmethod, @classmethod, and @property."
        ),
    },
    {
        # Response omits write overhead — a key point present in the ground truth (expected: FAIL)
        "query_id": 2,
        "query": "How do database indexes affect performance?",
        "generated_response": (
            "Database indexes improve read performance by allowing the database to locate data "
            "without scanning the entire table. B-tree indexes are the most common type."
        ),
        "ground_truth": (
            "Database indexes speed up data retrieval by avoiding full table scans. "
            "B-tree indexes are the most common type. "
            "Indexes add write overhead because they must be updated on every insert, update, or delete."
        ),
    },
    {
        # Response only covers readability and omits the I/O efficiency benefit from the ground truth (expected: FAIL)
        "query_id": 3,
        "query": "What are the benefits of using async/await in Python?",
        "generated_response": (
            "Async/await makes asynchronous Python code easier to read and write compared to "
            "traditional callback-based patterns."
        ),
        "ground_truth": (
            "Async/await improves the readability of asynchronous code compared to callbacks. "
            "It also increases I/O efficiency by preventing the main thread from blocking during I/O operations."
        ),
    },
]


# Main
def main():
    from dotenv import load_dotenv
    load_dotenv()

    judge = LLMJudge(
        provider=os.environ.get("LLM_JUDGE_PROVIDER", "groq"),
        model=os.environ.get("LLM_JUDGE_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        api_key=os.environ["LLM_JUDGE_API_KEY"],
    )

    for sample in SAMPLES:
        result = evaluate_completeness(
            judge,
            query=sample["query"],
            generated_response=sample["generated_response"],
            ground_truth=sample["ground_truth"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query         : {sample['query']}")
        print(f"  Response      : {sample['generated_response']}")
        print(f"  Reason        : {result['reason']}")
        if result["missing_points"]:
            for point in result["missing_points"]:
                print(f"  Missing point : {point}")
    print()


if __name__ == "__main__":
    main()
