"""
Factuality Evaluator

Checks whether every factual claim in an LLM-generated response is explicitly
supported by a known ground truth answer. Uses a two-step LLM-as-judge approach:
first extract atomic claims, then verify each claim against the ground truth.

NOTE: This evaluator checks claims against ground truth (known correct answer).
      For checking claims against retrieved context, use the faithfulness evaluator
      in examples/accuracy/grounded_accuracy/faithfulness_evaluator.py instead.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge

EVAL_PROMPT = """Purpose:
Extract all atomic factual claims from an AI-generated response and verify each one against the ground truth.

Following inputs will be provided:

Inputs:
- Ground Truth: The factually correct reference answer
- Generated Response: The AI-generated response to evaluate

Ground Truth:
{ground_truth}

Generated Response:
{generated_response}

Based on the above-provided inputs, use the rules below to extract and verify claims:

Evaluation Rules:
- Extract only statements from the Generated Response that assert factual information; ignore opinions, hedges, or stylistic commentary
- Break compound statements from the Generated Response into separate atomic claims, each self-contained and independently verifiable
- A claim is supported only if the Ground Truth explicitly contains the same information
- A claim is unsupported if the Ground Truth does not clearly state the same fact
- Do not use outside knowledge — only evaluate based on the Ground Truth above

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
  "results": [
    {{
      "claim": "claim text",
      "score": 1 or 0,
      "reason": "short explanation"
    }}
  ]
}}"""


def evaluate_factuality(judge: LLMJudge, query: str, generated_response: str, ground_truth: str) -> Dict[str, Any]:
    """
    Evaluate whether all claims in the generated response are supported by the ground truth.

    Returns a dict with:
        - passed: True if all claims are supported, False otherwise
        - score: Fraction of supported claims (0.0 to 1.0)
        - reason: Summary of supported vs total claims
        - total_claims: Number of claims extracted
        - unsupported_claims: List of claims not found in the ground truth
    """
    if not generated_response:
        return {"passed": False, "score": 0.0, "reason": "No generated response provided", "total_claims": 0, "unsupported_claims": []}
    if not ground_truth:
        return {"passed": False, "score": 0.0, "reason": "No ground truth provided", "total_claims": 0, "unsupported_claims": []}

    prompt = EVAL_PROMPT.format(ground_truth=ground_truth, generated_response=generated_response)
    result = judge.judge(prompt)
    results = result.get("results")
    if not isinstance(results, list):
        raise ValueError("Claim verification failed: 'results' must be a list.")

    if not results:
        return {"passed": True, "score": 1.0, "reason": "No factual claims detected in response", "total_claims": 0, "unsupported_claims": []}

    unsupported_claims = []
    supported_count = 0
    for item in results:
        if int(item.get("score", 0)) == 1:
            supported_count += 1
        else:
            unsupported_claims.append(item.get("claim", "Unknown claim"))

    total_claims = len(results)
    score = supported_count / total_claims
    return {
        "passed": len(unsupported_claims) == 0,
        "score": round(score, 2),
        "reason": f"{supported_count} out of {total_claims} claims are supported by the ground truth.",
        "total_claims": total_claims,
        "unsupported_claims": unsupported_claims,
    }


# Test data
SAMPLES = [
    {
        # All claims in the response are directly supported by the ground truth (expected: PASS)
        "query_id": 1,
        "query": "What are Python decorators and how are they used?",
        "generated_response": (
            "Python decorators use the @syntax to modify function behavior. "
            "They are commonly used for logging, authentication, and caching. "
            "Examples of built-in decorators are @staticmethod, @classmethod, and @property."
        ),
        "ground_truth": (
            "Python decorators modify function behavior using the @syntax. "
            "They are used for logging, authentication, and caching. "
            "Built-in decorators include @staticmethod, @classmethod, and @property."
        ),
    },
    {
        # Response introduces a write overhead claim absent from the ground truth (expected: FAIL)
        "query_id": 2,
        "query": "How do database indexes improve query performance?",
        "generated_response": (
            "Database indexes speed up data retrieval by avoiding full table scans. "
            "B-tree indexes are commonly used. Indexes also add write overhead because "
            "they must be updated on every insert, update, or delete."
        ),
        "ground_truth": (
            "Database indexes speed up data retrieval by avoiding full table scans. "
            "B-tree indexes are the most common type used for range queries and exact matches."
        ),
    },
    {
        # Response claims async/await helps with multi-threading, which is absent from the ground truth (expected: FAIL)
        "query_id": 3,
        "query": "What are the benefits of async/await in Python?",
        "generated_response": (
            "Async/await improves code readability and I/O performance. "
            "It also enables true multi-threading, allowing CPU-bound tasks to run in parallel."
        ),
        "ground_truth": (
            "Async/await allows writing asynchronous code that is more readable than callbacks. "
            "It improves I/O performance by not blocking the main thread during I/O operations."
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
        result = evaluate_factuality(
            judge,
            query=sample["query"],
            generated_response=sample["generated_response"],
            ground_truth=sample["ground_truth"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query             : {sample['query']}")
        print(f"  Response          : {sample['generated_response']}")
        print(f"  Score             : {result['score']}")
        print(f"  Reason            : {result['reason']}")
        if result["unsupported_claims"]:
            for claim in result["unsupported_claims"]:
                print(f"  Unsupported claim : {claim}")
    print()


if __name__ == "__main__":
    main()
