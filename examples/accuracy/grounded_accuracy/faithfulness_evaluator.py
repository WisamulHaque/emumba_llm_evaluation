"""
Faithfulness Evaluator (Grounded Accuracy)

Checks whether every factual claim in an LLM-generated response is explicitly
supported by the **provided context** (retrieved documents, tool outputs, API
responses).  This is different from Factuality — factuality checks claims
against a ground-truth answer, while faithfulness checks claims against the
context the system actually had access to.

Use this to detect hallucinations: claims the model invented that aren't in
the source material it was given.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Dict, Any, List
from examples.common.llm_judge import LLMJudge

EVAL_PROMPT = """Purpose:
Extract all atomic factual claims from an AI-generated response and verify
each one against the provided context.  The context represents the source
material the system had access to when generating the response.

Following inputs will be provided:

Inputs:
- Context: The source material (retrieved documents, tool outputs, API
  responses) that was available to the system when it generated the response
- Generated Response: The AI-generated response to evaluate

Context:
{context}

Generated Response:
{generated_response}

Based on the above-provided inputs, use the rules below to extract and verify claims:

Evaluation Rules:
- Extract only statements from the Generated Response that assert factual
  information; ignore opinions, hedges, or stylistic commentary
- Break compound statements from the Generated Response into separate atomic
  claims, each self-contained and independently verifiable
- A claim is supported only if the Context explicitly contains the same
  information
- A claim is unsupported if the Context does not clearly state the same fact
- Do not use outside knowledge — only evaluate based on the Context above
- Claims that are common knowledge greetings or conversational filler (e.g.,
  "Sure, here's the information") should be ignored, not counted as
  unsupported

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


def evaluate_faithfulness(judge: LLMJudge, query: str, generated_response: str, context: str) -> Dict[str, Any]:
    """
    Evaluate whether all claims in the generated response are supported by
    the provided context.

    Args:
        judge: LLMJudge instance
        query: The user's original question (for reference)
        generated_response: The AI-generated response to evaluate
        context: The source material the system had access to

    Returns a dict with:
        - passed: True if all claims are supported, False otherwise
        - score: Fraction of supported claims (0.0 to 1.0)
        - reason: Summary of supported vs total claims
        - total_claims: Number of claims extracted
        - unsupported_claims: List of claims not found in the context
    """
    if not generated_response:
        return {"passed": False, "score": 0.0, "reason": "No generated response provided", "total_claims": 0, "unsupported_claims": []}
    if not context:
        return {"passed": False, "score": 0.0, "reason": "No context provided", "total_claims": 0, "unsupported_claims": []}

    prompt = EVAL_PROMPT.format(context=context, generated_response=generated_response)
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
        "reason": f"{supported_count} out of {total_claims} claims are supported by the provided context.",
        "total_claims": total_claims,
        "unsupported_claims": unsupported_claims,
    }


# Test data
SAMPLES = [
    {
        # All claims are directly supported by the context (expected: PASS)
        "query_id": 1,
        "query": "What are Python decorators and how are they used?",
        "generated_response": (
            "Python decorators use the @syntax to modify function behavior. "
            "They are commonly used for logging, authentication, and caching. "
            "Examples of built-in decorators are @staticmethod, @classmethod, and @property."
        ),
        "context": (
            "Document 1: Python decorators modify function behavior using the @syntax. "
            "They are used for logging, authentication, and caching. "
            "Built-in decorators include @staticmethod, @classmethod, and @property."
        ),
    },
    {
        # Response introduces a write overhead claim not present in the context — hallucination (expected: FAIL)
        "query_id": 2,
        "query": "How do database indexes improve query performance?",
        "generated_response": (
            "Database indexes speed up data retrieval by avoiding full table scans. "
            "B-tree indexes are commonly used. Indexes also add write overhead because "
            "they must be updated on every insert, update, or delete."
        ),
        "context": (
            "Document 1: Database indexes speed up data retrieval by avoiding full table scans. "
            "B-tree indexes are the most common type used for range queries and exact matches."
        ),
    },
    {
        # Response claims multi-threading which the context never mentions — hallucination (expected: FAIL)
        "query_id": 3,
        "query": "What are the benefits of async/await in Python?",
        "generated_response": (
            "Async/await improves code readability and I/O performance. "
            "It also enables true multi-threading, allowing CPU-bound tasks to run in parallel."
        ),
        "context": (
            "Document 1: Async/await allows writing asynchronous code that is more readable "
            "than callbacks. It improves I/O performance by not blocking the main thread "
            "during I/O operations."
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
        result = evaluate_faithfulness(
            judge,
            query=sample["query"],
            generated_response=sample["generated_response"],
            context=sample["context"],
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
