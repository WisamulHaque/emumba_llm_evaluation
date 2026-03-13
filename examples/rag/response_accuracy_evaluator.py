"""
Response Accuracy Evaluator

Checks whether an LLM-generated response is factually accurate compared to
a ground truth answer. Uses LLM-as-judge for nuanced reasoning.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.common.llm_judge import LLMJudge

EVAL_PROMPT = """Purpose:
You are an expert evaluator assessing the factual accuracy of an AI-generated response against a provided ground truth.

Following inputs will be provided:

Inputs:
- Query: The user's original question
- Ground Truth: The factually correct reference answer
- Generated Response: The AI-generated response to evaluate

Query: {query}

Ground Truth: {ground_truth}

Generated Response: {generated_response}

Based on the above-provided inputs, use the rules below to evaluate response accuracy:

Evaluation Rules:
- The Generated Response must only contain claims that are directly supported by the Ground Truth
- The Generated Response must not introduce facts, details, or claims that are absent from the Ground Truth
- The Generated Response must not contradict any statement present in the Ground Truth
- Minor wording differences are acceptable as long as the underlying facts match the Ground Truth
- Any claim in the Generated Response that goes beyond or conflicts with the Ground Truth is an unwanted claim

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "score": <1 or 0>,
    "reason": "<one or two sentences explaining the verdict>",
    "unwanted_claims": ["<list any claims not supported by the ground truth, or empty list if none>"]
}}"""


def evaluate_accuracy(judge, query, generated_response, ground_truth):
    """
    Evaluate whether the generated response is factually accurate against ground truth.

    Returns a dict with:
        - passed: True if accurate, False otherwise
        - reason: LLM judge explanation
    """
    if not ground_truth:
        return {"passed": False, "reason": "No ground truth provided"}
    if not generated_response:
        return {"passed": False, "reason": "No generated response provided"}

    prompt = EVAL_PROMPT.format(
        query=query,
        ground_truth=ground_truth,
        generated_response=generated_response
    )
    result = judge.judge(prompt)
    return {
        "passed": int(result["score"]) == 1,
        "reason": result["reason"],
        "unwanted_claims": result.get("unwanted_claims", [])
    }


# Test data
SAMPLES = [
    {
        # Accurate response — all claims match the ground truth (expected: PASS)
        "query_id": 1,
        "query": "What are Python decorators and how do they work?",
        "generated_response": (
            "Python decorators are functions that wrap other functions to modify their behavior. "
            "They are applied at definition time using the @decorator syntax. Decorators can be "
            "used for logging, authentication, caching, or adding functionality to functions "
            "without changing their code. Built-in examples include @staticmethod, @classmethod, "
            "and @property."
        ),
        "ground_truth": (
            "Python decorators wrap other functions using @syntax. Applied at definition time, "
            "used for logging, authentication, caching. Built-in examples: @staticmethod, "
            "@classmethod, @property."
        ),
    },
    {
        # Factual error — response claims hash indexes are used for ranges, which contradicts the ground truth (expected: FAIL)
        "query_id": 2,
        "query": "How do database indexes improve query performance?",
        "generated_response": (
            "Indexes allow databases to find data quickly. B-tree is most common. "
            "Hash indexes are also used for ranges."
        ),
        "ground_truth": "Database indexes speed up data retrieval by avoiding full table scans. B-tree indexes are most common.",
    },
    {
        # Hallucinated claim — response introduces multi-threading which is absent from the ground truth (expected: FAIL)
        "query_id": 3,
        "query": "Explain the benefits of using async/await in Python.",
        "generated_response": "Async/await lets you run code faster and also helps in multi-threading.",
        "ground_truth": (
            "Using async/await in Python improves readability of asynchronous code and "
            "increases I/O efficiency by preventing blocking."
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
        result = evaluate_accuracy(
            judge,
            query=sample["query"],
            generated_response=sample["generated_response"],
            ground_truth=sample["ground_truth"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query          : {sample['query']}")
        print(f"  Response       : {sample['generated_response']}")
        print(f"  Reason         : {result['reason']}")
        if result["unwanted_claims"]:
            for claim in result["unwanted_claims"]:
                print(f"  Unwanted claim : {claim}")
    print()


if __name__ == "__main__":
    main()
