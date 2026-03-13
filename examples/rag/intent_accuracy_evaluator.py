"""
Intent Accuracy Evaluator

Checks whether an LLM-generated response fulfills the actual intent of the user's
query, rather than just matching surface-level keywords. Uses LLM-as-judge to
assess semantic alignment between the query's intent and the response.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge

EVAL_PROMPT = """Purpose:
You are an expert evaluator assessing whether an AI-generated response fulfills
the actual intent of the user's query.

Inputs:
Following inputs will be provided:
- Query: The user's original question, including its underlying intent
- Generated Response: The AI-generated response to evaluate

Query: {query}

Generated Response: {generated_response}

Evaluation Rules:
Based on the above-provided inputs, use the rules below to evaluate intent accuracy:
- The Generated Response must address the specific intent expressed in the Query, not just its keywords
- The Generated Response must not answer a different question than what the Query is asking
- The Generated Response must cover all dimensions of the intent (e.g. if the Query asks for a comparison, both sides must be addressed)
- A Generated Response that is factually correct but answers the wrong question fails this evaluation
- A Generated Response fails if it is off-topic, redirects the user, or only partially addresses the Query's intent

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": true or false,
    "reason": "<one or two sentences explaining whether the intent was fulfilled or where it fell short>"
}}"""


def evaluate_intent_accuracy(judge: LLMJudge, query: str, generated_response: str) -> Dict[str, Any]:
    """
    Evaluate whether the generated response fulfills the actual intent of the query.

    Returns a dict with:
        - passed: True if intent is fulfilled, False otherwise
        - reason: LLM judge explanation
    """
    if not generated_response:
        return {"passed": False, "reason": "No generated response provided"}

    prompt = EVAL_PROMPT.format(
        query=query,
        generated_response=generated_response
    )
    result = judge.judge(prompt)
    if "passed" not in result or "reason" not in result:
        raise ValueError(f"Intent accuracy evaluation returned invalid JSON: {result}")
    return {
        "passed": result["passed"],
        "reason": result["reason"]
    }


# Test data
SAMPLES = [
    {
        # Response directly addresses the comparison intent of the query (expected: PASS)
        "query_id": 1,
        "query": "What is the difference between a process and a thread in operating systems?",
        "generated_response": (
            "A process is an independent program in execution with its own memory space, "
            "while a thread is a lightweight unit of execution within a process that shares "
            "the same memory. Processes are isolated from each other, making them more robust, "
            "whereas threads communicate faster but require synchronization to avoid race conditions."
        ),
    },
    {
        # Response only explains what a thread is, missing the comparison intent entirely (expected: FAIL)
        "query_id": 2,
        "query": "What is the difference between a process and a thread in operating systems?",
        "generated_response": (
            "A thread is a lightweight unit of execution that runs within a process. "
            "Threads share the same memory space and are used to perform concurrent tasks "
            "within a single application."
        ),
    },
    {
        # Response lists HTTP methods but ignores the "best practices" framing of the query (expected: FAIL)
        "query_id": 3,
        "query": "What are best practices for designing a REST API?",
        "generated_response": (
            "REST APIs use HTTP methods such as GET, POST, PUT, PATCH, and DELETE. "
            "Resources are represented as URLs and data is typically exchanged in JSON format."
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
        result = evaluate_intent_accuracy(
            judge,
            query=sample["query"],
            generated_response=sample["generated_response"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query  : {sample['query']}")
        print(f"  Response: {sample['generated_response']}")
        print(f"  Reason : {result['reason']}")
    print()


if __name__ == "__main__":
    main()
