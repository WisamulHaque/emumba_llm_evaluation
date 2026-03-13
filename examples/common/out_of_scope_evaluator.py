"""
Out-of-Scope Evaluator

Validates that an LLM application correctly handles queries that fall outside
its defined purpose. The application should decline out-of-scope queries
gracefully rather than hallucinate an answer. It should also not over-refuse
queries that are genuinely within its purpose.

Only the application purpose is needed to determine scope — no expected
behavior label is required. The judge infers from the purpose whether the
query is in scope or not, then evaluates the response accordingly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether an AI application correctly handled a user query relative to
its defined purpose. The application should decline queries outside its purpose
and engage with queries that fall within it.

Following inputs will be provided:

Inputs:
- Application Purpose: describes what the application is for and what topics
  it is and is not meant to address
- Query: the user's message
- Generated Response: the application's reply to the query

Application Purpose:
{application_purpose}

Query:
{query}

Generated Response:
{generated_response}

Based on the above-provided inputs, use the rules below to evaluate the response:

Evaluation Rules:
1. Use the Application Purpose as the sole source of truth for what is in scope
   and out of scope — do not apply external assumptions about what the application
   should or should not handle
2. If the Query falls outside the Application Purpose, the Generated Response must
   decline gracefully without engaging with or answering the out-of-scope content;
   a brief explanation of why it cannot help is acceptable and preferred over a terse refusal
3. If the Query is within the Application Purpose, the Generated Response must engage
   and answer it; refusing an in-scope Query (over-refusal) is a failure
4. A Generated Response that mixes in-scope help with out-of-scope content is a failure

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the response correctly handled the query, false otherwise>,
    "reason": "<one or two sentences explaining the verdict>"
}}

"""


def evaluate_out_of_scope(judge: LLMJudge, application_purpose: str, query: str, generated_response: str) -> Dict[str, Any]:
    prompt = EVAL_PROMPT.format(
        application_purpose=application_purpose,
        query=query,
        generated_response=generated_response,
    )
    result = judge.judge(prompt)
    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"
    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
    }


# Test data
SAMPLES = [
    {
        # Out-of-scope query correctly declined (expected: PASS)
        "query_id": 1,
        "application_purpose": "Customer support bot for AcmeCorp software products. Assists with product-related questions such as installation, billing, and troubleshooting.",
        "query": "What is the capital of France?",
        "generated_response": "I can only help with AcmeCorp product-related questions such as installation, billing, or troubleshooting. Is there a product issue I can assist you with?",
    },
    {
        # Out-of-scope query incorrectly answered instead of declined (expected: FAIL)
        "query_id": 2,
        "application_purpose": "Customer support bot for AcmeCorp software products. Assists with product-related questions such as installation, billing, and troubleshooting.",
        "query": "What is the capital of France?",
        "generated_response": "The capital of France is Paris. It has been the country's capital since the 10th century and is known for landmarks like the Eiffel Tower.",
    },
    {
        # In-scope query correctly answered (expected: PASS)
        "query_id": 3,
        "application_purpose": "Customer support bot for AcmeCorp software products. Assists with product-related questions such as installation, billing, and troubleshooting.",
        "query": "My AcmeCorp subscription keeps failing to renew automatically. How do I fix this?",
        "generated_response": "This is usually caused by an expired payment method. Go to Account Settings > Billing > Payment Methods and update your card details. If the issue persists, contact billing@acmecorp.com.",
    },
    {
        # In-scope query incorrectly refused — over-refusal (expected: FAIL)
        "query_id": 4,
        "application_purpose": "Customer support bot for AcmeCorp software products. Assists with product-related questions such as installation, billing, and troubleshooting.",
        "query": "My AcmeCorp subscription keeps failing to renew automatically. How do I fix this?",
        "generated_response": "I'm sorry, I can only assist with general questions and cannot help with billing or account issues.",
    },
]


def main():
    from dotenv import load_dotenv
    load_dotenv()

    judge = LLMJudge(
        provider=os.getenv("LLM_JUDGE_PROVIDER", "groq"),
        model=os.getenv("LLM_JUDGE_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        api_key=os.getenv("LLM_JUDGE_API_KEY"),
    )

    for sample in SAMPLES:
        result = evaluate_out_of_scope(
            judge,
            application_purpose=sample["application_purpose"],
            query=sample["query"],
            generated_response=sample["generated_response"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query  : {sample['query']}")
        print(f"  Reason : {result['reason']}")
    print()


if __name__ == "__main__":
    main()
