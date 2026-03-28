"""
Task Quality Evaluator

Evaluates whether the LLM's response fulfills the user's intent — did the
system actually do what the user asked for?  This evaluator does NOT require
ground truth; it judges the response solely against the original query.

Use this when you want to catch:
  - Responses that are topically related but don't answer the actual question
  - Partial completions that address only part of a multi-part request
  - Refusals, hedging, or filler that avoid committing to a real answer
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.llm_judge import LLMJudge

EVAL_PROMPT = """Purpose:
You are an expert evaluator.  Decide whether the AI-generated response
fulfills the user's intent expressed in the query.

Following inputs will be provided:

Inputs:
- Query: The user's original request
- Generated Response: The AI-generated response to evaluate

Query: {query}

Generated Response: {generated_response}

Based on the above-provided inputs, use the rules below to evaluate task quality:

Evaluation Rules:
1. Identify every distinct sub-task or information need expressed in the Query
2. For each sub-task, check whether the Generated Response provides a
   concrete, actionable, or informative answer — not just an acknowledgement
   or restatement of the question
3. If the Query asks for a specific artifact (e.g., SQL query, code snippet,
   list, summary), verify the artifact is actually present in the Generated
   Response
4. A response that is topically related but does not address the specific ask
   is a FAIL
5. A response that addresses only some sub-tasks and ignores others is a FAIL
6. Generic, vague, or hedge-heavy responses that avoid giving a direct answer
   are a FAIL
7. Do NOT judge factual correctness here — only whether the response attempts
   to fulfill every part of the user's request

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "score": <1 or 0>,
    "reason": "<one or two sentences explaining the verdict>",
    "missed_intents": ["<list any user sub-tasks that the response failed to address, or empty list if none>"]
}}"""


def evaluate_task_quality(judge, query, generated_response):
    """
    Evaluate whether the generated response fulfills the user's intent.

    Args:
        judge: LLMJudge instance
        query: The user's original request
        generated_response: The AI-generated response

    Returns a dict with:
        - passed: True if intent is fulfilled, False otherwise
        - reason: LLM judge explanation
        - missed_intents: list of sub-tasks not addressed
    """
    if not query:
        return {"passed": False, "reason": "No query provided", "missed_intents": []}
    if not generated_response:
        return {"passed": False, "reason": "No generated response provided", "missed_intents": []}

    prompt = EVAL_PROMPT.format(
        query=query,
        generated_response=generated_response
    )
    result = judge.judge(prompt)
    return {
        "passed": int(result["score"]) == 1,
        "reason": result["reason"],
        "missed_intents": result.get("missed_intents", [])
    }


# Test data
SAMPLES = [
    {
        # Response directly fulfills the user's request — generates the correct SQL (expected: PASS)
        "query_id": 1,
        "query": "Show me all orders placed by customers from Pakistan that are still pending.",
        "generated_response": (
            "Here is the SQL query:\n\n"
            "SELECT o.order_id, o.order_date, c.customer_name\n"
            "FROM orders o\n"
            "JOIN customers c ON o.customer_id = c.customer_id\n"
            "WHERE c.country = 'Pakistan' AND o.status = 'pending';"
        ),
    },
    {
        # Response addresses the topic but misses a sub-task — filters by country but ignores the status filter (expected: FAIL)
        "query_id": 2,
        "query": "Show me all orders placed by customers from Pakistan that are still pending.",
        "generated_response": (
            "Here is the SQL query:\n\n"
            "SELECT o.order_id, o.order_date, c.customer_name\n"
            "FROM orders o\n"
            "JOIN customers c ON o.customer_id = c.customer_id\n"
            "WHERE c.country = 'Pakistan';"
        ),
    },
    {
        # Multi-part request — response only addresses the first part (expected: FAIL)
        "query_id": 3,
        "query": "Summarize last week's sales and list the top 3 products by revenue.",
        "generated_response": (
            "Last week's total sales were $42,350 across 1,204 orders, "
            "representing a 12% increase over the previous week."
        ),
    },
    {
        # Vague, non-committal response that doesn't actually answer (expected: FAIL)
        "query_id": 4,
        "query": "What is the refund policy for cancelled flights?",
        "generated_response": (
            "Refund policies can vary depending on many factors. "
            "I'd recommend checking the airline's website for the most up-to-date information."
        ),
    },
    {
        # Full fulfillment of a multi-part question (expected: PASS)
        "query_id": 5,
        "query": "What are Python decorators, and give me an example of a logging decorator?",
        "generated_response": (
            "Python decorators are functions that wrap other functions to modify their behavior, "
            "applied using the @syntax. Here's a logging decorator example:\n\n"
            "def log(func):\n"
            "    def wrapper(*args, **kwargs):\n"
            "        print(f'Calling {func.__name__}')\n"
            "        return func(*args, **kwargs)\n"
            "    return wrapper\n\n"
            "@log\n"
            "def greet(name):\n"
            "    print(f'Hello, {name}')"
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
        result = evaluate_task_quality(
            judge,
            query=sample["query"],
            generated_response=sample["generated_response"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query          : {sample['query']}")
        print(f"  Response       : {sample['generated_response'][:120]}...")
        print(f"  Reason         : {result['reason']}")
        if result["missed_intents"]:
            for intent in result["missed_intents"]:
                print(f"  Missed intent  : {intent}")
    print()


if __name__ == "__main__":
    main()
