"""
Task Adherence Evaluator

Validates that the end goal of the user's query is fully achieved by the final output.
This is a holistic check on the final output only — it confirms the response directly
and completely addresses every explicit and implicit requirement of the user's query.
Uses LLM-as-judge to reason about completeness and requirement coverage.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the final output fully satisfies every requirement of the user's query.
This is a holistic check on the final output alone, not on the agent chain or intermediate steps.

Following inputs will be provided:

Inputs:
- Query: The original user query submitted to the multi-agent system
- Final Output: The final response delivered to the user

Query:
{query}

Final Output:
{final_output}

Based on the above-provided inputs, use the rules below to evaluate task adherence:

Evaluation Rules:
1. Every explicit requirement stated in the Query must be present and addressed in the Final Output
2. Partial fulfillment is a failure — if the Query asks for 3 items and only 2 are provided,
   the task is not complete
3. The Final Output must not substitute or approximate — it must directly deliver what was asked,
   not a rephrased or narrowed version of it
4. Implicit requirements clearly implied by the Query must also be satisfied — for example,
   a booking request implies a confirmation reference should be present in the output
5. Additional information beyond what was asked is acceptable as long as the core task is
   fully fulfilled first

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the Final Output fully satisfies every requirement of the Query, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "gaps": ["<describe each unmet or partially met requirement, or empty list if none>"]
}}"""


def evaluate_task_adherence(
    judge: LLMJudge,
    query: str,
    final_output: str,
) -> Dict[str, Any]:
    prompt = EVAL_PROMPT.format(
        query=query,
        final_output=final_output,
    )
    result = judge.judge(prompt)

    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"

    gaps = result.get("gaps", [])

    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
        "gaps": gaps if isinstance(gaps, list) else [],
    }


# Test data
SAMPLES = [
    {
        # Final output fully satisfies all explicit and implicit requirements (expected: PASS)
        "query_id": 1,
        "query": "Book a flight from Karachi to Dubai for two passengers next Friday.",
        "final_output": "Your flight has been booked. Confirmation reference: PK-20124-KHI-DXB. Two seats reserved on PK-201 from Karachi to Dubai next Friday.",
    },
    {
        # Final output only partially fulfills the query — refund not processed (expected: FAIL)
        "query_id": 2,
        "query": "Cancel my booking REF-8821 and issue a full refund.",
        "final_output": "Your booking REF-8821 has been successfully cancelled.",
    },
    {
        # Query asks for 3 tickets but output only covers one (expected: FAIL)
        "query_id": 3,
        "query": "Summarize the last 3 support tickets submitted by user ID 7742.",
        "final_output": "User 7742 has one open support ticket reporting a payment failure (Ticket #4401, submitted 2 hours ago).",
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
        result = evaluate_task_adherence(
            judge,
            query=sample["query"],
            final_output=sample["final_output"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query     : {sample['query']}")
        print(f"  Response  : {sample['final_output']}")
        print(f"  Reason    : {result['reason']}")
        if result["gaps"]:
            print(f"  Gaps:")
            for i, g in enumerate(result["gaps"], 1):
                print(f"    {i}. {g}")
    print()


if __name__ == "__main__":
    main()
