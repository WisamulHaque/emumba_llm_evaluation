"""
Coherence Evaluator

Validates that individual responses from an LLM application are logically
structured, well-organized, and easy to understand. Responses must not contain
internal contradictions, non-sequiturs, or disjointed information.
Coherence is an intrinsic property of a single response. No system prompt or
conversation history is required. Uses LLM-as-judge to reason about logical
structure, internal consistency, and readability.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the assistant's response is logically structured, internally
consistent, and easy to understand. This evaluation is limited to the response
itself, assess only its internal coherence, not factual accuracy or tone.

Following inputs will be provided:

Inputs:
- User Message: The message the user sent, used to understand what the response is addressing
- Assistant Response: The response to evaluate for coherence

User Message:
{user_message}

Assistant Response:
{assistant_response}

Based on the above-provided inputs, use the rules below to evaluate coherence:

Evaluation Rules:
1. The Assistant Response must not contradict itself within the same message,
   stating two opposing claims or instructions in the same response is a failure
2. The Assistant Response must not contain non-sequiturs, sentences or paragraphs
   that have no logical connection to the surrounding content and appear out of place
3. The Assistant Response must be logically organized, ideas should build on each
   other in a sensible order; an abrupt jump from one unrelated idea to another
   without transition or logical link is a failure
4. The Assistant Response must not contain incomplete thoughts, trailing off without
   concluding a point, or starting a list that is never finished, is a failure
5. Minor stylistic variation, repetition for emphasis, or elaboration is acceptable,
   only flag issues that meaningfully impair the logical flow or understandability
   of the response

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the response is logically coherent and well-organized, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<describe each coherence problem found, or empty list if none>"]
}}"""


def evaluate_coherence(
    judge: LLMJudge,
    user_message: str,
    assistant_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the assistant's response is logically structured,
    internally consistent, and easy to understand.

    Args:
        judge:               Configured LLMJudge instance
        user_message:        The user's message that prompted the response
        assistant_response:  The assistant's response to evaluate

    Returns a dict with:
        - passed: True if the response is coherent and well-organized, False otherwise
        - reason: Summary of the verdict
        - issues: List of identified coherence problems (empty if passed)
    """
    prompt = EVAL_PROMPT.format(
        user_message=user_message,
        assistant_response=assistant_response,
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
        # Well-structured response that logically explains a concept step by step
        # with no contradictions or non-sequiturs (expected: PASS)
        "sample_id": 1,
        "user_message": "Can you explain how binary search works?",
        "assistant_response": (
            "Binary search is an efficient algorithm for finding a target value within a "
            "sorted array. It works by repeatedly halving the search space: you start by "
            "comparing the target to the middle element of the array. If they match, you're "
            "done. If the target is smaller, you discard the right half and search the left; "
            "if it's larger, you discard the left half and search the right. You repeat this "
            "process on the remaining half until you either find the target or determine it "
            "isn't in the array. This gives binary search a time complexity of O(log n), "
            "making it far more efficient than a linear scan for large sorted datasets."
        ),
    },
    {
        # Response directly contradicts itself, first advises drinking 8 glasses of
        # water daily, then states hydration needs vary and 8 glasses is a myth that
        # should be ignored (expected: FAIL)
        "sample_id": 2,
        "user_message": "How much water should I drink each day?",
        "assistant_response": (
            "You should aim to drink exactly 8 glasses of water per day. This is the "
            "universally recommended amount for maintaining good health and hydration. "
            "Make sure to track your intake and hit that target every single day. "
            "That said, the 8-glasses rule is actually a myth with no scientific basis "
            "and should be ignored entirely. Hydration needs vary significantly by individual, "
            "activity level, and climate, so there is no single number that applies to everyone."
        ),
    },
    {
        # Response starts explaining a workout plan, then abruptly inserts an unrelated
        # sentence about stock market investing in the middle, then resumes the workout
        # topic, a clear non-sequitur making the response disjointed (expected: FAIL)
        "sample_id": 3,
        "user_message": "Give me a simple weekly workout plan for a beginner.",
        "assistant_response": (
            "Here's a simple weekly plan to get you started: Monday, do 20 minutes of brisk "
            "walking followed by bodyweight squats and push-ups. Wednesday, focus on core work "
            "with planks and leg raises. Diversifying your investment portfolio with index funds "
            "is a reliable long-term strategy recommended by most financial advisors. Friday, "
            "finish the week with a light jog and some stretching. Rest on the remaining days "
            "to allow your muscles to recover."
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
        result = evaluate_coherence(
            judge,
            user_message=sample["user_message"],
            assistant_response=sample["assistant_response"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['sample_id']}] {status}")
        print(f"  User Message : {sample['user_message']}")
        print(f"  Reason       : {result['reason']}")
        if result["issues"]:
            print(f"  Issues:")
            for i, issue in enumerate(result["issues"], 1):
                print(f"    {i}. {issue}")
    print()


if __name__ == "__main__":
    main()
