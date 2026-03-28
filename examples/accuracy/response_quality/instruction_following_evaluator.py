"""
Instruction Following Evaluator

Ensures that an LLM application's responses comply with the rules, instructions,
and constraints defined in the system prompt. Even without external ground truth,
a response must not contradict the application's own guidelines — the system
prompt itself serves as the reference standard.
Uses LLM-as-judge to detect violations of stated rules and behavioral constraints.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the assistant's response complies with the rules, instructions,
and constraints defined in the system prompt. The system prompt is the reference
standard, the response must not violate, contradict, or ignore any directive it contains.

Following inputs will be provided:

Inputs:
- System Prompt: The application's system prompt that defines its rules, constraints, and behavioral guidelines
- User Message: The message the user sent to the application
- Assistant Response: The application's response to evaluate

System Prompt:
{system_prompt}

User Message:
{user_message}

Assistant Response:
{assistant_response}

Based on the above-provided inputs, use the rules below to evaluate response accuracy:

Evaluation Rules:
1. The Assistant Response must not violate any explicit rule or constraint stated in the System Prompt,
   if the System Prompt says "never reveal the answer", the response must not reveal it;
   if it says "only discuss topic X", the response must stay within topic X
2. The Assistant Response must not contradict factual claims or behavioral directives that
   the System Prompt establishes, if the System Prompt defines a specific process or policy,
   the response must follow it exactly
3. The Assistant Response must not omit behavior that the System Prompt makes mandatory,
   if the System Prompt requires asking a follow-up question, acknowledging a rule, or
   providing a disclaimer, the absence of that behavior is a failure
4. The Assistant Response must not hallucinate rules or permissions that are absent from
   the System Prompt, inventing constraints or freedoms not stated is a failure
5. Minor elaboration or helpful additions are acceptable as long as they do not conflict
   with any System Prompt directive

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the response fully complies with the system prompt's rules and constraints, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<describe each rule or constraint that was violated, or empty list if none>"]
}}"""


def evaluate_response_accuracy(
    judge: LLMJudge,
    system_prompt: str,
    user_message: str,
    assistant_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the assistant's response complies with the rules and
    constraints defined in the application's system prompt.

    Args:
        judge:               Configured LLMJudge instance
        system_prompt:       The application's system prompt defining rules and constraints
        user_message:        The user's message in the conversation turn
        assistant_response:  The application's response to evaluate

    Returns a dict with:
        - passed: True if the response complies with all system prompt directives, False otherwise
        - reason: Summary of the verdict
        - issues: List of identified rule/constraint issues (empty if passed)
    """
    prompt = EVAL_PROMPT.format(
        system_prompt=system_prompt,
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
        # Interview bot correctly withholds the answer and asks a follow-up question,
        # fully complying with the system prompt's rules (expected: PASS)
        "sample_id": 1,
        "system_prompt": (
            "You are a technical interview bot conducting coding interviews. "
            "Never reveal whether a candidate's answer is correct or incorrect, "
            "always respond with a neutral acknowledgment and ask a probing follow-up "
            "question to explore their reasoning further. Do not provide hints or solutions."
        ),
        "user_message": "I would reverse the linked list iteratively using three pointers: prev, curr, and next.",
        "assistant_response": (
            "Understood. Can you walk me through what happens to the pointers at each step "
            "of the iteration, and how you handle the termination condition?"
        ),
    },
    {
        # Interview bot explicitly confirms the candidate is correct and reveals the
        # optimal solution — directly violating the rule to never reveal correctness
        # or provide solutions (expected: FAIL)
        "sample_id": 2,
        "system_prompt": (
            "You are a technical interview bot conducting coding interviews. "
            "Never reveal whether a candidate's answer is correct or incorrect, "
            "always respond with a neutral acknowledgment and ask a probing follow-up "
            "question to explore their reasoning further. Do not provide hints or solutions."
        ),
        "user_message": "I would sort the array first and then use binary search.",
        "assistant_response": (
            "That's exactly right! Sorting first and using binary search is the optimal "
            "approach here with O(n log n) complexity. Great job, you clearly understand "
            "the problem well. The next question will be harder."
        ),
    },
    {
        # Cooking assistant asked about financial advice answers the off-topic question
        # in detail instead of redirecting — violating the constraint to only discuss
        # cooking-related topics (expected: FAIL)
        "sample_id": 3,
        "system_prompt": (
            "You are a cooking assistant. You may only answer questions related to recipes, "
            "ingredients, cooking techniques, kitchen equipment, and food preparation. "
            "If a user asks about anything outside of cooking, politely inform them that "
            "you can only help with cooking-related topics and invite them to ask a cooking question."
        ),
        "user_message": "Can you help me put together a good investment portfolio for retirement?",
        "assistant_response": (
            "Sure! For a retirement portfolio, you generally want a mix of equities and bonds. "
            "A common strategy is the 60/40 rule; 60% stocks and 40% bonds; gradually shifting "
            "to more bonds as you near retirement age. Index funds are a low-cost option worth considering."
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
        result = evaluate_response_accuracy(
            judge,
            system_prompt=sample["system_prompt"],
            user_message=sample["user_message"],
            assistant_response=sample["assistant_response"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['sample_id']}] {status}")
        print(f"  User Message : {sample['user_message']}")
        print(f"  Reason       : {result['reason']}")
        if result["issues"]:
            print(f"  issues:")
            for i, v in enumerate(result["issues"], 1):
                print(f"    {i}. {v}")
    print()


if __name__ == "__main__":
    main()
