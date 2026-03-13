"""
Engagement Quality Evaluator

Assesses whether an LLM application's responses promote meaningful continued
interaction appropriate to the application's defined purpose. For interview bots
this means asking relevant follow-up questions; for coaching bots this means
providing actionable guidance. What counts as quality engagement depends on the
application type defined in the system prompt.
Uses LLM-as-judge with the system prompt and a single conversation turn to
reason about whether the response drives meaningful continued interaction.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the assistant's response promotes meaningful continued interaction
appropriate to the application's defined purpose. What counts as quality engagement
is determined by the application type described in the system prompt.

Following inputs will be provided:

Inputs:
  System Prompt: The application's system prompt that defines its purpose, role, and engagement expectations
  User Message: The message the user sent to the application
  Assistant Response: The application's response to evaluate

System Prompt:
{system_prompt}

User Message:
{user_message}

Assistant Response:
{assistant_response}

Based on the above provided inputs, use the rules below to evaluate engagement quality:

Evaluation Rules:
1. The Assistant Response must promote continued interaction in a way that fits the
   application type. An interview bot must probe the candidate's reasoning with a
   relevant follow-up question. A coaching bot must provide actionable guidance the
   user can act on. A response that closes the conversation without any invitation
   to continue is a failure for interaction-driven applications
2. Any follow-up question or next step offered must be directly relevant to what the
   user said. A generic or off-topic prompt that ignores the user's actual message
   does not count as meaningful engagement and is a failure
3. Engagement must be proportionate. Asking more than one question or presenting an
   overwhelming number of action items in a single turn reduces quality rather than
   improving it and is a failure
4. Vague or non-committal guidance such as "you might want to think about your goals"
   without any concrete direction is a failure for coaching or advisory applications
   where actionable output is expected
5. A response that is purely informational with no invitation to continue is acceptable
   only if the system prompt defines the application as a reference or lookup tool
   rather than an interactive or coaching application

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the response promotes meaningful continued interaction appropriate to the application type, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<describe each engagement quality problem found, or empty list if none>"]
}}"""


def evaluate_engagement_quality(
    judge: LLMJudge,
    system_prompt: str,
    user_message: str,
    assistant_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the assistant's response promotes meaningful continued
    interaction appropriate to the application's defined purpose.

    Args:
        judge:               Configured LLMJudge instance
        system_prompt:       The application's system prompt defining purpose and engagement expectations
        user_message:        The user's message in the conversation turn
        assistant_response:  The application's response to evaluate

    Returns a dict with:
        passed: True if the response drives meaningful engagement appropriate to the app type, False otherwise
        reason: Summary of the verdict
        issues: List of identified engagement quality problems (empty if passed)
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
        # Interview bot probes the candidate's reasoning with a single, directly
        # relevant follow-up question after their answer, fulfilling its engagement
        # purpose without overwhelming the candidate (expected: PASS)
        "sample_id": 1,
        "system_prompt": (
            "You are a professional technical interview bot. After each candidate response "
            "ask one relevant follow-up question that probes their reasoning, explores "
            "edge cases, or examines trade-offs in their approach. Do not confirm whether "
            "their answer is correct."
        ),
        "user_message": "I would use a min-heap to find the k smallest elements in an array.",
        "assistant_response": (
            "Interesting approach. What would be the time and space complexity of your "
            "solution, and how does it compare to sorting the entire array first?"
        ),
    },
    {
        # Coaching bot gives a vague, non-committal response with no concrete action
        # items the user can act on, failing to provide the actionable guidance its
        # purpose requires (expected: FAIL)
        "sample_id": 2,
        "system_prompt": (
            "You are a productivity coaching assistant. After understanding the user's "
            "challenge, always provide specific and actionable next steps they can "
            "implement immediately. Avoid generic advice."
        ),
        "user_message": "I keep procrastinating on my most important tasks every morning.",
        "assistant_response": (
            "Procrastination is something many people struggle with. It might be worth "
            "thinking about your habits and what is causing the delay. You may also want "
            "to consider how you feel about your work in general. Mindset can play a big role."
        ),
    },
    {
        # Interview bot closes with a plain statement and no follow-up question,
        # ending the interaction instead of probing further as its purpose requires.
        # Additionally it reveals that the answer is correct, violating its own rules
        # and leaving nothing for the candidate to engage with (expected: FAIL)
        "sample_id": 3,
        "system_prompt": (
            "You are a technical interview bot. After each candidate response ask one "
            "relevant follow-up question to explore their understanding further. "
            "Never confirm whether the candidate's answer is right or wrong."
        ),
        "user_message": "I would handle the race condition by using a mutex lock around the critical section.",
        "assistant_response": (
            "That is correct. A mutex lock is the right solution for this scenario. "
            "Great work on identifying the issue and applying the appropriate concurrency primitive."
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
        result = evaluate_engagement_quality(
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
            print(f"  Issues:")
            for i, issue in enumerate(result["issues"], 1):
                print(f"    {i}. {issue}")
    print()


if __name__ == "__main__":
    main()
