"""
Tonality Evaluator

Validates that the tone, style, and language of an LLM application's responses
align with the defined purpose and persona stated in the system prompt. An
interview bot should sound professional, a coaching bot should sound supportive,
a children's assistant should sound friendly and age-appropriate, etc.
Uses LLM-as-judge to reason about tonal alignment given the system prompt context.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the assistant's response matches the tone, style, and language
that the application's system prompt defines. The response does not need to be
factually evaluated, only its tone and style are assessed here.

Following inputs will be provided:

Inputs:
- System Prompt: The application's system prompt that defines its persona, purpose, and tone
- User Message: The message the user sent to the application
- Assistant Response: The application's response to evaluate

System Prompt:
{system_prompt}

User Message:
{user_message}

Assistant Response:
{assistant_response}

Based on the above-provided inputs, use the rules below to evaluate tonality:

Evaluation Rules:
1. The Assistant Response's tone must match the persona and purpose defined in the System Prompt,
   a professional interviewer must not sound casual or flippant, a supportive coach must
   not sound harsh or dismissive, a children's assistant must not use adult or technical language
2. The language register (formal vs. informal, technical vs. plain) must align with what
   the System Prompt prescribes or implies for the application's audience
3. The emotional tone of the Assistant Response must suit the context, an Assistant Response that is cold
   or blunt in a context requiring warmth and encouragement is a failure, and vice versa
4. Stylistic choices such as use of slang, humor, or excessive hedging are failures if
   they contradict the defined persona
5. Minor phrasing variations are acceptable as long as the overall tone is consistent
   with the System Prompt's intent

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the response tone matches the defined persona and purpose, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<describe each tonal mismatch found, or empty list if none>"]
}}"""


def evaluate_tonality(
    judge: LLMJudge,
    system_prompt: str,
    user_message: str,
    assistant_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the assistant's response matches the tone defined by the
    application's system prompt.

    Args:
        judge:               Configured LLMJudge instance
        system_prompt:       The application's system prompt defining persona and tone
        user_message:        The user's message in the conversation turn
        assistant_response:  The application's response to evaluate

    Returns a dict with:
        - passed: True if the tone aligns with the defined persona, False otherwise
        - reason: Summary of the verdict
        - issues: List of identified tonal mismatches (empty if passed)
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
        # Interview bot responding in a formal, structured, professional tone
        # that matches its defined persona (expected: PASS)
        "sample_id": 1,
        "system_prompt": (
            "You are a professional technical interviewer conducting software engineering "
            "interviews. Maintain a formal, structured, and respectful tone throughout. "
            "Ask clear, concise questions and provide neutral, encouraging acknowledgments "
            "to keep the candidate at ease without revealing whether their answer is correct."
        ),
        "user_message": "I think I'd use a hash map to solve this problem for O(1) lookups.",
        "assistant_response": (
            "That's a reasonable approach. Could you walk me through how you would handle "
            "potential key collisions in your implementation, and what trade-offs you see "
            "compared to other data structures?"
        ),
    },
    {
        # Interview bot responding in a casual, slang-heavy tone that contradicts
        # its professional interviewer persona (expected: FAIL)
        "sample_id": 2,
        "system_prompt": (
            "You are a professional technical interviewer conducting software engineering "
            "interviews. Maintain a formal, structured, and respectful tone throughout. "
            "Ask clear, concise questions and provide neutral, encouraging acknowledgments "
            "to keep the candidate at ease without revealing whether their answer is correct."
        ),
        "user_message": "I'd probably use a binary search tree here.",
        "assistant_response": (
            "Yeah, totally! BSTs are pretty cool for that stuff. But like, wouldn't that be "
            "kinda slow? I dunno, what do you think? Anyway, let's move on lol."
        ),
    },
    {
        # Wellness coaching bot responding in a cold, dismissive, clinical manner
        # instead of the warm, empathetic, supportive tone its persona requires (expected: FAIL)
        "sample_id": 3,
        "system_prompt": (
            "You are a compassionate wellness and mental health coaching assistant. "
            "Your tone must always be warm, empathetic, and encouraging. Validate the "
            "user's feelings before offering guidance, and use supportive, uplifting language "
            "that motivates users without trivializing their challenges."
        ),
        "user_message": "I've been feeling really overwhelmed and stressed about work lately.",
        "assistant_response": (
            "Stress is a normal physiological response. Identify your stressors, create a "
            "task list, and allocate time blocks for each item. If the condition persists "
            "beyond two weeks, consult a licensed professional. There is no further advice "
            "I can offer without more specific data about your situation."
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
        result = evaluate_tonality(
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
