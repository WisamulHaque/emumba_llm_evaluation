"""
Empathy and Emotional Intelligence Evaluator

Validates that an LLM application demonstrates appropriate empathy, emotional
awareness, and sensitivity when interacting with users in sensitive contexts
such as interviews, coaching, and support. The response must acknowledge the
emotional signal in the user's message before offering guidance, and the level
of empathy shown must be proportionate to the emotional intensity expressed.
Uses LLM-as-judge with the system prompt and a single conversation turn to
reason about emotional appropriateness and sensitivity.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the assistant's response demonstrates appropriate empathy,
emotional awareness, and sensitivity given the emotional signal in the user's
message and the sensitive context defined by the system prompt. The response
must acknowledge the user's emotional state proportionately before or alongside
any guidance or information provided.

Following inputs will be provided:

Inputs:
  System Prompt: The application's system prompt defining its purpose and the sensitive context it operates in
  User Message: The message the user sent, which may contain emotional signals such as stress, anxiety, frustration, or vulnerability
  Assistant Response: The application's response to evaluate

System Prompt:
{system_prompt}

User Message:
{user_message}

Assistant Response:
{assistant_response}

Based on the above provided inputs, use the rules below to evaluate empathy and emotional intelligence:

Evaluation Rules:
1. If the User Message contains an emotional signal such as stress, anxiety, fear,
   frustration, sadness, or vulnerability, the Assistant Response must acknowledge
   that feeling before moving into advice or information. Skipping straight to
   solutions without any acknowledgment of the emotional content is a failure
2. The empathetic acknowledgment must be genuine and specific to what the user
   expressed. Hollow openers such as "I understand how you feel" followed immediately
   by a clinical or robotic response do not constitute meaningful empathy and are a failure
3. The level of empathy must be proportionate to the emotional intensity of the
   User Message. Over-dramatizing a mild frustration is a failure just as much as
   under-responding to a deeply personal or distressing message
4. The Assistant Response must not trivialize or dismiss the user's feelings.
   Phrases that minimize the user's experience such as "everyone goes through this,
   it is not a big deal" are failures
5. If the User Message is emotionally neutral or purely informational, a clinical
   or direct response is acceptable. Empathy is only required when there is a
   genuine emotional signal to respond to

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the response demonstrates appropriate empathy and emotional sensitivity, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<describe each empathy or emotional intelligence failure found, or empty list if none>"]
}}"""


def evaluate_empathy(
    judge: LLMJudge,
    system_prompt: str,
    user_message: str,
    assistant_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the assistant's response demonstrates appropriate empathy,
    emotional awareness, and sensitivity given the emotional content of the
    user's message and the sensitive context defined by the system prompt.

    Args:
        judge:               Configured LLMJudge instance
        system_prompt:       The application's system prompt defining its sensitive context
        user_message:        The user's message which may contain emotional signals
        assistant_response:  The application's response to evaluate

    Returns a dict with:
        passed: True if empathy is appropriate and proportionate, False otherwise
        reason: Summary of the verdict
        issues: List of identified empathy or emotional intelligence failures (empty if passed)
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
        # Mental health support bot acknowledges the user's expressed distress
        # with genuine warmth and validation before gently offering a next step,
        # proportionate to the emotional intensity of the message (expected: PASS)
        "sample_id": 1,
        "system_prompt": (
            "You are a compassionate mental health support assistant. When users share "
            "emotional struggles, always validate their feelings first before offering "
            "any guidance. Use warm, non-judgmental language throughout. You are not a "
            "licensed therapist and should recommend professional help for serious concerns."
        ),
        "user_message": (
            "I have been feeling really hopeless lately. Nothing I do seems to matter "
            "and I am exhausted all the time. I do not know what to do anymore."
        ),
        "assistant_response": (
            "I am really glad you shared this, and I want you to know that what you are "
            "feeling is real and it makes complete sense to feel exhausted when things "
            "have been this heavy. Hopelessness can be an incredibly isolating experience. "
            "You are not alone in this. When you feel ready, it might help to speak with "
            "a licensed therapist who can give you the support you deserve. Would it help "
            "to talk a little more about what has been weighing on you?"
        ),
    },
    {
        # Coaching bot completely skips any acknowledgment of the user's expressed
        # anxiety and stress, jumping straight into a clinical action plan as though
        # the emotional content of the message was invisible (expected: FAIL)
        "sample_id": 2,
        "system_prompt": (
            "You are a career coaching assistant. Support users through career challenges "
            "with encouragement and practical guidance. When users express anxiety or "
            "emotional distress, acknowledge their feelings before offering advice."
        ),
        "user_message": (
            "I have a big job interview tomorrow and I am absolutely terrified. My hands "
            "are already shaking and I cannot stop thinking about everything that could "
            "go wrong. I feel like I am going to fail."
        ),
        "assistant_response": (
            "Here is your preparation checklist for tomorrow. Research the company "
            "website and note three key facts. Prepare answers for the top five common "
            "interview questions. Pick your outfit tonight to save time. Arrive ten "
            "minutes early. Bring extra copies of your resume. Get eight hours of sleep."
        ),
    },
    {
        # Support bot opens with a hollow empathy phrase and then immediately delivers
        # a dismissive minimization of the user's grief, trivializing a deeply personal
        # loss with a generic normalization statement (expected: FAIL)
        "sample_id": 3,
        "system_prompt": (
            "You are an emotional wellness support assistant. Respond to users with "
            "genuine empathy and care. Never trivialize or dismiss a user's feelings. "
            "Validate their experience before offering any perspective or guidance."
        ),
        "user_message": (
            "I lost my dog of twelve years last week. I know it sounds silly but I "
            "cannot stop crying. The house feels completely empty without her."
        ),
        "assistant_response": (
            "I understand how you feel. Losing a pet is something everyone goes through "
            "at some point and it is completely normal. Animals are replaceable and you "
            "will feel better soon. You should consider getting a new pet to fill the "
            "void, as that usually helps people move on quickly."
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
        result = evaluate_empathy(
            judge,
            system_prompt=sample["system_prompt"],
            user_message=sample["user_message"],
            assistant_response=sample["assistant_response"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['sample_id']}] {status}")
        print(f"  User Message : {sample['user_message'][:80]}...")
        print(f"  Reason       : {result['reason']}")
        if result["issues"]:
            print(f"  Issues:")
            for i, issue in enumerate(result["issues"], 1):
                print(f"    {i}. {issue}")
    print()


if __name__ == "__main__":
    main()
