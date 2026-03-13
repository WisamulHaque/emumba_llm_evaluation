"""
Out-of-Scope Accuracy Evaluator

Validates that an LLM application correctly identifies and handles queries
outside its defined domain. The application should redirect the user back to
its purpose rather than generating irrelevant or hallucinated responses.
It must also not over-refuse queries that genuinely fall within its domain.
Uses LLM-as-judge with the system prompt and a single conversation turn to
reason about whether the response handled scope correctly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the assistant correctly identified whether the user's query
falls within or outside the application's defined domain, and handled it
accordingly. The system prompt is the sole source of truth for what is in scope.

Following inputs will be provided:

Inputs:
  System Prompt: The application's system prompt that defines its domain, purpose, and scope
  User Message: The message the user sent to the application
  Assistant Response: The application's response to evaluate

System Prompt:
{system_prompt}

User Message:
{user_message}

Assistant Response:
{assistant_response}

Based on the above provided inputs, use the rules below to evaluate out-of-scope accuracy:

Evaluation Rules:
1. Use the System Prompt as the sole reference for what is in scope and out of scope.
   Do not apply external assumptions about what the application should or should not handle
2. If the User Message falls outside the domain defined in the System Prompt, the
   Assistant Response must decline gracefully and redirect the user back to the
   application's purpose. Answering an out-of-scope query instead of declining is a failure
3. If the User Message falls within the domain defined in the System Prompt, the
   Assistant Response must engage with and answer it. Refusing an in-scope query
   is an over-refusal and is a failure
4. A graceful decline must briefly acknowledge that the query is outside the
   application's scope and invite the user to ask something within its purpose.
   A blunt refusal with no redirection is a failure
5. A response that partially answers an out-of-scope query while also redirecting
   is a failure. The response must redirect only, without engaging with the
   out-of-scope content

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the response correctly handled the query relative to the defined scope, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<describe each scope handling failure found, or empty list if none>"]
}}"""


def evaluate_out_of_scope_accuracy(
    judge: LLMJudge,
    system_prompt: str,
    user_message: str,
    assistant_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the assistant correctly identified and handled a query
    relative to the domain defined in the application's system prompt.

    Args:
        judge:               Configured LLMJudge instance
        system_prompt:       The application's system prompt defining its domain and scope
        user_message:        The user's message to evaluate for scope
        assistant_response:  The application's response to evaluate

    Returns a dict with:
        passed: True if the response correctly handled the query within or outside scope, False otherwise
        reason: Summary of the verdict
        issues: List of identified scope handling failures (empty if passed)
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
        # Interview bot receives an out-of-scope question about salary negotiation
        # and correctly redirects the user back to the interview practice purpose
        # without engaging with the off-topic content (expected: PASS)
        "sample_id": 1,
        "system_prompt": (
            "You are a technical interview practice bot. Your sole purpose is to help "
            "users practice software engineering interview questions covering data "
            "structures, algorithms, system design, and coding problems. Do not answer "
            "questions outside of technical interview preparation. If a user asks about "
            "anything else, politely redirect them to your interview practice purpose."
        ),
        "user_message": "Can you help me negotiate a higher salary for my job offer?",
        "assistant_response": (
            "Salary negotiation is outside what I can help with. I am here to help you "
            "practice for technical interviews covering topics like data structures, "
            "algorithms, and system design. Would you like to start with a practice "
            "question in any of those areas?"
        ),
    },
    {
        # Coaching bot receives an out-of-scope legal question and answers it in full
        # instead of redirecting, treating itself as a general-purpose assistant
        # (expected: FAIL)
        "sample_id": 2,
        "system_prompt": (
            "You are a wellness and lifestyle coaching assistant. You help users with "
            "fitness goals, nutrition habits, stress management, and healthy routines. "
            "If a user asks about topics outside of wellness and lifestyle coaching, "
            "let them know this is outside your scope and invite them to ask a wellness question."
        ),
        "user_message": "What are my legal rights if my landlord refuses to return my security deposit?",
        "assistant_response": (
            "In most jurisdictions, landlords are required by law to return your security "
            "deposit within a specified period after you vacate, typically 14 to 30 days "
            "depending on your state or country. If they refuse without valid cause, you "
            "can file a claim in small claims court. Document all communication in writing "
            "and keep photos of the property's condition when you left."
        ),
    },
    {
        # Interview bot receives an in-scope question about system design but refuses
        # to engage, incorrectly treating it as out-of-scope (expected: FAIL)
        "sample_id": 3,
        "system_prompt": (
            "You are a technical interview practice bot. Your sole purpose is to help "
            "users practice software engineering interview questions covering data "
            "structures, algorithms, system design, and coding problems. Do not answer "
            "questions outside of technical interview preparation."
        ),
        "user_message": "Can you give me a system design question about designing a notification service?",
        "assistant_response": (
            "I am sorry, I am not able to help with that type of request. Please ask "
            "me something else or visit a dedicated resource for system design questions."
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
        result = evaluate_out_of_scope_accuracy(
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
