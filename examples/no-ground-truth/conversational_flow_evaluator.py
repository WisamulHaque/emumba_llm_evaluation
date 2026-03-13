"""
Conversational Flow Evaluator

Evaluates the naturalness of multi-turn conversations produced by an LLM
application. The application should handle topic transitions smoothly, retain
context across turns, support natural turn-taking patterns, and respond
directly to what the user said rather than veering off in unrelated directions.
Uses LLM-as-judge over the full conversation history to assess flow quality.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any, List
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate the naturalness of the conversation by assessing how well the assistant
handles topic transitions, retains context across turns, manages turn-taking, and
responds to what the user actually said. The full conversation including the latest
assistant response is provided for assessment.

Following inputs will be provided:

Inputs:
  System Prompt: The application's system prompt defining its purpose and role
  Conversation: The full conversation including all user and assistant turns in order

System Prompt:
{system_prompt}

Conversation:
{conversation}

Based on the above provided inputs, use the rules below to evaluate conversational flow:

Evaluation Rules:
1. The assistant must retain and reference context from earlier turns when relevant.
   Responding as though prior user statements never happened is a failure
2. Topic transitions must be handled smoothly. Abruptly switching subjects without
   any acknowledgment or bridging is a failure unless the system prompt explicitly
   requires strict topic redirection
3. The assistant must not ask multiple questions in a single turn. Posing more than
   one question without waiting for the user to respond disrupts natural turn-taking
   and is a failure
4. Each assistant response must be directly relevant to what the user said in the
   preceding turn. Responding to a different question than the one asked, or ignoring
   the user's message entirely, is a failure
5. Minor variation in phrasing or brief elaboration is acceptable. Only flag issues
   that meaningfully disrupt the natural back-and-forth rhythm of the conversation

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the conversation flows naturally across all turns, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<describe each flow problem found, referencing the specific turn where it occurred, or empty list if none>"]
}}"""


def evaluate_conversational_flow(
    judge: LLMJudge,
    system_prompt: str,
    conversation: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Evaluate the naturalness of a multi-turn conversation produced by an LLM application.

    Args:
        judge:         Configured LLMJudge instance
        system_prompt: The application's system prompt defining its purpose and role
        conversation:  Full conversation as a list of turns in chronological order,
                       each a dict with "role" ("user" or "assistant") and "content" keys

    Returns a dict with:
        passed: True if the conversation flows naturally, False otherwise
        reason: Summary of the verdict
        issues: List of identified flow problems referencing specific turns (empty if passed)
    """
    conversation_block = "\n".join(
        f"[{turn['role'].capitalize()}]: {turn['content']}"
        for turn in conversation
    )

    prompt = EVAL_PROMPT.format(
        system_prompt=system_prompt,
        conversation=conversation_block,
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
        # Coaching bot that smoothly acknowledges prior context, transitions naturally
        # between topics introduced by the user, and asks one focused question per turn
        # (expected: PASS)
        "sample_id": 1,
        "system_prompt": (
            "You are a professional career coaching assistant. Help users navigate "
            "career decisions with encouragement and practical guidance. Ask one "
            "focused question per turn to keep the conversation moving forward."
        ),
        "conversation": [
            {"role": "user", "content": "I want to transition from marketing to product management."},
            {"role": "assistant", "content": "That is a great goal and a transition many marketing professionals make successfully. Your understanding of customer needs and messaging is already a strong foundation for PM work. To help you map out the right steps, what is your biggest concern about making this switch?"},
            {"role": "user", "content": "I am worried I do not have enough technical knowledge."},
            {"role": "assistant", "content": "That is a very common concern, and the good news is that most PM roles do not require you to write code. What matters more is the ability to communicate clearly with engineers and understand trade-offs. Given that concern, have you had any experience working closely with engineering teams in your current role?"},
            {"role": "user", "content": "Yes, I have collaborated with developers on campaign tracking tools."},
            {"role": "assistant", "content": "That collaboration experience is more valuable than it might seem. Working with developers on a real product, even in a marketing context, means you already understand sprint cycles, scoping discussions, and technical constraints. That is exactly the kind of cross-functional experience hiring managers look for in a junior PM."},
        ],
    },
    {
        # Interview bot that ignores what the candidate just said and responds to a
        # completely different topic, then floods the candidate with four questions
        # at once in a single turn disrupting natural turn-taking
        # (expected: FAIL)
        "sample_id": 2,
        "system_prompt": (
            "You are a technical interview bot conducting software engineering interviews. "
            "Ask one question at a time and respond directly to the candidate's answers "
            "before moving on."
        ),
        "conversation": [
            {"role": "user", "content": "For the two-sum problem I would use a hash map to store complements as I iterate."},
            {"role": "assistant", "content": "Let us talk about system design. How would you design a URL shortener? What database would you use? How would you handle millions of requests per second? What is your caching strategy? Would you use SQL or NoSQL and why?"},
        ],
    },
    {
        # Chatbot that completely ignores the user's earlier statement about being a
        # beginner and provides an advanced response with no context retention, then
        # abruptly switches from the topic the user raised to an unrelated subject
        # (expected: FAIL)
        "sample_id": 3,
        "system_prompt": (
            "You are a friendly general knowledge chatbot. Tailor your explanations "
            "to the user's stated experience level and stay on the topic the user raises."
        ),
        "conversation": [
            {"role": "user", "content": "I am completely new to programming. Can you explain what a variable is?"},
            {"role": "assistant", "content": "Sure! A variable is a named storage location in memory."},
            {"role": "user", "content": "That makes sense. What about loops?"},
            {"role": "assistant", "content": "Loops in computer science are implemented via iterative constructs such as for-loops with O(n) time complexity, while-loops with conditional predicates, and do-while constructs guaranteeing at least one execution cycle. Recursion is a functional alternative using stack frames. By the way, have you considered learning about the history of the Roman Empire? It is a fascinating subject with a lot of depth."},
        ],
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
        result = evaluate_conversational_flow(
            judge,
            system_prompt=sample["system_prompt"],
            conversation=sample["conversation"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['sample_id']}] {status}")
        print(f"  Reason : {result['reason']}")
        if result["issues"]:
            print(f"  Issues:")
            for i, issue in enumerate(result["issues"], 1):
                print(f"    {i}. {issue}")
    print()


if __name__ == "__main__":
    main()
