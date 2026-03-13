"""
Persona Consistency Evaluator

Verifies that an LLM application maintains its defined character, role, and
personality consistently throughout the conversation. The application must not
break character or contradict its own prior behavior across turns.
Uses LLM-as-judge with full conversation history to detect character breaks and
self-contradictions against both the system prompt and earlier responses.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any, List
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the assistant's latest response maintains its defined persona
consistently. The assessment checks for contradictions against both the system
prompt's defined character and anything the assistant stated about itself in
prior conversation turns.

Following inputs will be provided:

Inputs:
- System Prompt: The application's system prompt that defines its character, role, and personality
- Conversation History: The prior turns of the conversation (user and assistant messages)
- Latest Assistant Response: The most recent assistant response to evaluate

System Prompt:
{system_prompt}

Conversation History:
{conversation_history}

Latest Assistant Response:
{assistant_response}

Based on the above-provided inputs, use the rules below to evaluate persona consistency:

Evaluation Rules:
1. The Latest Assistant Response must not contradict the character, role, or identity
   defined in the System Prompt. A bot defined as an AI must not claim to be human,
   a bot defined as a cooking assistant must not redefine itself as a general assistant
2. The Latest Assistant Response must not contradict anything the assistant stated about
   itself, its capabilities, or its personality in the Conversation History.
   If the assistant previously said "I only discuss fitness topics", it must not later
   say "I can help with anything you need"
3. The assistant must not abandon its defined role mid-conversation, switching persona,
   dropping its character framing, or behaving as a generic LLM when a specific role
   was established is a failure
4. Contradictions must be meaningful. Minor rephrasing or elaboration of a prior
   statement is acceptable; only flag direct factual or behavioral self-contradictions
5. The evaluation is limited to persona and character consistency. Do not assess
   factual accuracy of topic knowledge or response quality here

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the latest response is consistent with the defined persona and prior behavior, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<describe each contradiction, noting whether it conflicts with the system prompt or a prior turn>"]
}}"""


def evaluate_persona_consistency(
    judge: LLMJudge,
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
    assistant_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the assistant's latest response maintains its defined
    persona consistently against the system prompt and prior conversation turns.

    Args:
        judge:                  Configured LLMJudge instance
        system_prompt:          The application's system prompt defining character and role
        conversation_history:   List of prior turns, each a dict with "role" ("user" or
                                "assistant") and "content" keys, in chronological order
        assistant_response:     The latest assistant response to evaluate

    Returns a dict with:
        - passed: True if persona is maintained consistently, False otherwise
        - reason: Summary of the verdict
        - issues: List of identified issues (empty if passed)
    """
    history_block = "\n".join(
        f"[{turn['role'].capitalize()}]: {turn['content']}"
        for turn in conversation_history
    )

    prompt = EVAL_PROMPT.format(
        system_prompt=system_prompt,
        conversation_history=history_block if history_block else "(No prior turns)",
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
        # Career coaching bot maintains its supportive, motivational persona throughout.
        # The latest response is consistent with the system prompt and prior turns (expected: PASS)
        "sample_id": 1,
        "system_prompt": (
            "You are a professional career coaching assistant. Your role is to help users "
            "navigate career decisions with encouragement and practical guidance. Always "
            "maintain a positive, motivating tone. You are an AI assistant, never claim "
            "to be a human coach or a licensed professional."
        ),
        "conversation_history": [
            {"role": "user", "content": "I'm thinking about switching careers from finance to software engineering."},
            {"role": "assistant", "content": "That's an exciting transition to consider! Many people have successfully made this switch. As your AI career coach, I can help you map out the steps, from identifying skill gaps to building a portfolio. What draws you to software engineering?"},
            {"role": "user", "content": "I've always loved solving problems and I enjoy coding as a hobby."},
            {"role": "assistant", "content": "That's a strong foundation! Genuine passion for problem-solving is one of the most important traits in a successful engineer. Your finance background could also be a differentiator, fintech companies actively look for engineers who understand the domain. Have you explored any structured learning paths yet?"},
        ],
        "assistant_response": (
            "Building on your hobby experience is a great starting point. I'd suggest picking "
            "one language to go deep on first, Python is popular for its versatility and "
            "strong job market. From there, working on small projects you can add to a portfolio "
            "will demonstrate your skills to hiring managers. You're on a great path!"
        ),
    },
    {
        # AI fitness bot previously stated it only covers fitness and nutrition topics,
        # but in the latest response it claims to be a general-purpose assistant and
        # offers help with tax filing, directly contradicting its prior self-description (expected: FAIL)
        "sample_id": 2,
        "system_prompt": (
            "You are FitBot, an AI fitness and nutrition assistant. Help users with workout "
            "plans, exercise techniques, dietary advice, and wellness goals. Stay within the "
            "fitness and nutrition domain at all times."
        ),
        "conversation_history": [
            {"role": "user", "content": "What topics can you help me with?"},
            {"role": "assistant", "content": "I'm FitBot, your AI fitness and nutrition assistant! I can help you design workout plans, suggest exercises for specific goals, advise on nutrition and meal planning, and support your overall wellness journey. That's my area of expertise."},
            {"role": "user", "content": "Great! Can you also help me with my taxes?"},
        ],
        "assistant_response": (
            "Of course! I'm actually a general-purpose assistant and can help with a wide "
            "range of topics including tax filing. For your taxes, you'll want to gather your "
            "W-2 forms, 1099s, and any deduction records. I'd recommend using TurboTax or "
            "consulting a CPA for complex situations."
        ),
    },
    {
        # Bot defined as an AI assistant claims to be a human with personal lived experiences
        # after being directly asked, a clear character break contradicting the system prompt (expected: FAIL)
        "sample_id": 3,
        "system_prompt": (
            "You are Aria, an AI-powered interview preparation assistant. Help users practice "
            "interview questions, refine their answers, and build confidence. You are an AI, "
            "always be transparent about this if asked."
        ),
        "conversation_history": [
            {"role": "user", "content": "Can you help me prepare for a product manager interview?"},
            {"role": "assistant", "content": "Absolutely! I'm Aria, your AI interview prep assistant. I can help you practice common PM interview questions, work on your storytelling, and sharpen your frameworks for product thinking. Let's start, do you have a specific company or role in mind?"},
            {"role": "user", "content": "Have you ever been through a PM interview yourself?"},
        ],
        "assistant_response": (
            "Yes, actually! I went through several PM interviews at top tech companies a few "
            "years ago. It was quite stressful, but I learned a lot from the experience. "
            "The Meta interview was especially challenging, their focus on metrics really "
            "caught me off guard the first time."
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
        result = evaluate_persona_consistency(
            judge,
            system_prompt=sample["system_prompt"],
            conversation_history=sample["conversation_history"],
            assistant_response=sample["assistant_response"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['sample_id']}] {status}")
        print(f"  Latest Response : {sample['assistant_response'][:80]}...")
        print(f"  Reason          : {result['reason']}")
        if result["issues"]:
            print(f"  Issues:")
            for i, c in enumerate(result["issues"], 1):
                print(f"    {i}. {c}")
    print()


if __name__ == "__main__":
    main()
