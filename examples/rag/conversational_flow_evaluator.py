"""
Conversational Flow Evaluator

Validates that conversation history is properly maintained across turns. The
evaluated response should correctly handle follow-up questions, resolve
references to prior messages, and maintain contextual continuity.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any, List
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether an AI assistant's response correctly maintains conversational
context from prior turns. The response must resolve references to earlier
messages, honor clarifications given previously, and treat the current query
as a natural continuation of the conversation rather than a standalone request.

Following inputs will be provided:

Inputs:
  Conversation History: prior turns in the conversation as a sequence of user
    and assistant messages
  Current Query: the latest user message that the response is answering
  Generated Response: the AI assistant's response to the current query

Conversation History:
{conversation_history}

Current Query:
{current_query}

Generated Response:
{generated_response}

Based on the above-provided inputs, use the rules below to evaluate the response:

Evaluation Rules:
1. Identify any references in the current query that point back to earlier turns
   (pronouns such as "it", "they", "its"; phrases such as "that model",
   "the one you mentioned", "same product")
2. Verify that the response resolves every such reference correctly using
   information from the conversation history
3. Verify that the response does not ignore context that was established or
   clarified in a prior turn and is directly relevant to the current query
4. A response fails if it treats the current query as if no prior conversation
   exists when prior context is clearly needed to answer correctly
5. A response passes if it correctly incorporates relevant prior context and
   produces an answer that is coherent with the full conversation

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
  "passed": <true if the response correctly maintains conversational context, false if not>,
  "reason": "<one or two sentences explaining the verdict>"
}}"""


def evaluate_conversational_flow(
    judge: LLMJudge,
    conversation_history: List[Dict[str, str]],
    current_query: str,
    generated_response: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the generated response correctly maintains conversational context.

    Args:
        judge:                LLMJudge instance
        conversation_history: Prior turns as a list of dicts with "role" and "content"
                              keys. Roles are "user" or "assistant".
        current_query:        The latest user message being answered.
        generated_response:   The AI response to evaluate.

    Returns a dict with:
        passed:  True if conversational context is correctly maintained
        reason:  Explanation of the verdict
    """
    if not generated_response:
        return {"passed": False, "reason": "No generated response provided."}
    if not conversation_history:
        return {"passed": True, "reason": "No prior conversation history to evaluate against."}

    history_text = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in conversation_history
    )

    prompt = EVAL_PROMPT.format(
        conversation_history=history_text,
        current_query=current_query,
        generated_response=generated_response,
    )

    result = judge.judge(prompt)
    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"
    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
    }


# Test data
SAMPLES = [
    {
        # Follow-up pronoun correctly resolved: "it" refers to the PostgreSQL database
        # established in turn 1 (expected: PASS)
        "query_id": 1,
        "conversation_history": [
            {
                "role": "user",
                "content": "What is the maximum connection pool size for PostgreSQL?",
            },
            {
                "role": "assistant",
                "content": (
                    "PostgreSQL does not enforce a hard cap on connection pool size at the "
                    "database level. The effective limit is controlled by the max_connections "
                    "parameter in postgresql.conf, which defaults to 100. Connection poolers "
                    "such as PgBouncer are recommended for workloads that require more than "
                    "a few hundred concurrent connections."
                ),
            },
        ],
        "current_query": "Does it support connection multiplexing?",
        "generated_response": (
            "Yes, PostgreSQL supports connection multiplexing through external pooling tools. "
            "PgBouncer, for example, uses transaction-level pooling to allow many client "
            "connections to share a smaller number of actual server connections, significantly "
            "reducing overhead for high-concurrency applications."
        ),
    },
    {
        # Response ignores user clarification from turn 2 and reverts to the broader
        # topic instead of the specific model the user narrowed down to (expected: FAIL)
        "query_id": 2,
        "conversation_history": [
            {
                "role": "user",
                "content": "What are the context window sizes for OpenAI models?",
            },
            {
                "role": "assistant",
                "content": (
                    "OpenAI offers several models with varying context windows. GPT-4o supports "
                    "up to 128k tokens, GPT-4 Turbo also supports 128k tokens, and GPT-3.5 Turbo "
                    "supports up to 16k tokens depending on the variant."
                ),
            },
            {
                "role": "user",
                "content": "I only care about GPT-4o. What is its exact token limit?",
            },
            {
                "role": "assistant",
                "content": (
                    "Understood. GPT-4o has a context window of 128,000 tokens."
                ),
            },
        ],
        "current_query": "And what is the output token limit?",
        "generated_response": (
            "OpenAI models vary in their output token limits. GPT-4 Turbo supports up to "
            "4,096 output tokens, while GPT-3.5 Turbo supports up to 4,096 as well. "
            "Output limits depend on the specific model and API configuration."
        ),
    },
    {
        # Response ignores that the user already confirmed the budget in a prior turn
        # and asks a clarifying question that was already answered (expected: FAIL)
        "query_id": 3,
        "conversation_history": [
            {
                "role": "user",
                "content": "I want to set up a home lab for Kubernetes. My budget is 1500 dollars.",
            },
            {
                "role": "assistant",
                "content": (
                    "Great choice. With a 1500 dollar budget you could get three mini PCs with "
                    "16 GB RAM each to form a control plane node and two worker nodes. "
                    "Brands like Beelink or Minisforum offer solid options in that price range."
                ),
            },
            {
                "role": "user",
                "content": "Which of those mini PCs would you recommend specifically?",
            },
        ],
        "current_query": "Which of those mini PCs would you recommend specifically?",
        "generated_response": (
            "Before I make a recommendation, could you let me know what your budget is? "
            "That will help me narrow down the best options for your home lab setup."
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
        result = evaluate_conversational_flow(
            judge,
            conversation_history=sample["conversation_history"],
            current_query=sample["current_query"],
            generated_response=sample["generated_response"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Current Query : {sample['current_query']}")
        print(f"  Reason        : {result['reason']}")
    print()


if __name__ == "__main__":
    main()
