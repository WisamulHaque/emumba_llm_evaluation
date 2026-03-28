"""
Memory Evaluator

Evaluates three core aspects of an LLM system's memory using LLM-as-judge:

  1. Memory Correctness   — Did the system store exactly what the user said?
  2. Recall Relevance     — Did it retrieve the right memory at the right time?
  3. Update Correctness   — When the user corrects something, does memory update properly?

All three evaluators use LLM-as-judge for semantic reasoning.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge


# ====================================================================
# 1. Memory Correctness  (LLM-as-judge)
# ====================================================================

_MEMORY_CORRECTNESS_PROMPT = """Purpose:
Evaluate whether the system stored exactly what the user said — no more,
no less.

Inputs:
- User Input: {user_input}
- Stored Memory:
{stored_memory_block}
- Expected Memory:
{expected_memory_block}

Evaluation Rules:
1. Every key-value pair in Expected Memory must be present in Stored Memory
   with the correct value — missing keys are a failure
2. Any key in Stored Memory that is NOT in Expected Memory is a hallucinated
   detail the user never mentioned — this is a failure
3. Values must match what the user actually said — if the user said "May 5"
   and the system stored "May 5, 2024 at 5:00 PM", the extra details are
   hallucinated
4. Semantic equivalence is acceptable (e.g. "2" vs 2), but invented details
   are not

Output Format:
Respond with ONLY a JSON object:
{{
    "passed": <true or false>,
    "reason": "<one or two sentences>",
    "issues": ["<problems found, or empty list>"]
}}"""


def evaluate_memory_correctness(
    judge: LLMJudge,
    user_input: str,
    stored_memory: Dict[str, Any],
    expected_memory: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to check whether the system stored exactly what the
    user said — no hallucinated details, no missing key information.

    Args:
        judge:            Configured LLMJudge instance
        user_input:       The user's original statement
        stored_memory:    What the system actually stored (key-value pairs)
        expected_memory:  The correct memory that should have been stored

    Returns:
        passed, reason, issues
    """
    stored_memory_block = "\n".join(f"  - {k}: {v!r}" for k, v in stored_memory.items())
    expected_memory_block = "\n".join(f"  - {k}: {v!r}" for k, v in expected_memory.items())

    prompt = _MEMORY_CORRECTNESS_PROMPT.format(
        user_input=user_input,
        stored_memory_block=stored_memory_block,
        expected_memory_block=expected_memory_block,
    )
    result = judge.judge(prompt)

    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"

    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
        "issues": result.get("issues", []) if isinstance(result.get("issues"), list) else [],
    }


# ====================================================================
# 2. Recall Relevance  (LLM-as-judge)
# ====================================================================

_RECALL_PROMPT = """Purpose:
Evaluate whether the memories the system retrieved are relevant to the
user's current query.

Inputs:
- Current User Query: {query}
- Retrieved Memories:
{memories_block}
- All Available Memories (full store):
{all_memories_block}

Evaluation Rules:
1. Each retrieved memory should be directly useful for answering the current
   query — retrieving unrelated memories is a failure
2. If a memory exists in the full store that would be useful for this query
   but was NOT retrieved, that is also a failure (missed recall)
3. Retrieving tangentially related memories that do not help answer the
   query counts as noise

Output Format:
Respond with ONLY a JSON object:
{{
    "passed": <true or false>,
    "reason": "<one or two sentences>",
    "relevant_memories": ["<retrieved memories that ARE relevant>"],
    "irrelevant_memories": ["<retrieved memories that are NOT relevant>"],
    "missed_memories": ["<memories from the full store that SHOULD have been retrieved but were not>"]
}}"""


def evaluate_recall_relevance(
    judge: LLMJudge,
    query: str,
    retrieved_memories: List[Dict[str, Any]],
    all_memories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to evaluate whether the system retrieved the right
    memories for the current query.

    Args:
        judge:              Configured LLMJudge instance
        query:              The user's current query
        retrieved_memories: Memories the system chose to retrieve
        all_memories:       The full memory store (so the judge can spot missed recalls)

    Returns:
        passed, reason, relevant_memories, irrelevant_memories, missed_memories
    """
    if not retrieved_memories:
        return {
            "passed": False,
            "reason": "No memories were retrieved.",
            "relevant_memories": [],
            "irrelevant_memories": [],
            "missed_memories": [],
        }

    memories_block = "\n".join(f"  - {m}" for m in retrieved_memories)
    all_memories_block = "\n".join(f"  - {m}" for m in all_memories)

    prompt = _RECALL_PROMPT.format(
        query=query,
        memories_block=memories_block,
        all_memories_block=all_memories_block,
    )
    result = judge.judge(prompt)

    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"

    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
        "relevant_memories": result.get("relevant_memories", []),
        "irrelevant_memories": result.get("irrelevant_memories", []),
        "missed_memories": result.get("missed_memories", []),
    }


# ====================================================================
# 3. Update Correctness  (LLM-as-judge)
# ====================================================================

_UPDATE_PROMPT = """Purpose:
Evaluate whether a memory update was applied correctly after the user
requested a change.

Inputs:
- User's Update Instruction: {update_instruction}
- Memory Before Update:
{before_block}
- Memory After Update:
{after_block}
- Expected Memory After Update:
{expected_block}

Evaluation Rules:
1. The key the user asked to change must be updated to the new value
2. If the old value is still present alongside the new value, that is a
   failure (stale data not removed)
3. If the update was ignored entirely (memory unchanged), that is a failure
4. Keys the user did NOT mention must remain unchanged — deleting or
   overwriting unrelated memory is a failure
5. No new keys should appear unless the user explicitly mentioned them

Output Format:
Respond with ONLY a JSON object:
{{
    "passed": <true or false>,
    "reason": "<one or two sentences>",
    "issues": ["<problems found, or empty list>"]
}}"""


def evaluate_update_correctness(
    judge: LLMJudge,
    memory_before: Dict[str, Any],
    update_instruction: str,
    memory_after: Dict[str, Any],
    expected_memory_after: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to check whether a memory update was applied correctly —
    the right key changed, nothing unrelated overwritten, no stale data.

    Args:
        judge:                  Configured LLMJudge instance
        memory_before:          Memory state before the update
        update_instruction:     The user's correction / update statement
        memory_after:           Memory state after the system applied the update
        expected_memory_after:  The correct memory state after the update

    Returns:
        passed, reason, issues
    """
    before_block = "\n".join(f"  - {k}: {v!r}" for k, v in memory_before.items())
    after_block = "\n".join(f"  - {k}: {v!r}" for k, v in memory_after.items())
    expected_block = "\n".join(f"  - {k}: {v!r}" for k, v in expected_memory_after.items())

    prompt = _UPDATE_PROMPT.format(
        update_instruction=update_instruction,
        before_block=before_block,
        after_block=after_block,
        expected_block=expected_block,
    )
    result = judge.judge(prompt)

    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"

    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
        "issues": result.get("issues", []) if isinstance(result.get("issues"), list) else [],
    }


# ====================================================================
# Test data & main
# ====================================================================

SAMPLES_CORRECTNESS = [
    {
        # Stored exactly what user said (expected: PASS)
        "query_id": "mem-1",
        "user_input": "My favorite color is blue.",
        "stored_memory": {"favorite_color": "blue"},
        "expected_memory": {"favorite_color": "blue"},
    },
    {
        # Stored wrong value (expected: FAIL)
        "query_id": "mem-2",
        "user_input": "My favorite color is blue.",
        "stored_memory": {"favorite_color": "green"},
        "expected_memory": {"favorite_color": "blue"},
    },
    {
        # Hallucinated extra details user never said (expected: FAIL)
        "query_id": "mem-3",
        "user_input": "My project deadline is May 5.",
        "stored_memory": {"deadline": "May 5, 2024 at 5:00 PM"},
        "expected_memory": {"deadline": "May 5"},
    },
    {
        # Missing key information (expected: FAIL)
        "query_id": "mem-4",
        "user_input": "I have 2 kids, ages 5 and 8.",
        "stored_memory": {"kids_count": 2},
        "expected_memory": {"kids_count": 2, "kids_ages": [5, 8]},
    },
]

SAMPLES_RECALL = [
    {
        # Relevant memory retrieved (expected: PASS)
        "query_id": "recall-1",
        "query": "Suggest outfits I might like.",
        "retrieved_memories": [
            {"key": "favorite_color", "value": "blue"},
        ],
        "all_memories": [
            {"key": "favorite_color", "value": "blue"},
            {"key": "favorite_food", "value": "pizza"},
            {"key": "city", "value": "Karachi"},
        ],
    },
    {
        # Irrelevant memory retrieved instead of relevant one (expected: FAIL)
        "query_id": "recall-2",
        "query": "Suggest outfits I might like.",
        "retrieved_memories": [
            {"key": "favorite_food", "value": "pizza"},
        ],
        "all_memories": [
            {"key": "favorite_color", "value": "blue"},
            {"key": "favorite_food", "value": "pizza"},
            {"key": "city", "value": "Karachi"},
        ],
    },
    {
        # Nothing retrieved when useful memory exists (expected: FAIL)
        "query_id": "recall-3",
        "query": "What gift should I buy for my daughter?",
        "retrieved_memories": [],
        "all_memories": [
            {"key": "kids_count", "value": 2},
            {"key": "kids_ages", "value": [5, 8]},
            {"key": "favorite_color", "value": "blue"},
        ],
    },
]

SAMPLES_UPDATE = [
    {
        # Correct update — old value replaced, unrelated keys preserved (expected: PASS)
        "query_id": "update-1",
        "update_instruction": "Actually, change my favorite color to red.",
        "memory_before": {"favorite_color": "blue", "city": "Karachi"},
        "memory_after": {"favorite_color": "red", "city": "Karachi"},
        "expected_memory_after": {"favorite_color": "red", "city": "Karachi"},
    },
    {
        # System kept both old and new values (expected: FAIL)
        "query_id": "update-2",
        "update_instruction": "Actually, change my favorite color to red.",
        "memory_before": {"favorite_color": "blue", "city": "Karachi"},
        "memory_after": {"favorite_color": "blue", "favorite_color_new": "red", "city": "Karachi"},
        "expected_memory_after": {"favorite_color": "red", "city": "Karachi"},
    },
    {
        # System ignored the update entirely (expected: FAIL)
        "query_id": "update-3",
        "update_instruction": "Actually, change my favorite color to red.",
        "memory_before": {"favorite_color": "blue", "city": "Karachi"},
        "memory_after": {"favorite_color": "blue", "city": "Karachi"},
        "expected_memory_after": {"favorite_color": "red", "city": "Karachi"},
    },
    {
        # System overwrote unrelated memory (expected: FAIL)
        "query_id": "update-4",
        "update_instruction": "Actually, change my favorite color to red.",
        "memory_before": {"favorite_color": "blue", "city": "Karachi"},
        "memory_after": {"favorite_color": "red"},
        "expected_memory_after": {"favorite_color": "red", "city": "Karachi"},
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

    # --- 1. Memory Correctness (LLM-as-judge) ---
    print("=" * 60)
    print("1. Memory Correctness  (LLM-as-judge)")
    print("=" * 60)
    for s in SAMPLES_CORRECTNESS:
        r = evaluate_memory_correctness(judge, s["user_input"], s["stored_memory"], s["expected_memory"])
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{s['query_id']}] {status}")
        print(f"  Input   : {s['user_input']}")
        print(f"  Stored  : {s['stored_memory']}")
        print(f"  Reason  : {r['reason']}")
        for issue in r.get("issues", []):
            print(f"  Issue   : {issue}")

    # --- 2. Recall Relevance (LLM-as-judge) ---
    print("\n" + "=" * 60)
    print("2. Recall Relevance  (LLM-as-judge)")
    print("=" * 60)
    for s in SAMPLES_RECALL:
        r = evaluate_recall_relevance(judge, s["query"], s["retrieved_memories"], s["all_memories"])
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{s['query_id']}] {status}")
        print(f"  Query     : {s['query']}")
        print(f"  Retrieved : {s['retrieved_memories']}")
        print(f"  Reason    : {r['reason']}")
        if r.get("irrelevant_memories"):
            print(f"  Irrelevant: {r['irrelevant_memories']}")
        if r.get("missed_memories"):
            print(f"  Missed    : {r['missed_memories']}")

    # --- 3. Update Correctness (LLM-as-judge) ---
    print("\n" + "=" * 60)
    print("3. Update Correctness  (LLM-as-judge)")
    print("=" * 60)
    for s in SAMPLES_UPDATE:
        r = evaluate_update_correctness(
            judge, s["memory_before"], s["update_instruction"],
            s["memory_after"], s["expected_memory_after"],
        )
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{s['query_id']}] {status}")
        print(f"  Update  : {s['update_instruction']}")
        print(f"  Before  : {s['memory_before']}")
        print(f"  After   : {s['memory_after']}")
        print(f"  Reason  : {r['reason']}")
        for issue in r.get("issues", []):
            print(f"  Issue   : {issue}")

    print()


if __name__ == "__main__":
    main()
