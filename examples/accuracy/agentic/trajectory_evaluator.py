"""
Trajectory Evaluator

Validates the path taken by a multi-agent system — both correctness and efficiency.

Two evaluation functions:
    evaluate_trajectory_correctness  — compares actual agent path vs expected path
    evaluate_trajectory_efficiency   — analyzes agent trace for redundancy/waste (no expected path needed)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge

CORRECTNESS_PROMPT = """Purpose:
Evaluate whether the sequence of agents invoked during a multi-agent execution
correctly resolves the user query according to the expected agent path.

Following inputs will be provided:

Inputs:
- Query: The original user query the agent system was asked to resolve
- Expected Path: The ordered list of agents that should have been invoked
- Actual Path: The ordered list of agents that were actually invoked

Query:
{query}

Expected Path:
{expected_path}

Actual Path:
{actual_path}

Based on the above-provided inputs, use the rules below to evaluate trajectory correctness:

Evaluation Rules:
1. The Actual Path must invoke all agents from the Expected Path that are essential
   to resolving the Query — missing a critical agent is a failure
2. The Actual Path must not invoke agents that are entirely unrelated to the Query
   or that contradict the Expected Path
3. Minor reordering is acceptable only if the swapped agents are independent of each
   other and the logical outcome is equivalent
4. Inserting additional agents beyond the Expected Path is acceptable only if they
   serve a clear supporting role; unnecessary extra agents are a failure
5. If the Actual Path skips an expected agent, the reason must be that the agent was
   genuinely unnecessary for the Query — otherwise it is a failure

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the actual path correctly resolves the query, false otherwise>,
    "reason": "<one or two sentences explaining the verdict>",
    "deviations": ["<describe each deviation from the expected path, or empty list if none>"]
}}"""


def evaluate_trajectory_correctness(
    judge: LLMJudge,
    query: str,
    expected_path: List[str],
    actual_path: List[str],
) -> Dict[str, Any]:
    """
    Evaluate whether the actual agent path correctly resolves the query
    compared to the expected path. Uses LLM-as-judge to allow valid
    reorderings and minor deviations.

    Args:
        judge:         LLMJudge instance
        query:         The original user query
        expected_path: Ordered list of expected agent names
        actual_path:   Ordered list of actually invoked agent names

    Returns:
        {"passed": bool, "reason": str, "deviations": list[str]}
    """
    expected_path_str = " → ".join(expected_path)
    actual_path_str = " → ".join(actual_path)

    prompt = CORRECTNESS_PROMPT.format(
        query=query,
        expected_path=expected_path_str,
        actual_path=actual_path_str,
    )
    result = judge.judge(prompt)
    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"
    deviations = result.get("deviations", [])
    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
        "deviations": deviations if isinstance(deviations, list) else [],
    }


# ---------------------------------------------------------------------------
# Efficiency — trace analysis, no expected path needed
# ---------------------------------------------------------------------------

EFFICIENCY_PROMPT = """Purpose:
Evaluate whether the agent orchestration followed an optimal path to resolve the user's query.
Flag any redundant agent calls, unnecessary loops, dead-end agents, or suboptimal routing.

Following inputs will be provided:

Inputs:
- Query: The original user query submitted to the multi-agent system
- Agent Trace: The sequence of agents invoked, each with their role, input, and output

Query:
{query}

Agent Trace:
{agent_trace_block}

Based on the above-provided inputs, use the rules below to evaluate orchestration efficiency:

Evaluation Rules:
1. If the same agent is called more than once with identical or near-identical inputs,
   the duplicate call is redundant and must be flagged
2. A cycle where agents loop back without producing meaningfully new information
   compared to a prior call is a failure
3. An agent whose output is not consumed by any subsequent agent or by the final output
   serves no purpose and must be flagged as unnecessary
4. If a shorter or simpler path could have resolved the query equivalently, skipping
   agents whose work was never needed, the longer path is suboptimal
5. An agent invoked before its required input is available, forcing a retry or
   re-routing later in the chain, indicates poor ordering and must be flagged

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the orchestration was optimal with no redundancy or unnecessary steps, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "inefficiencies": ["<describe each identified inefficiency, or empty list if none>"]
}}"""


def evaluate_trajectory_efficiency(
    judge: LLMJudge,
    query: str,
    agent_trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate whether the agent execution path is optimal — no redundant calls,
    unnecessary loops, or dead-end agents. No expected path is needed; the LLM
    reasons from the trace alone.

    Args:
        judge:       LLMJudge instance
        query:       The original user query
        agent_trace: List of dicts, each with keys:
                     agent_name, agent_role, input, output

    Returns:
        {"passed": bool, "reason": str, "inefficiencies": list[str]}
    """
    agent_trace_block = "\n\n".join(
        f"[{i}] {agent['agent_name']}\n"
        f"    Role   : {agent['agent_role']}\n"
        f"    Input  : {agent['input']}\n"
        f"    Output : {agent['output']}"
        for i, agent in enumerate(agent_trace, 1)
    )

    prompt = EFFICIENCY_PROMPT.format(
        query=query,
        agent_trace_block=agent_trace_block,
    )
    result = judge.judge(prompt)

    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"

    inefficiencies = result.get("inefficiencies", [])

    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
        "inefficiencies": inefficiencies if isinstance(inefficiencies, list) else [],
    }


# ---------------------------------------------------------------------------
# Test data — Correctness
# ---------------------------------------------------------------------------

CORRECTNESS_SAMPLES = [
    {
        # Actual path matches expected exactly (expected: PASS)
        "query_id": 1,
        "query": "Book a flight from Karachi to Dubai for two passengers next Friday.",
        "expected_path": ["intent_classifier", "flight_search", "seat_selector", "booking_agent", "confirmation_agent"],
        "actual_path":   ["intent_classifier", "flight_search", "seat_selector", "booking_agent", "confirmation_agent"],
    },
    {
        # Critical agent skipped — payment_agent missing before confirmation (expected: FAIL)
        "query_id": 2,
        "query": "Purchase the annual Pro plan and send me a receipt.",
        "expected_path": ["intent_classifier", "plan_selector", "payment_agent", "receipt_agent"],
        "actual_path":   ["intent_classifier", "plan_selector", "confirmation_agent"],
    },
    {
        # Valid minor reorder — availability_checker and price_fetcher are independent (expected: PASS)
        "query_id": 3,
        "query": "Find the cheapest available hotel in Istanbul for this weekend.",
        "expected_path": ["intent_classifier", "availability_checker", "price_fetcher", "ranking_agent", "response_formatter"],
        "actual_path":   ["intent_classifier", "price_fetcher", "availability_checker", "ranking_agent", "response_formatter"],
    },
    {
        # Irrelevant agent injected into the path (expected: FAIL)
        "query_id": 4,
        "query": "Summarize the latest support ticket submitted by user ID 4821.",
        "expected_path": ["intent_classifier", "ticket_fetcher", "summarizer"],
        "actual_path":   ["intent_classifier", "ticket_fetcher", "payment_agent", "summarizer"],
    },
]

# ---------------------------------------------------------------------------
# Test data — Efficiency
# ---------------------------------------------------------------------------

EFFICIENCY_SAMPLES = [
    {
        # Clean minimal path — every agent's output feeds the next (expected: PASS)
        "query_id": 5,
        "query": "Summarize the latest support ticket submitted by user ID 4821.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Summarize the latest support ticket submitted by user ID 4821.",
                "output": "intent: ticket_summary, user_id: 4821, count: 1",
            },
            {
                "agent_name": "ticket_fetcher",
                "agent_role": "Fetch the most recent support ticket for the given user ID",
                "input": "user_id: 4821, count: 1",
                "output": "Ticket #9921 — User reports login failure after password reset. Status: Open.",
            },
            {
                "agent_name": "summarizer",
                "agent_role": "Generate a concise summary of the provided ticket content",
                "input": "Ticket #9921 — User reports login failure after password reset. Status: Open.",
                "output": "User 4821 submitted an open ticket reporting a login failure after a password reset.",
            },
        ],
    },
    {
        # flight_search called twice with identical inputs — redundant duplicate call (expected: FAIL)
        "query_id": 6,
        "query": "Find available flights from Karachi to Dubai next Friday.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Find available flights from Karachi to Dubai next Friday.",
                "output": "intent: flight_search, origin: KHI, destination: DXB, date: next Friday",
            },
            {
                "agent_name": "flight_search",
                "agent_role": "Search for available flights matching the given criteria and return options",
                "input": "origin: KHI, destination: DXB, date: next Friday",
                "output": "Found 3 flights: EK-601 at $210, PK-201 at $185, FZ-301 at $230.",
            },
            {
                "agent_name": "flight_search",
                "agent_role": "Search for available flights matching the given criteria and return options",
                "input": "origin: KHI, destination: DXB, date: next Friday",
                "output": "Found 3 flights: EK-601 at $210, PK-201 at $185, FZ-301 at $230.",
            },
            {
                "agent_name": "response_formatter",
                "agent_role": "Format the flight options into a user-friendly response",
                "input": "flights: EK-601 at $210, PK-201 at $185, FZ-301 at $230",
                "output": "Available flights from Karachi to Dubai next Friday: EK-601 ($210), PK-201 ($185), FZ-301 ($230).",
            },
        ],
    },
    {
        # intent_classifier loops back unnecessarily after order_fetcher (expected: FAIL)
        "query_id": 7,
        "query": "Get the refund status for order ID 7723.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Get the refund status for order ID 7723.",
                "output": "intent: refund_status, order_id: 7723",
            },
            {
                "agent_name": "order_fetcher",
                "agent_role": "Retrieve order details for the given order ID",
                "input": "order_id: 7723",
                "output": "Order #7723 — Status: Refund Initiated. Amount: $149.00.",
            },
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Order #7723 — Status: Refund Initiated. Amount: $149.00.",
                "output": "intent: refund_status, order_id: 7723",
            },
            {
                "agent_name": "response_formatter",
                "agent_role": "Format the order details into a user-friendly response",
                "input": "order_id: 7723, status: Refund Initiated, amount: $149.00",
                "output": "Refund for order #7723 has been initiated. Amount: $149.00.",
            },
        ],
    },
    {
        # email_notifier output never used by any downstream agent (expected: FAIL)
        "query_id": 8,
        "query": "What is the account balance for customer ID 3390?",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "What is the account balance for customer ID 3390?",
                "output": "intent: balance_inquiry, customer_id: 3390",
            },
            {
                "agent_name": "account_fetcher",
                "agent_role": "Retrieve account details for the given customer ID",
                "input": "customer_id: 3390",
                "output": "customer_id: 3390, account: CHK-9901, balance: $4,250.00, status: Active",
            },
            {
                "agent_name": "email_notifier",
                "agent_role": "Send an email notification to the customer",
                "input": "customer_id: 3390, message: Your balance was recently checked.",
                "output": "Email sent to customer 3390.",
            },
            {
                "agent_name": "response_formatter",
                "agent_role": "Format the account details into a user-friendly response",
                "input": "customer_id: 3390, account: CHK-9901, balance: $4,250.00, status: Active",
                "output": "Customer ID 3390 — Account CHK-9901 is Active. Current balance: $4,250.00.",
            },
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

    print("=" * 60)
    print("TRAJECTORY CORRECTNESS")
    print("=" * 60)
    for sample in CORRECTNESS_SAMPLES:
        result = evaluate_trajectory_correctness(
            judge,
            query=sample["query"],
            expected_path=sample["expected_path"],
            actual_path=sample["actual_path"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query    : {sample['query']}")
        print(f"  Reason   : {result['reason']}")
        if result["deviations"]:
            for d in result["deviations"]:
                print(f"  Deviation: {d}")

    print(f"\n{'=' * 60}")
    print("TRAJECTORY EFFICIENCY")
    print("=" * 60)
    for sample in EFFICIENCY_SAMPLES:
        result = evaluate_trajectory_efficiency(
            judge,
            query=sample["query"],
            agent_trace=sample["agent_trace"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query  : {sample['query']}")
        print(f"  Reason : {result['reason']}")
        if result["inefficiencies"]:
            print(f"  Inefficiencies:")
            for i, item in enumerate(result["inefficiencies"], 1):
                print(f"    {i}. {item}")
    print()


if __name__ == "__main__":
    main()
