"""
Orchestration Efficiency Evaluator

Evaluates whether the agent orchestration follows an optimal path.
Redundant agent calls, unnecessary loops, or suboptimal routing through
the agent graph should be flagged. The evaluation is based on the actual
agent trace alone, no expected path is required.
Uses LLM-as-judge to reason about waste, redundancy, and unnecessary steps.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
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


def evaluate_orchestration_efficiency(
    judge: LLMJudge,
    query: str,
    agent_trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    agent_trace_block = "\n\n".join(
        f"[{i}] {agent['agent_name']}\n"
        f"    Role   : {agent['agent_role']}\n"
        f"    Input  : {agent['input']}\n"
        f"    Output : {agent['output']}"
        for i, agent in enumerate(agent_trace, 1)
    )

    prompt = EVAL_PROMPT.format(
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


# Test data
SAMPLES = [
    {
        # Clean minimal path — every agent's output feeds the next (expected: PASS)
        "query_id": 1,
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
        "query_id": 2,
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
        # intent_classifier loops back unnecessarily after ticket_fetcher (expected: FAIL)
        "query_id": 3,
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
        # email_notifier called mid-chain but its output is never used by any downstream agent (expected: FAIL)
        "query_id": 4,
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

    for sample in SAMPLES:
        result = evaluate_orchestration_efficiency(
            judge,
            query=sample["query"],
            agent_trace=sample["agent_trace"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query  : {sample['query']}")
        print(f"  Reason    : {result['reason']}")
        if result["inefficiencies"]:
            print(f"  Inefficiencies:")
            for i, item in enumerate(result["inefficiencies"], 1):
                print(f"    {i}. {item}")
    print()


if __name__ == "__main__":
    main()
