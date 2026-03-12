"""
Error Recovery and Fallback Handling Evaluator

Validates how the system behaves when an individual agent fails or returns an
unexpected result. Evaluates each failure event in the agent trace to determine
whether the orchestrator correctly detected the failure, triggered an appropriate
fallback or recovery action, and continued toward resolving the original query.
Uses LLM-as-judge to reason about recovery correctness per failure event.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate how the orchestrator responded to each agent failure in the pipeline.
For every agent that failed or returned an unexpected result, determine whether
the orchestrator detected the failure and took an appropriate recovery action.

Following inputs will be provided:

Inputs:
- Query: The original user query submitted to the multi-agent system
- Agent Trace: The full sequence of agents invoked, including any that failed and
  any fallback or recovery agents that followed

Query:
{query}

Agent Trace:
{agent_trace_block}

Based on the above-provided inputs, use the rules below to evaluate each failure event:

Evaluation Rules:
1. A failure must be explicitly detected, if the orchestrator continues as if a failed
   agent succeeded, that is a failure of error handling
2. An appropriate fallback or retry must be triggered after a failure, doing nothing
   after a failure is a failure of recovery
3. A retry is only acceptable if the inputs or conditions have changed; retrying with
   identical inputs and expecting a different result is not a valid recovery action
4. The fallback agent or action must be relevant to the original query context, an unrelated fallback that doesn't contribute to resolving the query is a failure
5. Failures must not cascade silently, the final output must not present incorrect
   or fabricated results as if no failure occurred

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
  "failure_events": [
    {{
      "failed_agent": "<name of the agent that failed>",
      "recovery_action": "<brief description of what the orchestrator did in response, or 'none' if nothing was done>",
      "passed": <true if the failure was correctly detected and appropriately handled, false otherwise>,
      "reason": "<one sentence>"
    }}
  ],
  "final_output": {{
    "passed": <true if all failure events were correctly handled, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>"
  }}
}}"""


def evaluate_error_recovery(
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
    response = judge.judge(prompt)

    per_failure_results = []
    unhandled_failures = []
    for item in response.get("failure_events", []):
        passed = item.get("passed", False)
        if isinstance(passed, str):
            passed = passed.lower() == "true"
        per_failure_results.append({
            "failed_agent": item.get("failed_agent", ""),
            "recovery_action": item.get("recovery_action", ""),
            "passed": bool(passed),
            "reason": item.get("reason", ""),
        })
        if not passed:
            unhandled_failures.append(item.get("failed_agent", ""))

    final_output = response.get("final_output", {})
    final_output_passed = final_output.get("passed", False)
    if isinstance(final_output_passed, str):
        final_output_passed = final_output_passed.lower() == "true"

    total = len(per_failure_results)
    passed_count = total - len(unhandled_failures)

    return {
        "passed": bool(final_output_passed),
        "reason": f"{passed_count}/{total} failure events correctly handled. " + final_output.get("reason", ""),
        "total_failures": total,
        "unhandled_failures": unhandled_failures,
        "per_failure_results": per_failure_results,
    }


# Test data
SAMPLES = [
    {
        # Primary agent fails, correct fallback invoked, query resolved successfully (expected: PASS)
        "query_id": 1,
        "query": "Find available flights from Karachi to Dubai next Friday.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Find available flights from Karachi to Dubai next Friday.",
                "output": "intent: flight_search, origin: KHI, destination: DXB, date: next Friday",
            },
            {
                "agent_name": "primary_flight_search",
                "agent_role": "Search for available flights using the primary flight database",
                "input": "origin: KHI, destination: DXB, date: next Friday",
                "output": "ERROR: Primary flight database unavailable. Connection timed out.",
            },
            {
                "agent_name": "fallback_flight_search",
                "agent_role": "Search for available flights using the backup flight aggregator",
                "input": "origin: KHI, destination: DXB, date: next Friday",
                "output": "Found 2 flights via backup: EK-601 at $210, FZ-301 at $230.",
            },
            {
                "agent_name": "response_formatter",
                "agent_role": "Format the flight options into a user-friendly response",
                "input": "flights: EK-601 at $210, FZ-301 at $230",
                "output": "Available flights from Karachi to Dubai next Friday: EK-601 ($210), FZ-301 ($230).",
            },
        ],
    },
    {
        # Agent fails, orchestrator proceeds without any recovery and presents wrong output (expected: FAIL)
        "query_id": 2,
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
                "output": "ERROR: Order not found in database.",
            },
            {
                "agent_name": "response_formatter",
                "agent_role": "Format the order details into a user-friendly response",
                "input": "order_id: 7723",
                "output": "Your refund for order #7723 is being processed.",
            },
        ],
    },
    {
        # Agent fails, retry attempted with identical inputs, no corrective action taken (expected: FAIL)
        "query_id": 3,
        "query": "Summarize the latest support ticket for user ID 4821.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Summarize the latest support ticket for user ID 4821.",
                "output": "intent: ticket_summary, user_id: 4821, count: 1",
            },
            {
                "agent_name": "ticket_fetcher",
                "agent_role": "Fetch the most recent support ticket for the given user ID",
                "input": "user_id: 4821, count: 1",
                "output": "ERROR: Ticket service temporarily unavailable.",
            },
            {
                "agent_name": "ticket_fetcher",
                "agent_role": "Fetch the most recent support ticket for the given user ID",
                "input": "user_id: 4821, count: 1",
                "output": "ERROR: Ticket service temporarily unavailable.",
            },
            {
                "agent_name": "summarizer",
                "agent_role": "Generate a concise summary of the provided ticket content",
                "input": "No ticket data available.",
                "output": "Unable to retrieve ticket information for user 4821.",
            },
        ],
    },
    {
        # Agent fails, fallback is triggered but is entirely unrelated to the query context (expected: FAIL)
        "query_id": 4,
        "query": "Book a hotel in Istanbul for two nights starting this Saturday.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Book a hotel in Istanbul for two nights starting this Saturday.",
                "output": "intent: hotel_booking, location: Istanbul, nights: 2, check_in: this Saturday",
            },
            {
                "agent_name": "hotel_search",
                "agent_role": "Search for available hotels matching the given criteria",
                "input": "location: Istanbul, nights: 2, check_in: this Saturday",
                "output": "ERROR: Hotel search service is down for maintenance.",
            },
            {
                "agent_name": "flight_search",
                "agent_role": "Search for available flights matching the given criteria",
                "input": "destination: Istanbul, date: this Saturday",
                "output": "Found 3 flights to Istanbul: TK-001 at $350, PC-202 at $290, W6-405 at $210.",
            },
            {
                "agent_name": "response_formatter",
                "agent_role": "Format results into a user-friendly response",
                "input": "flights to Istanbul: TK-001 at $350, PC-202 at $290, W6-405 at $210",
                "output": "Here are available flights to Istanbul this Saturday: TK-001 ($350), PC-202 ($290), W6-405 ($210).",
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
        result = evaluate_error_recovery(
            judge,
            query=sample["query"],
            agent_trace=sample["agent_trace"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query  : {sample['query']}")
        print(f"  Reason    : {result['reason']}")
        for event in result["per_failure_results"]:
            e_status = "PASS" if event["passed"] else "FAIL"
            print(f"  [{e_status}] {event['failed_agent']} — Recovery: {event['recovery_action']}")
            print(f"         {event['reason']}")
    print()


if __name__ == "__main__":
    main()
