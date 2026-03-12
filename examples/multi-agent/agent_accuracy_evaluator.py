"""
Agent Accuracy Evaluator

Evaluates the output of each individual agent in a multi-agent pipeline.
Each agent is assessed independently based on its role, input, and output.
Uses LLM-as-judge to reason about correctness within each agent's stated purpose.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate the accuracy of each agent in a multi-agent pipeline and the final output delivered to the user.

Following inputs will be provided:

Inputs:
- Query: The original user query submitted to the multi-agent system
- Agent Trace: The sequence of agents invoked, each with their role, input, and output
- Final Output: The final response delivered to the user after all agents have processed the Query

Query:
{query}

Agent Trace:
{agent_trace_block}

Final Output:
{final_output}

Based on the above-provided inputs, use the rules below to evaluate agent accuracy:

Evaluation Rules:
1. Each agent's output must correctly fulfill its Agent Role given its Agent Input
2. Each agent's output must not contain incorrect, irrelevant, or hallucinated information relative to its Agent Input
3. A partial agent output that misses key information required by the Agent Role is a failure
4. The Final Output must directly and completely address the Query
5. The Final Output must not contain incorrect or hallucinated information relative to the Query
6. A Final Output that is vague, incomplete, or only partially answers the Query is a failure

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
  "results": [
    {{
      "agent_name": "<agent name>",
      "passed": <true or false>,
      "reason": "<one sentence>"
    }}
  ],
  "final_output": {{
    "passed": <true if the Final Output correctly and completely resolves the Query, false otherwise>,
    "reason": "<one or two sentences explaining the verdict>"
  }}
}}"""


def evaluate_agent_accuracy(
    judge: LLMJudge,
    query: str,
    agent_trace: List[Dict[str, Any]],
    final_output: str,
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
        final_output=final_output,
    )
    response = judge.judge(prompt)

    per_agent_results = []
    failed_agents = []
    for item in response.get("results", []):
        passed = item.get("passed", False)
        if isinstance(passed, str):
            passed = passed.lower() == "true"
        per_agent_results.append({
            "agent_name": item.get("agent_name", ""),
            "passed": bool(passed),
            "reason": item.get("reason", ""),
        })
        if not passed:
            failed_agents.append(item.get("agent_name", ""))

    final = response.get("final_output", {})
    final_passed = final.get("passed", False)
    if isinstance(final_passed, str):
        final_passed = final_passed.lower() == "true"
    final_output_result = {
        "passed": bool(final_passed),
        "reason": final.get("reason", ""),
    }

    total = len(agent_trace)
    passed_count = total - len(failed_agents)
    final_status = "PASS" if final_output_result["passed"] else "FAIL"

    return {
        "passed": len(failed_agents) == 0 and final_output_result["passed"],
        "reason": f"{passed_count}/{total} agents passed. Final output: {final_status}.",
        "total_agents": total,
        "failed_agents": failed_agents,
        "per_agent_results": per_agent_results,
        "final_output_result": final_output_result,
    }


# Test data
SAMPLES = [
    {
        # All agents correct, final output correctly confirms the booking (expected: PASS)
        "query_id": 1,
        "query": "Book a flight from Karachi to Dubai for two passengers next Friday.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Book a flight from Karachi to Dubai for two passengers next Friday.",
                "output": "intent: flight_booking, origin: KHI, destination: DXB, passengers: 2, date: next Friday",
            },
            {
                "agent_name": "flight_search",
                "agent_role": "Search for available flights matching the given criteria and return options",
                "input": "origin: KHI, destination: DXB, passengers: 2, date: next Friday",
                "output": "Found 3 flights: EK-601 at $210, PK-201 at $185, FZ-301 at $230. All have 2 seats available.",
            },
            {
                "agent_name": "booking_agent",
                "agent_role": "Book the selected flight for the specified passengers and return a booking confirmation",
                "input": "selected_flight: PK-201, passengers: 2, date: next Friday",
                "output": "Booking confirmed. Reference: PK-20124-KHI-DXB. Two seats reserved on PK-201.",
            },
        ],
        "final_output": "Your flight has been booked. Confirmation reference: PK-20124-KHI-DXB. Two seats reserved on PK-201 from Karachi to Dubai next Friday.",
    },
    {
        # intent_classifier outputs wrong intent; final output fails to cancel or refund (expected: FAIL)
        "query_id": 2,
        "query": "Cancel my existing booking and issue a refund.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Cancel my existing booking and issue a refund.",
                "output": "intent: flight_booking, origin: unknown, destination: unknown",
            },
            {
                "agent_name": "flight_search",
                "agent_role": "Search for available flights matching the given criteria",
                "input": "intent: flight_booking, origin: unknown, destination: unknown",
                "output": "No flights found for the given criteria.",
            },
        ],
        "final_output": "I could not find any available flights matching your request.",
    },
    {
        # All agents correct, final output is a concise and accurate ticket summary (expected: PASS)
        "query_id": 3,
        "query": "Summarize the latest support ticket submitted by user ID 4821.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Summarize the latest support ticket submitted by user ID 4821.",
                "output": "intent: ticket_summary, user_id: 4821",
            },
            {
                "agent_name": "ticket_fetcher",
                "agent_role": "Fetch the most recent support ticket for the given user ID",
                "input": "user_id: 4821",
                "output": "Ticket #9921 — User reports login failure after password reset. Submitted 2 hours ago. Status: Open.",
            },
            {
                "agent_name": "summarizer",
                "agent_role": "Generate a concise summary of the provided support ticket content",
                "input": "Ticket #9921 — User reports login failure after password reset. Submitted 2 hours ago. Status: Open.",
                "output": "User 4821 submitted an open ticket 2 hours ago reporting a login failure following a password reset.",
            },
        ],
        "final_output": "User 4821 has an open support ticket (2 hours old) reporting a login failure after a password reset.",
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
        result = evaluate_agent_accuracy(
            judge,
            query=sample["query"],
            agent_trace=sample["agent_trace"],
            final_output=sample["final_output"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query  : {sample['query']}")
        print(f"  Result : {result['reason']}")
        for agent_result in result["per_agent_results"]:
            agent_status = "PASS" if agent_result["passed"] else "FAIL"
            print(f"\n  [{agent_result['agent_name']}] {agent_status}")
            print(f"    Reason : {agent_result['reason']}")
        final_status = "PASS" if result["final_output_result"]["passed"] else "FAIL"
        print(f"\n  [final_output] {final_status}")
        print(f"    Reason : {result['final_output_result']['reason']}")
    print()


if __name__ == "__main__":
    main()
