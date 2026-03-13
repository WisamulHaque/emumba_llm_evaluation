"""
Intent Resolution Evaluator

Analyzes the user query alongside the agent trace and final output to determine
if the user's original intent was correctly understood and resolved throughout
the entire agent chain. Uses LLM-as-judge to reason holistically across all
intermediate agent steps, not just the final response.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the user's original intent was correctly understood and resolved
throughout the entire agent chain, from the initial query to the final output.

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

Based on the above-provided inputs, use the rules below to evaluate intent resolution:

Evaluation Rules:
1. The user's core intent must be correctly identified at the start of the chain —
   misclassifying or misinterpreting the intent at entry is a failure
2. The intent must remain consistent as it passes through each agent — any agent that
   reframes, narrows, or distorts the intent relative to the original query is a deviation
3. Key parameters or constraints expressed in the Query (e.g. quantities, dates, names,
   conditions) must be preserved and propagated correctly across all agents
4. The Final Output must directly satisfy the user's original intent, not a
   reinterpreted or partial version of it
5. If the Final Output addresses a different goal than what the user asked for,
   even if it is technically coherent, it is a failure

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if the user's intent was correctly understood and fully resolved, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "deviations": ["<describe each point where intent was misunderstood or lost, or empty list if none>"]
}}"""


def evaluate_intent_resolution(
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


# Test data
SAMPLES = [
    {
        # Intent correctly understood, preserved, and resolved end-to-end (expected: PASS)
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
                "agent_role": "Book the selected flight for the specified passengers and return a confirmation",
                "input": "selected_flight: PK-201, passengers: 2, date: next Friday",
                "output": "Booking confirmed. Reference: PK-20124-KHI-DXB. Two seats reserved on PK-201.",
            },
        ],
        "final_output": "Your flight has been booked. Confirmation reference: PK-20124-KHI-DXB. Two seats reserved on PK-201 from Karachi to Dubai next Friday.",
    },
    {
        # Intent misclassified at entry — user wants a refund but chain processes a new booking (expected: FAIL)
        "query_id": 2,
        "query": "Cancel my booking REF-8821 and issue a full refund.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Cancel my booking REF-8821 and issue a full refund.",
                "output": "intent: flight_booking, reference: REF-8821",
            },
            {
                "agent_name": "flight_search",
                "agent_role": "Search for available flights matching the given criteria",
                "input": "intent: flight_booking, reference: REF-8821",
                "output": "No new flights found matching reference REF-8821.",
            },
        ],
        "final_output": "I was unable to find any available flights for your request.",
    },
    {
        # Intent understood but a key parameter (user ID) is dropped mid-chain (expected: FAIL)
        "query_id": 3,
        "query": "Summarize the last 3 support tickets submitted by user ID 7742.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Summarize the last 3 support tickets submitted by user ID 7742.",
                "output": "intent: ticket_summary, user_id: 7742, count: 3",
            },
            {
                "agent_name": "ticket_fetcher",
                "agent_role": "Fetch the most recent support tickets for the given user",
                "input": "user_id: 7742, count: 3",
                "output": "Fetched only the latest ticket: Ticket #4401 — payment failure. Status: Open.",
            },
            {
                "agent_name": "summarizer",
                "agent_role": "Generate a concise summary of the provided ticket content",
                "input": "Ticket #4401 — payment failure. Status: Open.",
                "output": "User 7742 has an open ticket reporting a payment failure.",
            },
        ],
        "final_output": "User 7742 has one open support ticket reporting a payment failure.",
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
        result = evaluate_intent_resolution(
            judge,
            query=sample["query"],
            agent_trace=sample["agent_trace"],
            final_output=sample["final_output"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query    : {sample['query']}")
        print(f"  Reason   : {result['reason']}")
        if result["deviations"]:
            print(f"  Deviations:")
            for i, d in enumerate(result["deviations"], 1):
                print(f"    {i}. {d}")
    print()


if __name__ == "__main__":
    main()
