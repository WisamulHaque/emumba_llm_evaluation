"""
Memory and Context Preservation Evaluator

Ensures that context is maintained and no information is lost as the user query
passes through each agent in the pipeline. Evaluates each consecutive agent
handoff to verify that intermediate results and key parameters are correctly
carried forward into the next agent's input.
Uses LLM-as-judge to reason about information transfer fidelity at each handoff.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether context and information are correctly preserved at each agent handoff
in a multi-agent pipeline. For each transition from one agent to the next, determine
whether the receiving agent's input faithfully carries forward all relevant data from
the previous agent's output.

Following inputs will be provided:

Inputs:
- Query: The original user query submitted to the multi-agent system
- Agent Trace: The sequence of agents invoked, each with their role, input, and output

Query:
{query}

Agent Trace:
{agent_trace_block}

Based on the above-provided inputs, use the rules below to evaluate each handoff:

Evaluation Rules:
1. The receiving agent's input must carry forward all information from the previous
   agent's output that is relevant to the query, silently dropping key data is a failure
2. Key entities and parameters established earlier in the chain (e.g. IDs, counts, dates,
   names) must remain present and accurate in all downstream agent inputs that require them
3. Reformatting or rephrasing data between agents is acceptable as long as all information
   is semantically preserved and none is lost
4. If a downstream agent depends on a result produced by an earlier agent (not just the
   immediate predecessor), that result must still be accessible in the downstream agent's input
5. Adding supplementary context at a handoff is acceptable; removing or overwriting required
   context is a failure

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
  "handoffs": [
    {{
      "from_agent": "<name of the agent passing context>",
      "to_agent": "<name of the agent receiving context>",
      "passed": <true if context was fully preserved at this handoff, false otherwise>,
      "reason": "<one sentence>"
    }}
  ],
  "overall": {{
    "passed": <true if all handoffs preserved context correctly, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>"
  }}
}}"""


def evaluate_memory_and_context_preservation(
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

    per_handoff_results = []
    failed_handoffs = []
    for item in response.get("handoffs", []):
        passed = item.get("passed", False)
        if isinstance(passed, str):
            passed = passed.lower() == "true"
        handoff_label = f"{item.get('from_agent', '')} → {item.get('to_agent', '')}"
        per_handoff_results.append({
            "from_agent": item.get("from_agent", ""),
            "to_agent": item.get("to_agent", ""),
            "passed": bool(passed),
            "reason": item.get("reason", ""),
        })
        if not passed:
            failed_handoffs.append(handoff_label)

    overall = response.get("overall", {})
    overall_passed = overall.get("passed", False)
    if isinstance(overall_passed, str):
        overall_passed = overall_passed.lower() == "true"

    total = len(per_handoff_results)
    passed_count = total - len(failed_handoffs)

    return {
        "passed": bool(overall_passed),
        "reason": f"{passed_count}/{total} handoffs preserved context. " + overall.get("reason", ""),
        "total_handoffs": total,
        "failed_handoffs": failed_handoffs,
        "per_handoff_results": per_handoff_results,
    }


# Test data
SAMPLES = [
    {
        # All context correctly passed at every handoff (expected: PASS)
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
                "input": "selected_flight: PK-201, passengers: 2, date: next Friday, origin: KHI, destination: DXB",
                "output": "Booking confirmed. Reference: PK-20124-KHI-DXB. Two seats reserved on PK-201.",
            },
        ],
    },
    {
        # Passenger count silently dropped at the flight_search -> booking_agent handoff (expected: FAIL)
        "query_id": 2,
        "query": "Book a flight from Lahore to Istanbul for three passengers this Saturday.",
        "agent_trace": [
            {
                "agent_name": "intent_classifier",
                "agent_role": "Classify the user intent and extract key parameters from the query",
                "input": "Book a flight from Lahore to Istanbul for three passengers this Saturday.",
                "output": "intent: flight_booking, origin: LHE, destination: IST, passengers: 3, date: this Saturday",
            },
            {
                "agent_name": "flight_search",
                "agent_role": "Search for available flights matching the given criteria and return options",
                "input": "origin: LHE, destination: IST, passengers: 3, date: this Saturday",
                "output": "Found 2 flights: TK-710 at $320 (3 seats available), PK-785 at $290 (2 seats available).",
            },
            {
                "agent_name": "booking_agent",
                "agent_role": "Book the selected flight for the specified passengers and return a confirmation",
                "input": "selected_flight: TK-710, date: this Saturday",
                "output": "Booking confirmed for 1 passenger on TK-710. Reference: TK-710-LHE-IST.",
            },
        ],
    },
    {
        # Ticket ID produced in step 2 not forwarded to the summarizer in step 3 (expected: FAIL)
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
                "output": "Ticket #4401 — payment failure. Ticket #4389 — login issue. Ticket #4301 — account locked.",
            },
            {
                "agent_name": "summarizer",
                "agent_role": "Generate a concise summary of the provided ticket content",
                "input": "user_id: 7742",
                "output": "User 7742 has recently submitted support tickets.",
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
        result = evaluate_memory_and_context_preservation(
            judge,
            query=sample["query"],
            agent_trace=sample["agent_trace"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query  : {sample['query']}")
        if result["passed"]:
            print(f"  Preserved : {result['reason']}")
        else:
            print(f"  Reason    : {result['reason']}")
        for handoff in result["per_handoff_results"]:
            h_status = "PASS" if handoff["passed"] else "FAIL"
            print(f"  [{h_status}] {handoff['from_agent']} → {handoff['to_agent']}: {handoff['reason']}")
    print()


if __name__ == "__main__":
    main()
