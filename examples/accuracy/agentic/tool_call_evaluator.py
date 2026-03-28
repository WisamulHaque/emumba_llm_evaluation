"""
Tool Call Evaluator

Validates that the correct tools are selected and invoked at each step of a
multi-agent pipeline. Verifies that each tool is called with the right parameters
and that fallback tools are used appropriately when primary tools fail.
Uses LLM-as-judge to reason about correctness of tool selection and parameterization.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the correct tools were selected and invoked with the correct parameters
at each step of a multi-agent pipeline execution.

Following inputs will be provided:

Inputs:
- Query: The original user query submitted to the multi-agent system
- Expected Tool Calls: The ordered list of tool calls that should have been made,
  each with the tool name and expected parameters
- Actual Tool Calls: The ordered list of tool calls that were actually made,
  each with the tool name, parameters used, and the tool's output

Query:
{query}

Expected Tool Calls:
{expected_tool_calls_block}

Actual Tool Calls:
{actual_tool_calls_block}

Based on the above-provided inputs, use the rules below to evaluate tool call accuracy:

Evaluation Rules:
1. Each actual tool call must invoke the correct tool as specified in the Expected Tool Calls
2. The parameters passed to each tool must be accurate and complete — missing or incorrect
   parameters relative to the expected values is a failure
3. A tool called with extraneous or irrelevant parameters not present in the expected call
   should be flagged as a failure
4. If a primary tool fails (as indicated by its output) and a fallback tool is used,
   the fallback tool must be appropriate and relevant to the query context
5. Unnecessary tool calls that have no corresponding expected call and serve no clear
   supporting role are a failure
6. A missing expected tool call is a failure unless the query clearly did not require it

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
  "results": [
    {{
      "tool_name": "<tool name from actual call>",
      "passed": <true or false>,
      "reason": "<one sentence>"
    }}
  ],
  "final_output": {{
    "passed": <true if all tool calls are correct and complete, false otherwise>,
    "reason": "<one or two sentences explaining the verdict>"
  }}
}}"""


def evaluate_tool_call_accuracy(
    judge: LLMJudge,
    query: str,
    expected_tool_calls: List[Dict[str, Any]],
    actual_tool_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    expected_tool_calls_block = "\n\n".join(
        f"[{i}] {call['tool_name']}\n"
        f"    Parameters : {call['parameters']}"
        for i, call in enumerate(expected_tool_calls, 1)
    )

    actual_tool_calls_block = "\n\n".join(
        f"[{i}] {call['tool_name']}\n"
        f"    Parameters : {call['parameters']}\n"
        f"    Output     : {call.get('output', 'N/A')}"
        for i, call in enumerate(actual_tool_calls, 1)
    )

    prompt = EVAL_PROMPT.format(
        query=query,
        expected_tool_calls_block=expected_tool_calls_block,
        actual_tool_calls_block=actual_tool_calls_block,
    )
    response = judge.judge(prompt)

    per_tool_results = []
    failed_tools = []
    for item in response.get("results", []):
        passed = item.get("passed", False)
        if isinstance(passed, str):
            passed = passed.lower() == "true"
        per_tool_results.append({
            "tool_name": item.get("tool_name", ""),
            "passed": bool(passed),
            "reason": item.get("reason", ""),
        })
        if not passed:
            failed_tools.append(item.get("tool_name", ""))

    final_output = response.get("final_output", {})
    overall_passed = final_output.get("passed", False)
    if isinstance(overall_passed, str):
        overall_passed = overall_passed.lower() == "true"

    total = len(actual_tool_calls)
    passed_count = total - len(failed_tools)

    return {
        "passed": bool(overall_passed),
        "reason": f"{passed_count}/{total} tool calls passed. " + final_output.get("reason", ""),
        "total_tools": total,
        "failed_tools": failed_tools,
        "per_tool_results": per_tool_results,
    }


# Test data
SAMPLES = [
    {
        # All tools called correctly with right parameters (expected: PASS)
        "query_id": 1,
        "query": "Book a flight from Karachi to Dubai for two passengers next Friday.",
        "expected_tool_calls": [
            {
                "tool_name": "search_flights",
                "parameters": {"origin": "KHI", "destination": "DXB", "passengers": 2, "date": "next Friday"},
            },
            {
                "tool_name": "select_seat",
                "parameters": {"flight_id": "PK-201", "passengers": 2},
            },
            {
                "tool_name": "confirm_booking",
                "parameters": {"flight_id": "PK-201", "passengers": 2, "seat_numbers": ["12A", "12B"]},
            },
        ],
        "actual_tool_calls": [
            {
                "tool_name": "search_flights",
                "parameters": {"origin": "KHI", "destination": "DXB", "passengers": 2, "date": "next Friday"},
                "output": "Found 3 flights: EK-601 at $210, PK-201 at $185, FZ-301 at $230.",
            },
            {
                "tool_name": "select_seat",
                "parameters": {"flight_id": "PK-201", "passengers": 2},
                "output": "Seats 12A and 12B reserved on PK-201.",
            },
            {
                "tool_name": "confirm_booking",
                "parameters": {"flight_id": "PK-201", "passengers": 2, "seat_numbers": ["12A", "12B"]},
                "output": "Booking confirmed. Reference: PK-20124-KHI-DXB.",
            },
        ],
    },
    {
        # Wrong tool called at one step — cancel_booking used instead of confirm_booking (expected: FAIL)
        "query_id": 2,
        "query": "Book a flight from Lahore to Istanbul for one passenger tomorrow.",
        "expected_tool_calls": [
            {
                "tool_name": "search_flights",
                "parameters": {"origin": "LHE", "destination": "IST", "passengers": 1, "date": "tomorrow"},
            },
            {
                "tool_name": "confirm_booking",
                "parameters": {"flight_id": "TK-710", "passengers": 1},
            },
        ],
        "actual_tool_calls": [
            {
                "tool_name": "search_flights",
                "parameters": {"origin": "LHE", "destination": "IST", "passengers": 1, "date": "tomorrow"},
                "output": "Found 2 flights: TK-710 at $320, PK-785 at $290.",
            },
            {
                "tool_name": "cancel_booking",
                "parameters": {"flight_id": "TK-710", "passengers": 1},
                "output": "Booking TK-710 has been cancelled.",
            },
        ],
    },
    {
        # Correct tool called but with a missing required parameter (expected: FAIL)
        "query_id": 3,
        "query": "Get the weather forecast for New York for the next 5 days.",
        "expected_tool_calls": [
            {
                "tool_name": "get_weather_forecast",
                "parameters": {"location": "New York", "days": 5},
            },
        ],
        "actual_tool_calls": [
            {
                "tool_name": "get_weather_forecast",
                "parameters": {"location": "New York"},
                "output": "Today in New York: Partly cloudy, 18°C.",
            },
        ],
    },
    {
        # Unnecessary extra tool call injected alongside otherwise correct calls (expected: FAIL)
        "query_id": 4,
        "query": "Summarize the latest support ticket submitted by user ID 4821.",
        "expected_tool_calls": [
            {
                "tool_name": "fetch_ticket",
                "parameters": {"user_id": 4821, "order": "latest"},
            },
            {
                "tool_name": "summarize_text",
                "parameters": {"text": "Ticket #9921 — login failure after password reset."},
            },
        ],
        "actual_tool_calls": [
            {
                "tool_name": "fetch_ticket",
                "parameters": {"user_id": 4821, "order": "latest"},
                "output": "Ticket #9921 — User reports login failure after password reset. Status: Open.",
            },
            {
                "tool_name": "send_email",
                "parameters": {"recipient": "support@example.com", "subject": "Ticket Alert"},
                "output": "Email sent successfully.",
            },
            {
                "tool_name": "summarize_text",
                "parameters": {"text": "Ticket #9921 — login failure after password reset."},
                "output": "User 4821 has an open ticket reporting a login failure after a password reset.",
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
        result = evaluate_tool_call_accuracy(
            judge,
            query=sample["query"],
            expected_tool_calls=sample["expected_tool_calls"],
            actual_tool_calls=sample["actual_tool_calls"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query       : {sample['query']}")
        print(f"  Reason      : {result['reason']}")
        if result["failed_tools"]:
            print(f"  Failed Tools: {', '.join(result['failed_tools'])}")
        for tool_result in result["per_tool_results"]:
            tool_status = "PASS" if tool_result["passed"] else "FAIL"
            print(f"  [{tool_status}] {tool_result['tool_name']}: {tool_result['reason']}")
    print()


if __name__ == "__main__":
    main()
