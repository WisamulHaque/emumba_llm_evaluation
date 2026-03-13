---
layout: default
title: "Category 2: Multi-Agent Applications"
parent: LLM Evaluation Guidelines
grand_parent: Use Cases
nav_order: 2
permalink: /use-cases/category-2-multi-agent-apps/
---

# Category 2: Multi-Agent Applications

These are complex applications composed of multiple specialized agents, tools, and decision pathways. The LLM orchestrates across these agents to resolve a user query, often involving multi-step reasoning and tool usage.

---

## Evaluation Areas

**I. Behavioral Accuracy (Journey Validation)**
Validate that the pathway or journey taken by the LLM through the agent graph is correct. Given a user query, the sequence of agents invoked and decisions made should follow the expected logical flow.

*Evaluator:* [`examples/multi-agent/behavioral_accuracy_evaluator.py`](../../examples/multi-agent/behavioral_accuracy_evaluator.py)

*Run:*
```bash
python examples/multi-agent/behavioral_accuracy_evaluator.py
```

*Examples:*

| Query | Expected Path | Actual Path | Result |
|---|---|---|---|
| Book a flight from Karachi to Dubai for two passengers next Friday. | intent_classifier, flight_search, seat_selector, booking_agent, confirmation_agent | Same sequence | PASS — actual path matches the expected journey exactly |
| Purchase the annual Pro plan and send me a receipt. | intent_classifier, plan_selector, payment_agent, receipt_agent | intent_classifier, plan_selector, confirmation_agent | FAIL — payment_agent skipped; receipt_agent never invoked |

**II. Agent Accuracy**
Evaluate the output of each individual agent in the pipeline. Each agent should produce accurate results within its scope, and the final aggregated output should be correct.

*Evaluator:* [`examples/multi-agent/agent_accuracy_evaluator.py`](../../examples/multi-agent/agent_accuracy_evaluator.py)

*Run:*
```bash
python examples/multi-agent/agent_accuracy_evaluator.py
```

*Examples:*

| Query | Agent Trace Summary | Final Output | Result |
|---|---|---|---|
| Book a flight from Karachi to Dubai for two passengers next Friday. | intent_classifier correctly extracts booking intent; flight_search returns valid options; booking_agent confirms PK-201 for 2 seats | "Your flight has been booked. Reference: PK-20124-KHI-DXB. Two seats reserved on PK-201." | PASS — all agents accurate; final output correctly reflects the booking |
| Cancel my existing booking and issue a refund. | intent_classifier outputs wrong intent (flight_booking instead of cancellation); flight_search returns no results | "I could not find any available flights matching your request." | FAIL — intent misclassified at entry; chain processed a new search instead of a cancellation |

**III. Tool Call Accuracy**
Validate that the correct tools are selected and invoked at each step. This includes verifying that the right tool is called with the right parameters and that fallback tools are used appropriately when primary tools fail.

*Evaluator:* [`examples/multi-agent/tool_call_accuracy_evaluator.py`](../../examples/multi-agent/tool_call_accuracy_evaluator.py)

*Run:*
```bash
python examples/multi-agent/tool_call_accuracy_evaluator.py
```

*Examples:*

| Query | Expected Tools | Actual Tools | Result |
|---|---|---|---|
| Book a flight from Karachi to Dubai for two passengers next Friday. | search_flights, select_seat, confirm_booking | Same three tools with correct parameters | PASS — all tools called in the right order with matching parameters |
| Book a flight from Lahore to Istanbul for one passenger tomorrow. | search_flights, select_seat, confirm_booking | search_flights, select_seat, cancel_booking | FAIL — cancel_booking called instead of confirm_booking at the final step |

**IV. Intent Resolution**
Analyze the user query alongside the tool calls and agent outputs to determine if the user's original intent was correctly understood and resolved throughout the entire agent chain.

*Evaluator:* [`examples/multi-agent/intent_resolution_evaluator.py`](../../examples/multi-agent/intent_resolution_evaluator.py)

*Run:*
```bash
python examples/multi-agent/intent_resolution_evaluator.py
```

*Examples:*

| Query | Agent Chain Outcome | Final Output | Result |
|---|---|---|---|
| Book a flight from Karachi to Dubai for two passengers next Friday. | Intent correctly extracted; flight found and booking confirmed for 2 seats on PK-201 | Booking reference and seat details returned | PASS — original intent preserved and resolved end-to-end |
| Cancel my booking REF-8821 and issue a full refund. | intent_classifier outputs "flight_booking" instead of cancellation; chain searches for new flights instead | "I was unable to find any available flights for your request." | FAIL — intent misclassified at entry; refund never processed |

**V. Task Adherence**
Validate that the end goal of the user's query is fully achieved. This is a holistic check on the final output to confirm it directly and completely addresses what the user asked for.

*Evaluator:* [`examples/multi-agent/task_adherence_evaluator.py`](../../examples/multi-agent/task_adherence_evaluator.py)

*Run:*
```bash
python examples/multi-agent/task_adherence_evaluator.py
```

*Examples:*

| Query | Final Output | Result |
|---|---|---|
| Book a flight from Karachi to Dubai for two passengers next Friday. | "Your flight has been booked. Confirmation reference: PK-20124-KHI-DXB. Two seats reserved on PK-201 from Karachi to Dubai next Friday." | PASS — final output fully satisfies all explicit requirements of the query |
| Cancel my booking REF-8821 and issue a full refund. | "Your booking REF-8821 has been successfully cancelled." | FAIL — cancellation confirmed but refund not processed; query only partially fulfilled |

**VI. Memory and Context Preservation**
Ensure that context is maintained and no information is lost as the user query passes through each agent in the pipeline. Intermediate results and conversation history should be preserved across agent handoffs.

*Evaluator:* [`examples/multi-agent/memory_context_preservation_evaluator.py`](../../examples/multi-agent/memory_context_preservation_evaluator.py)

*Run:*
```bash
python examples/multi-agent/memory_context_preservation_evaluator.py
```

*Examples:*

| Query | Handoff Summary | Result |
|---|---|---|
| Book a flight from Karachi to Dubai for two passengers next Friday. | intent_classifier passes origin, destination, passengers, and date; flight_search and booking_agent both receive and use all fields correctly | PASS — all context correctly passed at every handoff |
| Book a flight from Lahore to Istanbul for three passengers this Saturday. | intent_classifier correctly extracts passengers: 3; booking_agent receives only selected_flight and date; passenger count silently dropped | FAIL — passenger count lost at the flight_search to booking_agent handoff; booking confirmed for 1 passenger instead of 3 |

**VII. Orchestration Efficiency**
Evaluate whether the agent orchestration follows an optimal path. Redundant agent calls, unnecessary loops, or suboptimal routing through the agent graph should be flagged.

*Evaluator:* [`examples/multi-agent/orchestration_efficiency_evaluator.py`](../../examples/multi-agent/orchestration_efficiency_evaluator.py)

*Run:*
```bash
python examples/multi-agent/orchestration_efficiency_evaluator.py
```

*Examples:*

| Query | Agent Trace | Result |
|---|---|---|
| Summarize the latest support ticket submitted by user ID 4821. | intent_classifier, ticket_fetcher, summarizer — clean 3-step path with each output feeding the next | PASS — minimal path with no redundant or unused steps |
| Find available flights from Karachi to Dubai next Friday. | intent_classifier, flight_search, flight_search (same inputs), response_formatter | FAIL — flight_search called twice with identical parameters; redundant duplicate agent call |

**VIII. Error Recovery and Fallback Handling**
Validate how the system behaves when an individual agent fails or returns an unexpected result. The orchestrator should have fallback mechanisms and should not cascade failures silently.

*Evaluator:* [`examples/multi-agent/error_recovery_evaluator.py`](../../examples/multi-agent/error_recovery_evaluator.py)

*Run:*
```bash
python examples/multi-agent/error_recovery_evaluator.py
```

*Examples:*

| Query | Failure Event | Recovery Action | Result |
|---|---|---|---|
| Find available flights from Karachi to Dubai next Friday. | primary_flight_search times out with a connection error | fallback_flight_search invoked and returns 2 flights via backup aggregator | PASS — failure correctly detected and fallback successfully resolved the query |
| Get the refund status for order ID 7723. | order_fetcher returns ERROR: Order not found | No fallback invoked; response_formatter proceeds with empty data and fabricates a status | FAIL — failure not handled; fabricated output presented to user as if the order was found |

**IX. Plan Accuracy**
Validate the Plan generated by LLM is accurate and selects the Right tools.

*Evaluator:* [`examples/multi-agent/plan_accuracy_evaluator.py`](../../examples/multi-agent/plan_accuracy_evaluator.py)

*Run:*
```bash
python examples/multi-agent/plan_accuracy_evaluator.py
```

*Examples:*

| Query | Plan | Result |
|---|---|---|
| Book a flight from Karachi to Dubai for two passengers next Friday. | intent_classifier, flight_search, booking_agent, response_formatter — all required tools in correct order | PASS — plan is complete, correctly ordered, and sufficient to resolve the query |
| Reserve a hotel in Istanbul for two nights starting this Saturday. | intent_classifier, hotel_search, response_formatter | FAIL — booking_agent missing; plan can search for hotels but cannot actually complete the reservation |
