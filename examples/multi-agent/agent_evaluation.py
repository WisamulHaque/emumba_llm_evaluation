"""
Category 2: Multi-Agent Application Evaluation — Dummy Example

Demonstrates evaluation of a multi-agent LLM system that orchestrates
across several specialized agents and tools. Covers journey validation,
tool call accuracy, orchestration efficiency, and error recovery.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Represents a single tool/agent invocation."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    success: bool = True


@dataclass
class AgentStep:
    """A single step in the agent pipeline."""
    agent_name: str
    input_data: Any
    output_data: Any
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class MultiAgentEvalSample:
    """Complete trace for evaluating a multi-agent execution."""
    query: str
    expected_journey: List[str]           # expected ordered agent names
    expected_tool_calls: List[str]        # expected ordered tool names
    actual_steps: List[AgentStep]
    final_output: str
    expected_output: str
    plan: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class MultiAgentEvaluator:
    """
    Evaluator for Category 2 — Multi-Agent Applications.

    Evaluation areas covered (from the guidelines):
        1. Behavioral Accuracy (Journey Validation)
        2. Agent Accuracy
        3. Tool Call Accuracy
        4. Intent Resolution
        5. Task Adherence
        6. Memory & Context Preservation
        7. Orchestration Efficiency
        8. Error Recovery & Fallback Handling
        9. Plan Accuracy
    """

    # ----- 1. Behavioral Accuracy (Journey Validation) -----
    @staticmethod
    def evaluate_journey(
        expected_journey: List[str], actual_steps: List[AgentStep]
    ) -> Dict[str, Any]:
        """Validate the sequence of agents matches the expected flow."""
        actual_journey = [step.agent_name for step in actual_steps]
        match = actual_journey == expected_journey
        # Compute longest-common-subsequence ratio as partial score
        lcs = 0
        j = 0
        for agent in expected_journey:
            for k in range(j, len(actual_journey)):
                if actual_journey[k] == agent:
                    lcs += 1
                    j = k + 1
                    break
        max_len = max(len(expected_journey), len(actual_journey), 1)
        score = lcs / max_len
        return {
            "score": round(score, 3),
            "exact_match": match,
            "expected": expected_journey,
            "actual": actual_journey,
        }

    # ----- 2. Agent Accuracy -----
    @staticmethod
    def evaluate_agent_accuracy(actual_steps: List[AgentStep]) -> Dict[str, Any]:
        """Check each agent produced a non-empty, successful output."""
        results = []
        for step in actual_steps:
            ok = step.output_data is not None and str(step.output_data).strip() != ""
            results.append({"agent": step.agent_name, "produced_output": ok})
        score = sum(1 for r in results if r["produced_output"]) / len(results) if results else 0
        return {"score": round(score, 3), "per_agent": results}

    # ----- 3. Tool Call Accuracy -----
    @staticmethod
    def evaluate_tool_calls(
        expected_tools: List[str], actual_steps: List[AgentStep]
    ) -> Dict[str, Any]:
        """Validate the correct tools were invoked in the correct order."""
        actual_tools = []
        for step in actual_steps:
            for tc in step.tool_calls:
                actual_tools.append(tc.tool_name)
        match = actual_tools == expected_tools
        correct = sum(1 for a, e in zip(actual_tools, expected_tools) if a == e)
        total = max(len(expected_tools), len(actual_tools), 1)
        return {
            "score": round(correct / total, 3),
            "exact_match": match,
            "expected": expected_tools,
            "actual": actual_tools,
        }

    # ----- 4. Intent Resolution -----
    @staticmethod
    def evaluate_intent_resolution(
        query: str, final_output: str
    ) -> Dict[str, Any]:
        """Check if the final output resolves the user's original intent."""
        q_words = set(query.lower().split()) - {"can", "you", "please", "the", "a", "?"}
        f_words = set(final_output.lower().split())
        matched = q_words & f_words
        score = len(matched) / len(q_words) if q_words else 1.0
        return {"score": round(score, 3), "matched_terms": sorted(matched)}

    # ----- 5. Task Adherence -----
    @staticmethod
    def evaluate_task_adherence(
        final_output: str, expected_output: str
    ) -> Dict[str, Any]:
        """Holistic check — does the final output match expectations?"""
        exp_tokens = set(expected_output.lower().split())
        fin_tokens = set(final_output.lower().split())
        stop = {"the", "a", "is", "of", "and", "to", "in"}
        exp_tokens -= stop
        fin_tokens -= stop
        overlap = exp_tokens & fin_tokens
        score = len(overlap) / len(exp_tokens) if exp_tokens else 1.0
        return {"score": round(score, 3), "overlapping_tokens": sorted(overlap)}

    # ----- 6. Memory & Context Preservation -----
    @staticmethod
    def evaluate_context_preservation(
        actual_steps: List[AgentStep],
    ) -> Dict[str, Any]:
        """Ensure important tokens survive across agent handoffs."""
        if len(actual_steps) < 2:
            return {"score": 1.0, "note": "Single step — nothing to compare"}
        carry_scores = []
        for i in range(1, len(actual_steps)):
            prev_out = str(actual_steps[i - 1].output_data).lower().split()
            curr_in = str(actual_steps[i].input_data).lower().split()
            prev_set = set(prev_out)
            curr_set = set(curr_in)
            overlap = prev_set & curr_set
            score = len(overlap) / len(prev_set) if prev_set else 1.0
            carry_scores.append(round(score, 3))
        avg = sum(carry_scores) / len(carry_scores) if carry_scores else 1.0
        return {"score": round(avg, 3), "per_handoff": carry_scores}

    # ----- 7. Orchestration Efficiency -----
    @staticmethod
    def evaluate_orchestration_efficiency(
        expected_journey: List[str], actual_steps: List[AgentStep]
    ) -> Dict[str, Any]:
        """Flag redundant or unnecessary agent calls."""
        actual_names = [s.agent_name for s in actual_steps]
        redundant = len(actual_names) - len(set(actual_names))
        extra = max(0, len(actual_names) - len(expected_journey))
        score = max(0, 1.0 - 0.2 * (redundant + extra))
        return {
            "score": round(score, 3),
            "redundant_calls": redundant,
            "extra_calls": extra,
        }

    # ----- 8. Error Recovery & Fallback -----
    @staticmethod
    def evaluate_error_recovery(actual_steps: List[AgentStep]) -> Dict[str, Any]:
        """Check how the system handled any tool failures."""
        failures = []
        for step in actual_steps:
            for tc in step.tool_calls:
                if not tc.success:
                    failures.append({
                        "agent": step.agent_name,
                        "tool": tc.tool_name,
                        "had_fallback": step.output_data is not None,
                    })
        if not failures:
            return {"score": 1.0, "failures": 0, "note": "No failures detected"}
        recovered = sum(1 for f in failures if f["had_fallback"])
        score = recovered / len(failures) if failures else 1.0
        return {"score": round(score, 3), "failures": len(failures), "details": failures}

    # ----- 9. Plan Accuracy -----
    @staticmethod
    def evaluate_plan_accuracy(
        plan: Optional[str], expected_tools: List[str]
    ) -> Dict[str, Any]:
        """Validate the plan text mentions the right tools."""
        if not plan:
            return {"score": 0.0, "note": "No plan provided"}
        plan_lower = plan.lower()
        mentioned = [t for t in expected_tools if t.lower() in plan_lower]
        score = len(mentioned) / len(expected_tools) if expected_tools else 1.0
        return {"score": round(score, 3), "tools_in_plan": mentioned}

    # ----- Run All -----
    def evaluate(self, sample: MultiAgentEvalSample) -> Dict[str, Any]:
        return {
            "journey_validation": self.evaluate_journey(
                sample.expected_journey, sample.actual_steps
            ),
            "agent_accuracy": self.evaluate_agent_accuracy(sample.actual_steps),
            "tool_call_accuracy": self.evaluate_tool_calls(
                sample.expected_tool_calls, sample.actual_steps
            ),
            "intent_resolution": self.evaluate_intent_resolution(
                sample.query, sample.final_output
            ),
            "task_adherence": self.evaluate_task_adherence(
                sample.final_output, sample.expected_output
            ),
            "context_preservation": self.evaluate_context_preservation(
                sample.actual_steps
            ),
            "orchestration_efficiency": self.evaluate_orchestration_efficiency(
                sample.expected_journey, sample.actual_steps
            ),
            "error_recovery": self.evaluate_error_recovery(sample.actual_steps),
            "plan_accuracy": self.evaluate_plan_accuracy(
                sample.plan, sample.expected_tool_calls
            ),
        }


# ---------------------------------------------------------------------------
# Dummy data & main
# ---------------------------------------------------------------------------

def main():
    """Run the multi-agent evaluator against dummy data."""

    # Simulate a travel-planning multi-agent system
    sample = MultiAgentEvalSample(
        query="Book me a round-trip flight from NYC to London and a hotel near Big Ben for 3 nights.",
        expected_journey=["intent_parser", "flight_agent", "hotel_agent", "summary_agent"],
        expected_tool_calls=["search_flights", "book_flight", "search_hotels", "book_hotel"],
        plan=(
            "1. Parse user intent to extract origin, destination, dates.\n"
            "2. Use search_flights to find round-trip options NYC→London.\n"
            "3. Use book_flight to reserve the best option.\n"
            "4. Use search_hotels near Big Ben for 3 nights.\n"
            "5. Use book_hotel to confirm the reservation.\n"
            "6. Summarise the full itinerary."
        ),
        actual_steps=[
            AgentStep(
                agent_name="intent_parser",
                input_data="Book me a round-trip flight from NYC to London and a hotel near Big Ben for 3 nights.",
                output_data={"origin": "NYC", "destination": "London", "hotel_location": "Big Ben", "nights": 3},
                tool_calls=[],
            ),
            AgentStep(
                agent_name="flight_agent",
                input_data={"origin": "NYC", "destination": "London"},
                output_data={"flight_id": "BA117", "price": "$450", "departure": "08:00", "arrival": "20:00"},
                tool_calls=[
                    ToolCall("search_flights", {"from": "NYC", "to": "London", "round_trip": True},
                             result=[{"id": "BA117", "price": 450}]),
                    ToolCall("book_flight", {"flight_id": "BA117"},
                             result={"confirmation": "FL-98321"}),
                ],
            ),
            AgentStep(
                agent_name="hotel_agent",
                input_data={"location": "Big Ben", "nights": 3},
                output_data={"hotel": "Park Plaza Westminster", "total": "$600"},
                tool_calls=[
                    ToolCall("search_hotels", {"near": "Big Ben", "nights": 3},
                             result=[{"name": "Park Plaza Westminster", "price_per_night": 200}]),
                    ToolCall("book_hotel", {"hotel": "Park Plaza Westminster", "nights": 3},
                             result={"confirmation": "HT-44210"}),
                ],
            ),
            AgentStep(
                agent_name="summary_agent",
                input_data={
                    "flight": {"flight_id": "BA117", "price": "$450"},
                    "hotel": {"hotel": "Park Plaza Westminster", "total": "$600"},
                },
                output_data=(
                    "Your trip is booked! Flight BA117 NYC→London ($450) and "
                    "3 nights at Park Plaza Westminster near Big Ben ($600). "
                    "Total: $1,050."
                ),
                tool_calls=[],
            ),
        ],
        final_output=(
            "Your trip is booked! Flight BA117 NYC→London ($450) and "
            "3 nights at Park Plaza Westminster near Big Ben ($600). Total: $1,050."
        ),
        expected_output=(
            "Round-trip flight NYC to London booked. Hotel near Big Ben for 3 nights booked."
        ),
    )

    evaluator = MultiAgentEvaluator()
    results = evaluator.evaluate(sample)

    print("=" * 65)
    print("Category 2 — Multi-Agent Application Evaluation (Dummy)")
    print("=" * 65)
    print(f"\nQuery  : {sample.query}")
    print(f"Output : {sample.final_output[:80]}...")
    print(f"\n{json.dumps(results, indent=2)}")

    print("\n" + "-" * 65)
    print("Score Summary")
    print("-" * 65)
    for area, detail in results.items():
        print(f"  {area:30s}  {detail['score']:.2f}")


if __name__ == "__main__":
    main()
