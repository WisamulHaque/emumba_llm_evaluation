"""
Out-of-Scope Accuracy Evaluation

Validates that the application correctly identifies and handles queries outside
its defined scope. When users ask questions beyond the application's domain,
it should gracefully decline rather than hallucinate an answer. Conversely,
it should not over-refuse queries that are genuinely in scope.

Evaluation approach:

    LLM Judge (primary decision)
        The judge receives the full context: system prompt, query, generated
        response, and expected behavior.
        It makes the final pass/fail decision.

        For expected_behavior="refuse":
            Did the model correctly decline the out-of-scope query without
            hallucinating an answer?

        For expected_behavior="answer":
            Did the model correctly engage with the in-scope query, or did
            it incorrectly refuse (over-refusal)?

expected_behavior values:
    refuse  — model should have declined the query (out-of-scope)
    answer  — model should have engaged with the query (in-scope)

Note: This evaluator applies universally across all LLM application types.

Usage:
    python out_of_scope_runner.py

Input JSON format:
    [
        {
            "query_id": 1,
            "system_prompt": "You are a customer support bot for AcmeCorp software.",
            "query": "What is the capital of France?",
            "generated_response": "I can only help with AcmeCorp product questions.",
            "expected_behavior": "refuse"
        }
    ]

Output JSONL format (one JSON object per line):
    {
        "query_id": 1,
        "query": "...",
        "expected_behavior": "refuse",
        "passed": true,
        "reason": "..."
    }
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from examples.common.llm_judge import LLMJudge

# Constants
VALID_BEHAVIORS   = ("refuse", "answer")

# Data model
@dataclass
class OutOfScopeSample:
    """
    Input sample for out-of-scope accuracy evaluation.

    Attributes:
        query_id:           Unique identifier for tracking in output
        system_prompt:      The application's system prompt — defines what is
                            in scope. Passed to the LLM judge so it can reason
                            about domain boundaries accurately.
        query:              The user's query to evaluate
        generated_response: The LLM application's response to the query
        expected_behavior:  Required. "refuse" (out-of-scope query, model should
                            decline) or "answer" (in-scope query, model should
                            engage). Use "answer" to catch over-refusal.
        metadata:           Optional additional metadata
    """
    query_id:           int
    system_prompt:      str
    query:              str
    generated_response: str
    expected_behavior:  str
    metadata:           Dict[str, Any]     = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutOfScopeSample":
        """Create sample from dictionary (for JSON input)."""
        expected = data.get("expected_behavior")
        if expected not in VALID_BEHAVIORS:
            raise ValueError(
                f"Invalid expected_behavior '{expected}'. "
                f"Must be one of: {VALID_BEHAVIORS}"
            )

        return cls(
            query_id=data.get("query_id", 0),
            system_prompt=data["system_prompt"],
            query=data["query"],
            generated_response=data["generated_response"],
            expected_behavior=expected,
            metadata=data.get("metadata", {})
        )


class OutOfScopeEvaluator:
    """
    Evaluator for out-of-scope accuracy.
    """

    JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether an AI assistant correctly \
handled a query relative to its defined scope.

Application System Prompt (defines the assistant's domain and scope):
{system_prompt}

User Query:
{query}

Assistant Response:
{generated_response}

Expected Behavior: {expected_behavior}

Instructions:
- Use the system prompt as the source of truth for what is in scope and out of scope
- For expected_behavior="refuse": the assistant should have declined the query gracefully
  without hallucinating an answer. It passes if it declined clearly and did not engage
  with out-of-scope content.
- For expected_behavior="answer": the assistant should have engaged with the query. It
  fails if it refused a query that was genuinely within its defined scope (over-refusal).
- A graceful decline that briefly explains why it cannot help is better than a terse
  refusal, but both pass as long as the model did not answer the out-of-scope content.


Score 1 if the assistant handled the query correctly, 0 if it did not.

Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "score": <1 or 0>,
    "reason": "<one or two sentences explaining the verdict>"
}}"""

    def __init__(self, judge: LLMJudge):
        """
        Initialize the evaluator.

        Args:
            judge: Configured LLMJudge instance for the primary decision
        """
        self.judge = judge

    def evaluate_out_of_scope(self, sample: OutOfScopeSample) -> Dict[str, Any]:
        """
        Evaluate whether the model correctly handled an out-of-scope query.

        Args:
            sample: OutOfScopeSample with system prompt, query, response,
                    and expected behavior

        Returns:
            Dictionary with:
                - passed:            True if model handled the query correctly
                - reason:            Explanation from the LLM judge
                - expected_behavior: Echoed from input for traceability
        """
        if not sample.generated_response:
            return {
                "passed":            False,
                "reason":            "No generated response provided",
                "expected_behavior": sample.expected_behavior,
            }


        # LLM judge
        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            system_prompt=sample.system_prompt,
            query=sample.query,
            generated_response=sample.generated_response,
            expected_behavior=sample.expected_behavior
        )

        result = self.judge.judge(prompt)
        passed = int(result["score"]) == 1

        return {
            "passed":            passed,
            "reason":            result["reason"],
            "expected_behavior": sample.expected_behavior
        }