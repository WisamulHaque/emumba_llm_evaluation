"""
Consistency Evaluation

Verifies that the application produces consistent quality outputs for similar
queries. Repeated queries with the same intent should not yield wildly different
quality levels or contradictory answers.

Two dimensions are evaluated independently:

    Factual Consistency
        Do the responses contradict each other on facts? A system that gives
        conflicting answers to paraphrased versions of the same question has
        a reliability problem regardless of which answer is correct.

    Quality Consistency
        Is the helpfulness, completeness, and depth of responses similar
        across the group? Wild variance in quality for the same intent
        indicates an unreliable system.

Note: This evaluator applies universally across all LLM application types.

Usage:
    python run_eval.py consistency

"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.common.llm_judge import LLMJudge


# Data model
MIN_GROUP_SIZE = 2
MAX_GROUP_SIZE = 5

@dataclass
class QueryResponsePair:
    """A single query and its generated response within a consistency group."""
    query_id: int
    query:    str
    generated_response: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryResponsePair":
        return cls(
            query_id=data.get("query_id", 0),
            query=data["query"],
            generated_response=data["generated_response"],
        )


@dataclass
class ConsistencyGroup:
    """
    A group of query-response pairs sharing the same intent.

    Attributes:
        group_id:  Unique identifier for the group
        intent:    Human-defined description of what all queries in the group
                   are asking. Used by the judge as the reference point for
                   evaluating whether responses address the same underlying need.
        queries:   List of QueryResponsePair — minimum 2, maximum 5.
        metadata:  Optional additional metadata
    """
    group_id: int
    intent:   str
    queries:  List[QueryResponsePair]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsistencyGroup":
        """Create group from dictionary (for JSON input)."""
        queries = [QueryResponsePair.from_dict(q) for q in data.get("queries", [])]

        if len(queries) < MIN_GROUP_SIZE:
            raise ValueError(
                f"Group {data.get('group_id')} has {len(queries)} queries. "
                f"Minimum group size is {MIN_GROUP_SIZE}."
            )
        
        if len(queries) > MAX_GROUP_SIZE:
            raise ValueError(
                f"Group {data.get('group_id')} has {len(queries)} queries. "
                f"Maximum group size is {MAX_GROUP_SIZE}."
            )

        return cls(
            group_id=data.get("group_id", 0),
            intent=data["intent"],
            queries=queries,
            metadata=data.get("metadata", {}),
        )


# Consistency Evaluator
class ConsistencyEvaluator:
    """
    Evaluator for response consistency across paraphrased queries.

    The judge receives the full group — all queries and responses together —
    and evaluates both factual consistency and quality consistency as a set.

    A group passes only if both dimensions pass.
    """

    JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether an AI assistant \
produces consistent outputs for queries with the same underlying intent.

Shared Intent of All Queries Below:
{intent}

Queries and Responses:
{queries_block}

Evaluate the responses on TWO dimensions:

1. FACTUAL CONSISTENCY
   Do the responses contradict each other on any facts, claims, or information?
   - PASS: All responses are factually compatible — no contradictions
   - FAIL: One or more responses make claims that directly contradict another response

2. QUALITY CONSISTENCY
   Is the helpfulness, completeness, and depth of responses similar across the group?
   - PASS: Responses are at a comparable quality level — no wild variance
   - FAIL: Quality varies wildly — e.g. one response is detailed and accurate while
           another is vague, incomplete, or unhelpful for the same intent

For each contradiction found, describe it specifically (e.g. "Response 2 says X while Response 1 says Y").
If no contradictions, return an empty list.

Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "factual_consistency": {{
        "passed": <true or false>,
        "reason": "<one or two sentences>"
    }},
    "quality_consistency": {{
        "passed": <true or false>,
        "reason": "<one or two sentences>"
    }},
    "contradictions": ["<contradiction description>", ...]
}}"""

    def __init__(self, judge: LLMJudge):
        """
        Initialize the evaluator.

        Args:
            judge: Configured LLMJudge instance
        """
        self.judge = judge

    def evaluate_consistency(self, group: ConsistencyGroup) -> Dict[str, Any]:
        """
        Evaluate response consistency across a group of paraphrased queries.

        Args:
            group: ConsistencyGroup with intent and list of query-response pairs

        Returns:
            Dictionary with:
                - passed:               True if both dimensions pass
                - factual_consistency:  {"passed": bool, "reason": str}
                - quality_consistency:  {"passed": bool, "reason": str}
                - contradictions:       List of specific contradiction descriptions
                                        (empty list if none found)
        """
        queries_block = self._build_queries_block(group.queries)

        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            intent=group.intent,
            queries_block=queries_block,
        )

        result = self.judge.judge(prompt)
        return self._build_result(result)

    def _build_queries_block(self, queries: List[QueryResponsePair]) -> str:
        """Format all query-response pairs for the judge prompt."""
        blocks = []
        for i, qr in enumerate(queries, 1):
            blocks.append(
                f"[Query {i}]: {qr.query}\n"
                f"[Response {i}]: {qr.generated_response}"
            )
        return "\n\n".join(blocks)

    def _build_result(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalise the judge's response into the output contract.

        Both dimensions must pass for the overall result to pass.
        """
        factual = raw.get("factual_consistency", {})
        quality = raw.get("quality_consistency", {})
        contradictions = raw.get("contradictions", [])

        # Normalise — judge may return bool or 0/1
        factual_passed = bool(factual.get("passed", False))
        quality_passed = bool(quality.get("passed", False))

        return {
            "passed":               factual_passed and quality_passed,
            "factual_consistency":  {
                "passed": factual_passed,
                "reason": factual.get("reason", ""),
            },
            "quality_consistency":  {
                "passed": quality_passed,
                "reason": quality.get("reason", ""),
            },
            "contradictions": contradictions if isinstance(contradictions, list) else [],
        }