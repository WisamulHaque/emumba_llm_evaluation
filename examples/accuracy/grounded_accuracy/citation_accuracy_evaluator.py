"""
Citation Accuracy Evaluator

Verifies that citations in an LLM-generated response are accurate: the cited
source must actually contain the information attributed to it, and the claim
must not be attributed to the wrong source.

Uses an LLM judge to verify each citation independently, then aggregates
pass/fail across all citations.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Dict, Any, List
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Verify whether each citation in an AI-generated response is accurate. A citation
is accurate only if the cited source explicitly contains the information attributed
to it in the response.

Following inputs will be provided:

Inputs:
  Query: the user's question
  Generated Response: the AI response containing citations
  Citations: a list of cited sources, each with its source name and actual content

Query:
{query}

Generated Response:
{generated_response}

Citations:
{citations_block}

Based on the above-provided inputs, use the rules below to evaluate each citation:

Evaluation Rules:
1. For each citation, identify the specific claim the response attributes to that source
2. A citation passes only if the cited source content explicitly supports the attributed claim
3. A citation fails if the source content does not contain the attributed information, even
   if the claim is factually correct from general knowledge
4. A citation fails if the claim is correct but attributed to the wrong source
5. Do not use outside knowledge, only evaluate based on the cited source content provided above

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
  "results": [
    {{
      "source": "<source name>",
      "claim": "<the claim attributed to this source in the response>",
      "passed": <true if the citation is accurate, false if not>,
      "reason": "<one sentence explaining the verdict>"
    }}
  ]
}}"""


def evaluate_citation_accuracy(
    judge: LLMJudge,
    query: str,
    generated_response: str,
    citations: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Evaluate whether citations in the generated response are accurate.

    Args:
        judge:              LLMJudge instance
        query:              The user's question
        generated_response: The LLM response containing citations
        citations:          List of dicts, each with:
                              "source"  the citation label used in the response (e.g. "Doc A")
                              "content" the actual text content of that source

    Returns a dict with:
        passed:               True if all citations are accurate
        reason:               Summary of accurate vs total citations
        total_citations:      Number of citations evaluated
        inaccurate_citations: List of source names whose citations failed
    """
    if not generated_response:
        return {"passed": False, "reason": "No generated response provided.", "total_citations": 0, "inaccurate_citations": []}
    if not citations:
        return {"passed": True, "reason": "No citations provided to evaluate.", "total_citations": 0, "inaccurate_citations": []}

    citations_block = "\n\n".join(
        f"[{c['source']}]: {c['content']}" for c in citations
    )

    prompt = EVAL_PROMPT.format(
        query=query,
        generated_response=generated_response,
        citations_block=citations_block,
    )

    result = judge.judge(prompt)
    results = result.get("results")
    if not isinstance(results, list):
        raise ValueError("Citation verification failed: 'results' must be a list.")

    if not results:
        return {"passed": True, "reason": "No citations detected in response.", "total_citations": 0, "inaccurate_citations": []}

    accurate_count = 0
    inaccurate_citations = []
    for item in results:
        if item.get("passed", False):
            accurate_count += 1
        else:
            inaccurate_citations.append(item.get("source", "Unknown source"))

    total = len(results)
    score = accurate_count / total
    return {
        "passed": len(inaccurate_citations) == 0,
        "score": round(score, 2),
        "reason": f"{accurate_count} out of {total} citations are accurate.",
        "total_citations": total,
        "inaccurate_citations": inaccurate_citations,
    }


# Test data
SAMPLES = [
    {
        # All citations are accurate: each cited source contains the attributed information (expected: PASS)
        "query_id": 1,
        "query": "What are the main health benefits of regular exercise?",
        "generated_response": (
            "Regular exercise improves cardiovascular health by strengthening the heart muscle "
            "and lowering blood pressure [Doc A]. It also reduces the risk of type 2 diabetes "
            "by improving insulin sensitivity [Doc B]."
        ),
        "citations": [
            {
                "source": "Doc A",
                "content": (
                    "Cardiovascular exercise strengthens the heart muscle over time, leading to "
                    "improved cardiac output and lower resting blood pressure in most adults."
                ),
            },
            {
                "source": "Doc B",
                "content": (
                    "Physical activity improves the body's sensitivity to insulin, which helps "
                    "regulate blood sugar levels and significantly reduces the risk of developing "
                    "type 2 diabetes."
                ),
            },
        ],
    },
    {
        # One citation is inaccurate: Doc B is cited for a claim it does not support (expected: FAIL)
        "query_id": 2,
        "query": "How does sleep affect cognitive performance?",
        "generated_response": (
            "Sleep deprivation impairs memory consolidation and reduces attention span [Doc A]. "
            "Getting eight hours of sleep also boosts creativity and problem-solving ability [Doc B]."
        ),
        "citations": [
            {
                "source": "Doc A",
                "content": (
                    "Lack of sleep negatively impacts the brain's ability to consolidate memories "
                    "and sustains attention. Studies show reaction times and focus deteriorate "
                    "significantly after 24 hours without sleep."
                ),
            },
            {
                "source": "Doc B",
                "content": (
                    "The recommended sleep duration for adults is seven to nine hours per night. "
                    "Consistent sleep schedules help regulate the body's circadian rhythm."
                ),
            },
        ],
    },
    {
        # Citation attributes a correct claim to the wrong source (expected: FAIL)
        "query_id": 3,
        "query": "What causes inflation?",
        "generated_response": (
            "Inflation is caused by excess money supply relative to goods and services [Doc A]. "
            "Supply chain disruptions can also drive up prices by reducing the availability "
            "of goods [Doc A]."
        ),
        "citations": [
            {
                "source": "Doc A",
                "content": (
                    "Inflation occurs when the money supply grows faster than the production of "
                    "goods and services, reducing the purchasing power of currency."
                ),
            },
            {
                "source": "Doc B",
                "content": (
                    "Supply chain disruptions reduce the availability of goods, which drives up "
                    "prices due to reduced supply meeting the same level of demand."
                ),
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
        result = evaluate_citation_accuracy(
            judge,
            query=sample["query"],
            generated_response=sample["generated_response"],
            citations=sample["citations"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query                : {sample['query']}")
        print(f"  Reason               : {result['reason']}")
        if result["inaccurate_citations"]:
            for source in result["inaccurate_citations"]:
                print(f"  Inaccurate citation  : {source}")
    print()


if __name__ == "__main__":
    main()
