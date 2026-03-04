"""
Category 1: Data Source Application Evaluation — Dummy Example

Demonstrates evaluation of a simple LLM application that interacts with
external data sources (RAG, APIs, Databases). Covers retrieval accuracy,
faithfulness, response completeness, and common cross-cutting concerns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DataSourceEvalSample:
    """A single evaluation sample for data-source-backed LLM apps."""
    query: str
    ground_truth: str
    retrieved_contexts: List[str]
    generated_response: str
    cited_sources: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class DataSourceEvaluator:
    """
    Evaluator for Category 1 — Simple LLM Apps with Data Sources.

    Evaluation areas covered (from the guidelines):
        1. Retrieval Accuracy
        2. Response Accuracy
        3. Faithfulness (Hallucination Detection)
        4. Intent Accuracy
        5. Response Completeness
        6. Citation Accuracy
        7. Out-of-Scope Accuracy
    """

    # ----- 1. Retrieval Accuracy -----
    @staticmethod
    def evaluate_retrieval_accuracy(
        query: str, retrieved_contexts: List[str], relevant_keywords: List[str]
    ) -> Dict[str, Any]:
        """Check whether retrieved chunks are relevant to the query."""
        hits = []
        for i, ctx in enumerate(retrieved_contexts):
            ctx_lower = ctx.lower()
            matched = [kw for kw in relevant_keywords if kw.lower() in ctx_lower]
            hits.append({"chunk_index": i, "matched_keywords": matched,
                         "relevant": len(matched) > 0})

        relevant_count = sum(1 for h in hits if h["relevant"])
        score = relevant_count / len(hits) if hits else 0.0
        return {"score": round(score, 3), "per_chunk": hits}

    # ----- 2. Response Accuracy -----
    @staticmethod
    def evaluate_response_accuracy(
        generated: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Simple token-overlap proxy for response accuracy."""
        gen_tokens = set(generated.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        stop = {"the", "a", "an", "is", "are", "of", "in", "to", "and", "for"}
        gen_tokens -= stop
        gt_tokens -= stop
        overlap = gen_tokens & gt_tokens
        score = len(overlap) / len(gt_tokens) if gt_tokens else 1.0
        return {"score": round(score, 3), "overlapping_tokens": sorted(overlap)}

    # ----- 3. Faithfulness -----
    @staticmethod
    def evaluate_faithfulness(
        generated: str, contexts: List[str]
    ) -> Dict[str, Any]:
        """Check that claims in the response are grounded in retrieved context."""
        context_blob = " ".join(contexts).lower()
        words = set(generated.lower().split())
        stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                "to", "and", "for", "it", "that", "this", "with"}
        content = words - stop
        grounded = {w for w in content if w in context_blob}
        score = len(grounded) / len(content) if content else 1.0
        return {
            "score": round(score, 3),
            "grounded_terms": sorted(grounded),
            "ungrounded_terms": sorted(content - grounded),
        }

    # ----- 4. Intent Accuracy -----
    @staticmethod
    def evaluate_intent_accuracy(
        query: str, generated: str
    ) -> Dict[str, Any]:
        """Rough keyword-overlap heuristic for intent fulfilment."""
        q_words = set(query.lower().split()) - {"what", "who", "where", "how",
                                                  "is", "the", "?", "a"}
        g_words = set(generated.lower().split())
        matched = q_words & g_words
        score = len(matched) / len(q_words) if q_words else 1.0
        return {"score": round(score, 3), "matched_intent_terms": sorted(matched)}

    # ----- 5. Response Completeness -----
    @staticmethod
    def evaluate_response_completeness(
        generated: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Check if key facts from the ground truth appear in the response."""
        gt_sentences = [s.strip() for s in ground_truth.split(".") if s.strip()]
        covered = []
        for sent in gt_sentences:
            key_words = set(sent.lower().split()) - {"the", "a", "is", "of", "and"}
            gen_lower = generated.lower()
            hits = sum(1 for w in key_words if w in gen_lower)
            covered.append({
                "fact": sent,
                "coverage": round(hits / len(key_words), 2) if key_words else 1.0,
            })
        avg = sum(c["coverage"] for c in covered) / len(covered) if covered else 1.0
        return {"score": round(avg, 3), "fact_coverage": covered}

    # ----- 6. Citation Accuracy -----
    @staticmethod
    def evaluate_citation_accuracy(
        cited_sources: List[str], contexts: List[str]
    ) -> Dict[str, Any]:
        """Verify that cited sources actually exist in the retrieved contexts."""
        if not cited_sources:
            return {"score": 1.0, "note": "No citations to verify"}
        verified = []
        for src in cited_sources:
            found = any(src.lower() in ctx.lower() for ctx in contexts)
            verified.append({"source": src, "verified": found})
        score = sum(1 for v in verified if v["verified"]) / len(verified)
        return {"score": round(score, 3), "details": verified}

    # ----- 7. Out-of-Scope Accuracy -----
    @staticmethod
    def evaluate_out_of_scope(
        generated: str, scope_keywords: List[str]
    ) -> Dict[str, Any]:
        """Check if the response stays within the application's defined scope."""
        gen_lower = generated.lower()
        in_scope_hits = [kw for kw in scope_keywords if kw.lower() in gen_lower]
        score = len(in_scope_hits) / len(scope_keywords) if scope_keywords else 1.0
        return {"score": round(score, 3), "in_scope_keywords_found": in_scope_hits}

    # ----- Run All -----
    def evaluate(
        self,
        sample: DataSourceEvalSample,
        relevant_keywords: List[str],
        scope_keywords: List[str],
    ) -> Dict[str, Any]:
        """Run all evaluation areas on a single sample."""
        return {
            "retrieval_accuracy": self.evaluate_retrieval_accuracy(
                sample.query, sample.retrieved_contexts, relevant_keywords
            ),
            "response_accuracy": self.evaluate_response_accuracy(
                sample.generated_response, sample.ground_truth
            ),
            "faithfulness": self.evaluate_faithfulness(
                sample.generated_response, sample.retrieved_contexts
            ),
            "intent_accuracy": self.evaluate_intent_accuracy(
                sample.query, sample.generated_response
            ),
            "response_completeness": self.evaluate_response_completeness(
                sample.generated_response, sample.ground_truth
            ),
            "citation_accuracy": self.evaluate_citation_accuracy(
                sample.cited_sources or [], sample.retrieved_contexts
            ),
            "out_of_scope": self.evaluate_out_of_scope(
                sample.generated_response, scope_keywords
            ),
        }


# ---------------------------------------------------------------------------
# Dummy data & main
# ---------------------------------------------------------------------------

def main():
    """Run the evaluator against dummy data."""

    sample = DataSourceEvalSample(
        query="What are the health benefits of green tea?",
        ground_truth=(
            "Green tea contains antioxidants called catechins. "
            "It may reduce the risk of heart disease and improve brain function. "
            "Regular consumption is associated with lower cholesterol levels."
        ),
        retrieved_contexts=[
            "Green tea is rich in catechins, a type of antioxidant that protects cells.",
            "Studies show green tea consumption may lower LDL cholesterol levels.",
            "The caffeine and L-theanine in green tea can improve brain function.",
        ],
        generated_response=(
            "Green tea offers several health benefits. It is rich in catechins, "
            "powerful antioxidants that protect cells. Research suggests it may lower "
            "cholesterol and improve brain function thanks to caffeine and L-theanine."
        ),
        cited_sources=["catechins", "LDL cholesterol"],
    )

    evaluator = DataSourceEvaluator()
    results = evaluator.evaluate(
        sample,
        relevant_keywords=["green tea", "antioxidant", "catechins", "cholesterol"],
        scope_keywords=["health", "tea", "benefit"],
    )

    print("=" * 65)
    print("Category 1 — Data Source Application Evaluation (Dummy)")
    print("=" * 65)
    print(f"\nQuery      : {sample.query}")
    print(f"Response   : {sample.generated_response[:80]}...")
    print(f"\n{json.dumps(results, indent=2)}")

    print("\n" + "-" * 65)
    print("Score Summary")
    print("-" * 65)
    for area, detail in results.items():
        print(f"  {area:30s}  {detail['score']:.2f}")


if __name__ == "__main__":
    main()
