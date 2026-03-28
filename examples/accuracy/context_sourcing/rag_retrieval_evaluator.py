"""
RAG Retrieval Evaluator

Measures how well a retrieval system finds the right chunks for a given query.
Two complementary metrics:

  Recall    — Of all the chunks that SHOULD have been retrieved, how many
              did the system actually find?
              Low recall = the system is missing important information.

  Precision — Of all the chunks the system DID retrieve, how many are
              actually relevant?
              Low precision = the system is pulling in noise / off-topic chunks.

Both use embedding-based cosine similarity (sentence-transformers) to match
chunks, so paraphrased or reworded content still counts as a match.

Requires: pip install sentence-transformers
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import List, Dict, Any
from examples.common.utils import EmbeddingSimilarity, truncate_text


# ---------------------------------------------------------------------------
# Recall — "Did we find everything we needed?"
# ---------------------------------------------------------------------------

def evaluate_recall(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    threshold: float = 0.7,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Measure what fraction of the ground-truth chunks were found in the
    retrieved set.

    For each ground-truth chunk, we find the best-matching retrieved chunk
    using cosine similarity.  If the best score >= threshold, that
    ground-truth chunk counts as "found".

        Recall = found_ground_truth_chunks / total_ground_truth_chunks

    A recall of 1.0 means every expected chunk was retrieved.
    A recall of 0.33 means only one-third of the expected information
    was retrieved — the rest is missing from the context window.

    Args:
        retrieved_chunks:    Chunks the retrieval system returned
        ground_truth_chunks: Chunks that should have been retrieved
        threshold:           Minimum cosine similarity to count as a match
        model_name:          Sentence-transformer model for embeddings

    Returns:
        passed:         True if recall >= threshold
        recall:         Fraction of ground-truth chunks found (0.0–1.0)
        found_chunks:   Ground-truth chunks that were matched
        missing_chunks: Ground-truth chunks that were NOT found
    """
    if not retrieved_chunks:
        return {"passed": False, "recall": 0.0, "reason": "No chunks were retrieved",
                "found_chunks": [], "missing_chunks": [truncate_text(c) for c in ground_truth_chunks]}
    if not ground_truth_chunks:
        return {"passed": False, "recall": 0.0, "reason": "No ground truth chunks provided",
                "found_chunks": [], "missing_chunks": []}

    EmbeddingSimilarity.load_model(model_name)

    found = []
    missing = []
    for gt_chunk in ground_truth_chunks:
        scores = EmbeddingSimilarity.batch_cosine_similarity(gt_chunk, retrieved_chunks)
        if max(scores) >= threshold:
            found.append(truncate_text(gt_chunk))
        else:
            missing.append(truncate_text(gt_chunk))

    recall = len(found) / len(ground_truth_chunks)
    passed = recall >= threshold

    reason = f"{len(found)} of {len(ground_truth_chunks)} expected chunks found (recall: {recall:.2f})"
    if not passed:
        reason = f"Only {reason} — below {threshold} threshold"

    return {
        "passed": passed,
        "recall": round(recall, 3),
        "reason": reason,
        "found_chunks": found,
        "missing_chunks": missing,
    }


# ---------------------------------------------------------------------------
# Precision — "Is what we retrieved actually relevant?"
# ---------------------------------------------------------------------------

def evaluate_precision(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    threshold: float = 0.7,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Measure what fraction of the retrieved chunks are actually relevant.

    For each retrieved chunk, we find the best-matching ground-truth chunk
    using cosine similarity.  If the best score >= threshold, that
    retrieved chunk counts as "relevant".

        Precision = relevant_retrieved_chunks / total_retrieved_chunks

    A precision of 1.0 means every retrieved chunk was relevant — no noise.
    A precision of 0.5 means half the retrieved chunks were off-topic,
    wasting context window space and potentially confusing the LLM.

    Args:
        retrieved_chunks:    Chunks the retrieval system returned
        ground_truth_chunks: Chunks that should have been retrieved
        threshold:           Minimum cosine similarity to count as relevant
        model_name:          Sentence-transformer model for embeddings

    Returns:
        passed:           True if precision >= threshold
        precision:        Fraction of retrieved chunks that are relevant (0.0–1.0)
        relevant_chunks:  Retrieved chunks that matched a ground-truth chunk
        noise_chunks:     Retrieved chunks that matched nothing — noise
    """
    if not retrieved_chunks:
        return {"passed": False, "precision": 0.0, "reason": "No chunks were retrieved",
                "relevant_chunks": [], "noise_chunks": []}
    if not ground_truth_chunks:
        return {"passed": False, "precision": 0.0, "reason": "No ground truth chunks provided",
                "relevant_chunks": [], "noise_chunks": [truncate_text(c) for c in retrieved_chunks]}

    EmbeddingSimilarity.load_model(model_name)

    relevant = []
    noise = []
    for ret_chunk in retrieved_chunks:
        scores = EmbeddingSimilarity.batch_cosine_similarity(ret_chunk, ground_truth_chunks)
        if max(scores) >= threshold:
            relevant.append(truncate_text(ret_chunk))
        else:
            noise.append(truncate_text(ret_chunk))

    precision = len(relevant) / len(retrieved_chunks)
    passed = precision >= threshold

    reason = f"{len(relevant)} of {len(retrieved_chunks)} retrieved chunks are relevant (precision: {precision:.2f})"
    if not passed:
        reason = f"Only {reason} — below {threshold} threshold"

    return {
        "passed": passed,
        "precision": round(precision, 3),
        "reason": reason,
        "relevant_chunks": relevant,
        "noise_chunks": noise,
    }


# ---------------------------------------------------------------------------
# Combined — Recall + Precision in one call
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    threshold: float = 0.7,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Run both recall and precision in a single call.

    Passes only if BOTH recall and precision meet the threshold.

    Returns:
        passed:    True if both metrics >= threshold
        recall:    Recall score (0.0–1.0)
        precision: Precision score (0.0–1.0)
        details:   Full recall and precision result dicts
    """
    recall_result = evaluate_recall(retrieved_chunks, ground_truth_chunks, threshold, model_name)
    precision_result = evaluate_precision(retrieved_chunks, ground_truth_chunks, threshold, model_name)

    passed = recall_result["passed"] and precision_result["passed"]
    reason_parts = []
    if not recall_result["passed"]:
        reason_parts.append(f"recall too low ({recall_result['recall']:.2f})")
    if not precision_result["passed"]:
        reason_parts.append(f"precision too low ({precision_result['precision']:.2f})")

    if passed:
        reason = f"Recall {recall_result['recall']:.2f}, Precision {precision_result['precision']:.2f} — both above {threshold} threshold"
    else:
        reason = f"FAIL — {'; '.join(reason_parts)}"

    return {
        "passed": passed,
        "recall": recall_result["recall"],
        "precision": precision_result["precision"],
        "reason": reason,
        "details": {
            "recall": recall_result,
            "precision": precision_result,
        },
    }


# ===== Test data =====

SAMPLES = [
    {
        # Good retrieval — all 2 expected chunks found, no noise pulled in
        # Recall: 2/2 = 1.0, Precision: 2/2 = 1.0 (expected: PASS)
        "query_id": 1,
        "query": "What is the company's remote work policy?",
        "retrieved_chunks": [
            "Employees may work remotely up to 3 days per week with manager approval.",
            "Remote work requests must be submitted through the HR portal at least 48 hours in advance.",
        ],
        "ground_truth_chunks": [
            "Employees may work remotely up to 3 days per week with manager approval.",
            "Remote work requests must be submitted through the HR portal at least 48 hours in advance.",
        ],
    },
    {
        # Low recall — only 1 of 3 expected chunks was retrieved
        # Recall: 1/3 = 0.33, Precision: 1/2 = 0.5 (expected: FAIL — missing critical info)
        "query_id": 2,
        "query": "What are the health insurance benefits for employees?",
        "retrieved_chunks": [
            "Full-time employees are eligible for health insurance after 90 days of employment.",
            "The company cafeteria serves lunch from 12:00 PM to 2:00 PM on weekdays.",
        ],
        "ground_truth_chunks": [
            "Full-time employees are eligible for health insurance after 90 days of employment.",
            "The company covers 80% of the monthly premium for individual plans.",
            "Dental and vision coverage can be added for an additional $45/month.",
        ],
    },
    {
        # Low precision — all expected info was found, but 2 noise chunks came along
        # Recall: 2/2 = 1.0, Precision: 2/4 = 0.5 (expected: FAIL — too much noise)
        "query_id": 3,
        "query": "How do I request annual leave?",
        "retrieved_chunks": [
            "Annual leave requests are submitted through the employee self-service portal.",
            "Requests must be approved by your direct manager at least 5 business days in advance.",
            "The company was founded in 2003 and is headquartered in Karachi, Pakistan.",
            "Our parking garage has 200 spots available on a first-come, first-served basis.",
        ],
        "ground_truth_chunks": [
            "Annual leave requests are submitted through the employee self-service portal.",
            "Requests must be approved by your direct manager at least 5 business days in advance.",
        ],
    },
    {
        # Completely off-topic retrieval — nothing relevant was found
        # Recall: 0/2 = 0.0, Precision: 0/2 = 0.0 (expected: FAIL)
        "query_id": 4,
        "query": "What is the process for reporting a workplace safety hazard?",
        "retrieved_chunks": [
            "The weather forecast for tomorrow shows sunny skies with temperatures reaching 30°C.",
            "Our Q3 revenue increased by 12% compared to the same quarter last year.",
        ],
        "ground_truth_chunks": [
            "Safety hazards should be reported immediately to your floor warden or through the incident reporting system.",
            "All reported hazards are reviewed within 24 hours by the Health & Safety committee.",
        ],
    },
]


# Main
def main():
    print("=" * 70)
    print("RAG Retrieval Evaluation — Recall + Precision")
    print("=" * 70)

    for sample in SAMPLES:
        result = evaluate_retrieval(
            retrieved_chunks=sample["retrieved_chunks"],
            ground_truth_chunks=sample["ground_truth_chunks"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query     : {sample['query']}")
        print(f"  Recall    : {result['recall']}")
        print(f"  Precision : {result['precision']}")
        print(f"  Reason    : {result['reason']}")

        recall_detail = result["details"]["recall"]
        if recall_detail["missing_chunks"]:
            for chunk in recall_detail["missing_chunks"]:
                print(f"  Missing   : {chunk}")

        precision_detail = result["details"]["precision"]
        if precision_detail["noise_chunks"]:
            for chunk in precision_detail["noise_chunks"]:
                print(f"  Noise     : {chunk}")
    print()


if __name__ == "__main__":
    main()
