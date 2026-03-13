"""
Retrieval Accuracy Evaluator

Validates that chunks retrieved from a document store match the expected
ground truth chunks for a given query. Uses embedding-based cosine similarity
(via sentence-transformers) to handle paraphrasing and synonyms that lexical
methods would miss.

Requires: pip install sentence-transformers
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Any
from examples.common.utils import EmbeddingSimilarity, truncate_text


def evaluate_retrieval_accuracy(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    threshold: float = 0.7,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Evaluate whether the retrieved chunks cover all expected ground truth chunks.

    For each ground truth chunk, finds the best matching retrieved chunk using
    embedding cosine similarity. A ground truth chunk is considered matched if
    its best similarity score >= threshold.

    Score = recall = matched_gt_chunks / total_gt_chunks
    Passed = recall >= threshold

    Returns a dict with:
        - passed: True if recall meets the threshold, False otherwise
        - reason: Summary of matched vs total ground truth chunks
        - recall_score: Fraction of ground truth chunks that were matched (0.0–1.0)
        - missing_chunks: Ground truth chunks not found in the retrieved set
    """
    if not retrieved_chunks:
        return {"passed": False, "reason": "No chunks were retrieved", "recall_score": 0.0, "missing_chunks": list(ground_truth_chunks)}
    if not ground_truth_chunks:
        return {"passed": False, "reason": "No ground truth chunks provided", "recall_score": 0.0, "missing_chunks": []}

    EmbeddingSimilarity.load_model(model_name)

    matched = []
    missing = []

    for gt_chunk in ground_truth_chunks:
        scores = EmbeddingSimilarity.batch_cosine_similarity(gt_chunk, retrieved_chunks)
        if max(scores) >= threshold:
            matched.append(truncate_text(gt_chunk))
        else:
            missing.append(truncate_text(gt_chunk))

    recall = len(matched) / len(ground_truth_chunks)
    passed = recall >= threshold

    if passed:
        reason = f"Retrieved {len(matched)}/{len(ground_truth_chunks)} expected chunks (recall: {recall:.2f})"
    else:
        reason = (
            f"Only {len(matched)}/{len(ground_truth_chunks)} expected chunks were retrieved "
            f"(recall: {recall:.2f}, threshold: {threshold})"
        )

    return {
        "passed": passed,
        "reason": reason,
        "recall_score": round(recall, 3),
        "missing_chunks": missing,
    }


# Test data
SAMPLES = [
    {
        # All ground truth chunks are present in the retrieved set (expected: PASS)
        "query_id": 1,
        "query": "What are Python decorators and how do they work?",
        "retrieved_chunks": [
            "Python decorators are functions that modify the behavior of other functions using the @decorator syntax.",
            "They are commonly used for logging, authentication, and caching.",
            "Common built-in decorators in Python include @staticmethod, @classmethod, and @property.",
        ],
        "ground_truth_chunks": [
            "Python decorators are functions that modify the behavior of other functions using the @decorator syntax.",
            "Common built-in decorators in Python include @staticmethod, @classmethod, and @property.",
        ],
    },
    {
        # Only one of the two ground truth chunks is retrieved — recall below threshold (expected: FAIL)
        "query_id": 2,
        "query": "How do database indexes improve query performance?",
        "retrieved_chunks": [
            "Database indexes speed up data retrieval by avoiding full table scans.",
            "The history of ancient Rome spans over a thousand years, from the founding of the city in 753 BC.",
        ],
        "ground_truth_chunks": [
            "Database indexes speed up data retrieval by avoiding full table scans.",
            "B-tree indexes are the most common type and are used for range queries and exact matches.",
            "Indexes add write overhead because they must be updated on every insert, update, or delete.",
        ],
    },
    {
        # Retrieved chunks are entirely off-topic relative to the ground truth (expected: FAIL)
        "query_id": 3,
        "query": "What are the benefits of using async/await in Python?",
        "retrieved_chunks": [
            "The weather forecast for tomorrow shows sunny skies with temperatures reaching 75 degrees Fahrenheit.",
            "Recipe for chocolate chip cookies: mix flour, sugar, and butter, then bake at 350 degrees.",
        ],
        "ground_truth_chunks": [
            "Async/await allows writing asynchronous code that is more readable than callback-based approaches.",
            "It improves I/O performance by not blocking the main thread during I/O operations.",
        ],
    },
]


# Main
def main():
    for sample in SAMPLES:
        result = evaluate_retrieval_accuracy(
            retrieved_chunks=sample["retrieved_chunks"],
            ground_truth_chunks=sample["ground_truth_chunks"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Recall        : {result['recall_score']}")
        print(f"  Reason        : {result['reason']}")
        if result["missing_chunks"]:
            for chunk in result["missing_chunks"]:
                print(f"  Missing chunk : {chunk}")
    print()


if __name__ == "__main__":
    main()