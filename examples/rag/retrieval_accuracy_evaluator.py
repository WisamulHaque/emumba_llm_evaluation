"""
Retrieval Accuracy Evaluation

Validates that chunks retrieved from a document store based on a user query
are relevant and accurate. Poor retrieval leads to poor generation regardless
of LLM capability.

This module uses embedding-based cosine similarity (via sentence-transformers)
to score how semantically relevant each retrieved chunk is to the user query.
This handles paraphrasing and synonyms that lexical methods would miss.

    Example:
        Query : "What is the company's leave policy?"
        Chunk : "Employees are entitled to 20 days of paid time off annually."
        Embedding score : ~0.78 (semantically close)

Requires: pip install sentence-transformers

Configuration:
    Create a .env file (see .env.example) with:
        RETRIEVAL_EVAL_INPUT=sample_input.json
        RETRIEVAL_EVAL_OUTPUT=retrieval_results.jsonl
        RETRIEVAL_EVAL_THRESHOLD=0.5
        RETRIEVAL_EVAL_MODEL=all-MiniLM-L6-v2

Usage:
    # Run with custom input file (requires .env with RETRIEVAL_EVAL_INPUT set)
    python retrieval_accuracy.py

Input JSON format:
    [
        {
            "query_id": 1,
            "query": "What is Python?",
            "retrieved_chunks": ["Python is a programming language...", "..."],
            "ground_truth_chunks": ["optional", "list"]  // optional field
        }
    ]

Output JSONL format (one JSON object per line, written immediately after each evaluation):
    {"query_id": 1, "query": "...", "passed": true, "reason": "...", "relevance_score": 0.78, "relevant_chunks": ["..."]}

Relevance Score Logic:
    Without ground truth:
        - Score each chunk against the query using embedding cosine similarity
        - Final score = best individual chunk score
        - Passed = best_score >= threshold (at least one relevant chunk found)

    With ground truth:
        - For each ground truth chunk, find the best matching retrieved chunk
          using embedding cosine similarity
        - score = recall = matched_gt_chunks / total_gt_chunks
        - Passed = recall >= threshold
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.common.utils import EmbeddingSimilarity, truncate_text

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass


def get_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    return {
        "input_path":  os.environ.get("RETRIEVAL_EVAL_INPUT"),
        "output_path": os.environ.get("RETRIEVAL_EVAL_OUTPUT", "retrieval_results.jsonl"),
        "threshold":   float(os.environ.get("RETRIEVAL_EVAL_THRESHOLD", "0.5")),
        "model_name":  os.environ.get("RETRIEVAL_EVAL_MODEL", "all-MiniLM-L6-v2"),
    }


@dataclass
class RetrievalEvaluationSample:
    """
    Input sample for retrieval accuracy evaluation.
    
    Attributes:
        query_id: Unique identifier for the query
        query: The user's search query
        retrieved_chunks: List of text chunks retrieved from the document store
        ground_truth_chunks: Optional list of expected relevant chunks (for ground-truth evaluation)
        metadata: Optional additional metadata about the retrieval
    """
    query_id: int
    query: str
    retrieved_chunks: List[str]
    ground_truth_chunks: Optional[List[str]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalEvaluationSample":
        """Create sample from dictionary (for JSON input)."""
        return cls(
            query_id=data.get("query_id", 0),
            query=data["query"],
            retrieved_chunks=data["retrieved_chunks"],
            ground_truth_chunks=data.get("ground_truth_chunks") or [],
            metadata=data.get("metadata", {})
        )


class RetrievalEvaluator:
    """
    Evaluator for RAG retrieval accuracy.
    
    Determines whether retrieved chunks are relevant to the user's query using
    embedding cosine similarity. Returns a binary True/False verdict with explanation.

    Ground truth mode:
        When ground_truth_chunks are provided, matching uses similarity-based
        comparison to handle real-world chunking
        variations where the same content may be worded slightly differently.
    """
    
    def __init__(self, relevance_threshold: float = 0.5, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the evaluator.
        
        Args:
            relevance_threshold: Minimum similarity score (0-1) for a chunk to be
                                 considered relevant, and minimum recall (0-1) to
                                 pass in ground truth mode. Default 0.5.
            model_name: Sentence-transformer model to use for embeddings.
                        See https://www.sbert.net/docs/pretrained_models.html
        """
        self.relevance_threshold = relevance_threshold
        self.model_name = model_name
        # Pre-load the model once so the first evaluation isn't slow
        EmbeddingSimilarity.load_model(model_name)
    
    def evaluate_retrieval_accuracy(
        self, 
        sample: RetrievalEvaluationSample
    ) -> Dict[str, Any]:
        """
        Evaluate whether retrieved chunks are relevant to the query.
        
        Strategy:
        - If ground_truth_chunks provided: similarity-based recall against expected chunks
        - Otherwise: average embedding similarity between query and each retrieved chunk
        
        Args:
            sample: RetrievalEvaluationSample containing query and retrieved chunks
            
        Returns:
            Dictionary with:
                - passed: True if retrieval is accurate, False otherwise
                - reason: Explanation of the verdict
                - relevance_score: Internal score used for the decision (0-1)
                - relevant_chunks: List of chunks that passed the threshold
        """
        if not sample.retrieved_chunks:
            return {
                "passed": False,
                "reason": "No chunks were retrieved",
                "relevance_score": 0.0,
                "relevant_chunks": []
            }
        
        if sample.ground_truth_chunks:
            return self._evaluate_with_ground_truth(sample)
        
        return self._evaluate_without_ground_truth(sample)
    
    def _evaluate_with_ground_truth(
        self, 
        sample: RetrievalEvaluationSample
    ) -> Dict[str, Any]:
        """
        Evaluate using ground truth chunks.

        For each ground truth chunk, finds the best matching retrieved chunk
        using embedding similarity. A ground truth chunk is considered matched
        if the best similarity score >= relevance_threshold.

        Score = recall = matched_gt_chunks / total_gt_chunks
        """
        matched_gt = []
        unmatched_gt = []

        for gt_chunk in sample.ground_truth_chunks:
            scores = EmbeddingSimilarity.batch_cosine_similarity(gt_chunk, sample.retrieved_chunks)
            best_score = max(scores)

            if best_score >= self.relevance_threshold:
                matched_gt.append(truncate_text(gt_chunk))
            else:
                unmatched_gt.append(truncate_text(gt_chunk))

        recall = len(matched_gt) / len(sample.ground_truth_chunks)
        passed = recall >= self.relevance_threshold

        if passed:
            reason = f"Retrieved {len(matched_gt)}/{len(sample.ground_truth_chunks)} expected chunks (recall: {recall:.2f})"
        else:
            reason = (
                f"Only {len(matched_gt)}/{len(sample.ground_truth_chunks)} expected chunks were retrieved "
                f"(recall: {recall:.2f}, threshold: {self.relevance_threshold}). "
                f"Missing: {unmatched_gt}"
            )

        return {
            "passed": passed,
            "reason": reason,
            "relevance_score": round(recall, 3),
            "relevant_chunks": matched_gt
        }
    
    def _evaluate_without_ground_truth(
        self, 
        sample: RetrievalEvaluationSample
    ) -> Dict[str, Any]:
        """
        Evaluate using embedding similarity between query and each retrieved chunk.

        Pass condition: at least one chunk scores >= relevance_threshold.
        This reflects how RAG actually works — the LLM only needs one relevant
        chunk to generate a good answer. Averaging would penalize retrievers that
        fetch one strong chunk alongside weaker ones.

        Score = best individual chunk score across all retrieved chunks.
        """
        scores = EmbeddingSimilarity.batch_cosine_similarity(sample.query, sample.retrieved_chunks)
        best_score = max(scores)

        relevant_chunks = [
            truncate_text(chunk)
            for chunk, score in zip(sample.retrieved_chunks, scores)
            if score >= self.relevance_threshold
        ]
        relevant_count = len(relevant_chunks)

        passed = best_score >= self.relevance_threshold

        if passed:
            reason = f"{relevant_count}/{len(scores)} chunks are relevant to the query (best score: {best_score:.2f})"
        else:
            reason = f"No retrieved chunks are sufficiently relevant (best score: {best_score:.2f}, threshold: {self.relevance_threshold})"

        return {
            "passed": passed,
            "reason": reason,
            "relevance_score": round(best_score, 3),
            "relevant_chunks": relevant_chunks
        }