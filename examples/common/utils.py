"""
Common Utilities for LLM Evaluation

Shared utilities, metrics, and helper functions used across
different evaluation examples.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import os
import re
import time
from dotenv import load_dotenv
load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False


def truncate_text(text: str, max_length: int = 60) -> str:
    """Truncate text and add ellipsis if too long."""
    text = text.strip().replace('\n', ' ')
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


# LLM provider imports — each is optional, error raised only if used
try:
    import anthropic as anthropic_sdk
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import openai as openai_sdk
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Standard evaluation result structure."""
    metric_name: str
    score: float
    details: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "details": self.details,
            "timestamp": self.timestamp
        }


class MetricsCalculator:
    """Common metrics calculations."""
    
    @staticmethod
    def precision_at_k(relevant: List[int], retrieved: List[Any], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            relevant: List of relevant item indices
            retrieved: List of retrieved items
            k: Number of top results to consider
            
        Returns:
            Precision@K score
        """
        if k <= 0 or not retrieved:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for i, _ in enumerate(top_k) if i in relevant)
        
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(relevant: List[int], retrieved: List[Any], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            relevant: List of relevant item indices
            retrieved: List of retrieved items
            k: Number of top results to consider
            
        Returns:
            Recall@K score
        """
        if not relevant:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for i, _ in enumerate(top_k) if i in relevant)
        
        return relevant_in_top_k / len(relevant)
    
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_reciprocal_rank(relevant: List[int], retrieved: List[Any]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            relevant: List of relevant item indices
            retrieved: List of retrieved items
            
        Returns:
            MRR score
        """
        for i, _ in enumerate(retrieved):
            if i in relevant:
                return 1.0 / (i + 1)
        
        return 0.0


class TextSimilarity:
    """Text similarity metrics."""
    
    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    @staticmethod
    def word_overlap(text1: str, text2: str) -> float:
        """
        Calculate word overlap ratio.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Overlap ratio
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        min_len = min(len(words1), len(words2))
        
        return overlap / min_len


class EmbeddingSimilarity:
    """
    Semantic similarity using sentence-transformers.

    Requires: pip install sentence-transformers

    Handles cases where lexical similarity fails due to paraphrasing or synonyms.
    Example:
        Query : "What is the company's leave policy?"
        Chunk : "Employees are entitled to 20 days of paid time off annually."
        Lexical score   : ~0.0  (no word overlap)
        Embedding score : ~0.78 (semantically close)
    """

    _model: Optional[object] = None
    _model_name: str = ""

    @classmethod
    def load_model(cls, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Load the sentence-transformer model. Called once and reused.

        Args:
            model_name: Model from https://www.sbert.net/docs/pretrained_models.html
                        Recommended:
                        - "all-MiniLM-L6-v2"  → fast, 80MB, good general quality
                        - "all-mpnet-base-v2" → slower, higher accuracy
        """
        if not _EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers"
            )
        if cls._model is None or cls._model_name != model_name:
            cls._model = SentenceTransformer(model_name)
            cls._model_name = model_name

    @classmethod
    def batch_cosine_similarity(cls, query: str, candidates: List[str]) -> List[float]:
        """
        Cosine similarity between one query and multiple candidates in a single
        model call — more efficient than scoring each candidate individually.

        Args:
            query: The search query or reference text
            candidates: List of chunks or candidate texts to score against

        Returns:
            List of cosine similarity scores (0–1), one per candidate
        """
        if not _EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers"
            )
        if cls._model is None:
            cls.load_model()

        all_texts = [query] + candidates
        embeddings = cls._model.encode(all_texts)

        query_vec = embeddings[0]
        query_norm = float(np.linalg.norm(query_vec))

        scores = []
        for i in range(1, len(embeddings)):
            chunk_vec = embeddings[i]
            dot = float(np.dot(query_vec, chunk_vec))
            norm = query_norm * float(np.linalg.norm(chunk_vec))
            scores.append(dot / norm if norm > 0 else 0.0)

        return scores


class EvaluationReporter:
    """Generate evaluation reports."""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
    
    def add_result(self, result: EvaluationResult):
        """Add an evaluation result."""
        self.results.append(result)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all results."""
        if not self.results:
            return {"message": "No results to summarize"}
        
        summary = {
            "total_evaluations": len(self.results),
            "metrics": {}
        }
        
        for result in self.results:
            if result.metric_name not in summary["metrics"]:
                summary["metrics"][result.metric_name] = {
                    "scores": [],
                    "count": 0
                }
            
            summary["metrics"][result.metric_name]["scores"].append(result.score)
            summary["metrics"][result.metric_name]["count"] += 1
        
        # Calculate averages
        for metric, data in summary["metrics"].items():
            data["average"] = sum(data["scores"]) / len(data["scores"])
            data["min"] = min(data["scores"])
            data["max"] = max(data["scores"])
        
        return summary
    
    def to_json(self, filepath: str = None) -> str:
        """Export results to JSON."""
        data = {
            "results": [r.to_dict() for r in self.results],
            "summary": self.generate_summary()
        }
        
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str


# ── Guardrails helpers ──────────────────────────────────────────────────────

def _check_openai_moderation(
    api_key: str,
    user_message: str,
    assistant_response: str,
    score_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Check whether a response violates OpenAI content policies."""
    if not _OPENAI_AVAILABLE:
        raise ImportError("OpenAI SDK not installed. Run: pip install openai")
    client = openai_sdk.OpenAI(api_key=api_key)
    result = client.moderations.create(
        input=f"User: {user_message}\nAssistant: {assistant_response}"
    ).results[0]

    if score_threshold is not None:
        violated = [cat for cat, score in result.category_scores.__dict__.items() if score > score_threshold]
    else:
        violated = [cat for cat, flagged in result.categories.__dict__.items() if flagged]

    passed = len(violated) == 0
    return {
        "passed": passed,
        "reason": "No policy violations detected." if passed else f"Violated: {', '.join(violated)}.",
        "violated_categories": violated,
    }


# Example usage
if __name__ == "__main__":
    # Test metrics
    calc = MetricsCalculator()
    
    relevant = [0, 2, 4]
    retrieved = list(range(10))
    
    print("Metrics Calculator Demo")
    print("=" * 40)
    print(f"Precision@3: {calc.precision_at_k(relevant, retrieved, 3):.3f}")
    print(f"Recall@3: {calc.recall_at_k(relevant, retrieved, 3):.3f}")
    print(f"MRR: {calc.mean_reciprocal_rank(relevant, retrieved):.3f}")
    
    # Test similarity
    sim = TextSimilarity()
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "The fast brown fox leaps over the sleepy dog"
    
    print("\nText Similarity Demo")
    print("=" * 40)
    print(f"Jaccard Similarity: {sim.jaccard_similarity(text1, text2):.3f}")
    print(f"Word Overlap: {sim.word_overlap(text1, text2):.3f}")