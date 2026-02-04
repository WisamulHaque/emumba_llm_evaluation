"""
Common Utilities for LLM Evaluation

Shared utilities, metrics, and helper functions used across
different evaluation examples.
"""

from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import json
import time


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


def create_evaluation_prompt(
    criteria: str,
    question: str,
    response: str,
    context: str = None
) -> str:
    """
    Create a prompt for LLM-as-judge evaluation.
    
    Args:
        criteria: Evaluation criteria description
        question: The original question
        response: The response to evaluate
        context: Optional context provided
        
    Returns:
        Formatted evaluation prompt
    """
    prompt = f"""You are an expert evaluator. Evaluate the following response based on the given criteria.

Criteria: {criteria}

Question: {question}

"""
    
    if context:
        prompt += f"""Context Provided:
{context}

"""
    
    prompt += f"""Response to Evaluate:
{response}

Provide your evaluation as a JSON object with the following structure:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<brief explanation>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"]
}}
"""
    
    return prompt


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
