"""
RAG Application Evaluation - Basic Example

This script demonstrates how to evaluate a RAG (Retrieval-Augmented Generation)
application using common metrics like faithfulness and answer relevance.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json


@dataclass
class RAGEvaluationSample:
    """Single evaluation sample for RAG systems."""
    question: str
    ground_truth: str
    contexts: List[str]
    generated_answer: str
    metadata: Dict[str, Any] = None


class RAGEvaluator:
    """
    Basic RAG Evaluation class.
    
    Evaluates RAG systems on:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevance: Does the answer address the question?
    - Context Precision: Is the retrieved context relevant?
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the evaluator.
        
        Args:
            llm_client: Optional LLM client for LLM-as-judge evaluation
        """
        self.llm_client = llm_client
    
    def evaluate_faithfulness(self, sample: RAGEvaluationSample) -> Dict[str, Any]:
        """
        Evaluate if the generated answer is faithful to the provided context.
        
        Args:
            sample: RAGEvaluationSample containing question, contexts, and answer
            
        Returns:
            Dictionary with faithfulness score and details
        """
        # Simple heuristic: Check if key terms from answer appear in context
        context_text = " ".join(sample.contexts).lower()
        answer_words = set(sample.generated_answer.lower().split())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'it', 'this', 'that', 'these', 'those', 'and', 'or', 'but'}
        
        content_words = answer_words - stop_words
        
        if not content_words:
            return {"score": 1.0, "grounded_terms": [], "ungrounded_terms": []}
        
        grounded = [word for word in content_words if word in context_text]
        ungrounded = [word for word in content_words if word not in context_text]
        
        score = len(grounded) / len(content_words) if content_words else 1.0
        
        return {
            "score": round(score, 3),
            "grounded_terms": grounded,
            "ungrounded_terms": ungrounded
        }
    
    def evaluate_answer_relevance(self, sample: RAGEvaluationSample) -> Dict[str, Any]:
        """
        Evaluate if the generated answer is relevant to the question.
        
        Args:
            sample: RAGEvaluationSample containing question and answer
            
        Returns:
            Dictionary with relevance score and details
        """
        # Simple heuristic: Check question-answer term overlap
        question_words = set(sample.question.lower().split())
        answer_words = set(sample.generated_answer.lower().split())
        
        # Remove stop words
        stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are',
                      'the', 'a', 'an', 'do', 'does', '?', 'can', 'could'}
        
        question_content = question_words - stop_words
        
        if not question_content:
            return {"score": 1.0, "matched_terms": []}
        
        matched = question_content.intersection(answer_words)
        score = len(matched) / len(question_content)
        
        return {
            "score": round(score, 3),
            "matched_terms": list(matched)
        }
    
    def evaluate_context_precision(self, sample: RAGEvaluationSample) -> Dict[str, Any]:
        """
        Evaluate if the retrieved contexts are relevant to the question.
        
        Args:
            sample: RAGEvaluationSample containing question and contexts
            
        Returns:
            Dictionary with precision score per context
        """
        question_words = set(sample.question.lower().split())
        context_scores = []
        
        for i, context in enumerate(sample.contexts):
            context_words = set(context.lower().split())
            overlap = len(question_words.intersection(context_words))
            score = overlap / len(question_words) if question_words else 0
            context_scores.append({
                "context_index": i,
                "score": round(score, 3)
            })
        
        avg_score = sum(c["score"] for c in context_scores) / len(context_scores) if context_scores else 0
        
        return {
            "average_score": round(avg_score, 3),
            "per_context_scores": context_scores
        }
    
    def evaluate(self, sample: RAGEvaluationSample) -> Dict[str, Any]:
        """
        Run all evaluations on a sample.
        
        Args:
            sample: RAGEvaluationSample to evaluate
            
        Returns:
            Dictionary with all evaluation results
        """
        return {
            "faithfulness": self.evaluate_faithfulness(sample),
            "answer_relevance": self.evaluate_answer_relevance(sample),
            "context_precision": self.evaluate_context_precision(sample)
        }


def main():
    """Example usage of RAG evaluator."""
    
    # Create sample data
    sample = RAGEvaluationSample(
        question="What is the capital of France?",
        ground_truth="The capital of France is Paris.",
        contexts=[
            "Paris is the capital and most populous city of France.",
            "France is a country located in Western Europe.",
            "The Eiffel Tower is a famous landmark in Paris."
        ],
        generated_answer="The capital of France is Paris, which is also its largest city."
    )
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate(sample)
    
    # Print results
    print("=" * 60)
    print("RAG Evaluation Results")
    print("=" * 60)
    print(f"\nQuestion: {sample.question}")
    print(f"Generated Answer: {sample.generated_answer}")
    print(f"\nEvaluation Results:")
    print(json.dumps(results, indent=2))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary Scores")
    print("=" * 60)
    print(f"Faithfulness:       {results['faithfulness']['score']:.2f}")
    print(f"Answer Relevance:   {results['answer_relevance']['score']:.2f}")
    print(f"Context Precision:  {results['context_precision']['average_score']:.2f}")


if __name__ == "__main__":
    main()
