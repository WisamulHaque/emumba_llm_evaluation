"""
Chat API Evaluation

This script demonstrates how to evaluate conversational AI responses
for quality, relevance, and safety.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class ResponseQuality(Enum):
    """Quality ratings for chat responses."""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    UNACCEPTABLE = 1


@dataclass
class ChatMessage:
    """Single message in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSample:
    """A complete conversation for evaluation."""
    conversation_id: str
    messages: List[ChatMessage]
    expected_intent: Optional[str] = None
    expected_resolution: bool = True


class ChatEvaluator:
    """
    Evaluator for chat/conversational AI systems.
    
    Evaluates:
    - Response relevance
    - Conversation coherence
    - Context retention
    - Safety/toxicity
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the chat evaluator.
        
        Args:
            llm_client: Optional LLM client for advanced evaluation
        """
        self.llm_client = llm_client
        
        # Simple toxicity word list (in production, use a proper classifier)
        self.toxic_patterns = [
            'hate', 'kill', 'violent', 'attack', 'threat'
        ]
    
    def evaluate_relevance(
        self, 
        user_message: str, 
        assistant_response: str
    ) -> Dict[str, Any]:
        """
        Evaluate if the response is relevant to the user's message.
        
        Args:
            user_message: The user's input
            assistant_response: The assistant's response
            
        Returns:
            Dictionary with relevance score and details
        """
        # Simple keyword overlap heuristic
        user_words = set(user_message.lower().split())
        response_words = set(assistant_response.lower().split())
        
        # Remove common words
        stop_words = {'i', 'me', 'my', 'the', 'a', 'an', 'is', 'are', 'was',
                      'be', 'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for',
                      'with', 'can', 'you', 'your', 'please', 'help', 'need'}
        
        user_content = user_words - stop_words
        
        if not user_content:
            return {"score": 1.0, "matched_terms": []}
        
        matched = user_content.intersection(response_words)
        score = len(matched) / len(user_content)
        
        return {
            "score": round(score, 3),
            "matched_terms": list(matched),
            "user_key_terms": list(user_content)
        }
    
    def evaluate_coherence(self, response: str) -> Dict[str, Any]:
        """
        Evaluate the coherence and readability of a response.
        
        Args:
            response: The assistant's response
            
        Returns:
            Dictionary with coherence metrics
        """
        sentences = response.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Basic metrics
        word_count = len(response.split())
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Score based on reasonable sentence length (10-20 words is ideal)
        if 10 <= avg_sentence_length <= 25:
            length_score = 1.0
        elif 5 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 35:
            length_score = 0.7
        else:
            length_score = 0.4
        
        return {
            "score": length_score,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 1)
        }
    
    def evaluate_context_retention(
        self, 
        conversation: ConversationSample
    ) -> Dict[str, Any]:
        """
        Evaluate if the assistant maintains context across turns.
        
        Args:
            conversation: Full conversation to evaluate
            
        Returns:
            Dictionary with context retention score
        """
        if len(conversation.messages) < 3:
            return {"score": 1.0, "note": "Too few turns to evaluate"}
        
        # Check if later responses reference earlier context
        all_user_content = set()
        retention_scores = []
        
        for i, msg in enumerate(conversation.messages):
            if msg.role == 'user':
                # Collect user terms
                words = set(msg.content.lower().split())
                all_user_content.update(words)
            elif msg.role == 'assistant' and i > 0:
                # Check if assistant references previous context
                response_words = set(msg.content.lower().split())
                if all_user_content:
                    overlap = len(all_user_content.intersection(response_words))
                    score = min(overlap / 5, 1.0)  # Cap at 5 matching terms
                    retention_scores.append(score)
        
        avg_score = sum(retention_scores) / len(retention_scores) if retention_scores else 1.0
        
        return {
            "score": round(avg_score, 3),
            "turns_evaluated": len(retention_scores)
        }
    
    def evaluate_safety(self, response: str) -> Dict[str, Any]:
        """
        Evaluate the safety of a response (basic toxicity check).
        
        Args:
            response: The assistant's response
            
        Returns:
            Dictionary with safety score and flagged content
        """
        response_lower = response.lower()
        flagged = [pattern for pattern in self.toxic_patterns 
                   if pattern in response_lower]
        
        # Score: 1.0 = safe, 0.0 = toxic
        score = 1.0 if not flagged else max(0, 1 - len(flagged) * 0.25)
        
        return {
            "score": round(score, 3),
            "is_safe": len(flagged) == 0,
            "flagged_patterns": flagged
        }
    
    def evaluate_response(
        self, 
        user_message: str, 
        assistant_response: str
    ) -> Dict[str, Any]:
        """
        Run all evaluations on a single response.
        
        Args:
            user_message: The user's input
            assistant_response: The assistant's response
            
        Returns:
            Dictionary with all evaluation results
        """
        return {
            "relevance": self.evaluate_relevance(user_message, assistant_response),
            "coherence": self.evaluate_coherence(assistant_response),
            "safety": self.evaluate_safety(assistant_response)
        }
    
    def evaluate_conversation(
        self, 
        conversation: ConversationSample
    ) -> Dict[str, Any]:
        """
        Evaluate a complete conversation.
        
        Args:
            conversation: Full conversation to evaluate
            
        Returns:
            Dictionary with conversation-level metrics
        """
        turn_evaluations = []
        
        for i in range(0, len(conversation.messages) - 1, 2):
            if (i + 1 < len(conversation.messages) and 
                conversation.messages[i].role == 'user' and
                conversation.messages[i + 1].role == 'assistant'):
                
                user_msg = conversation.messages[i].content
                asst_msg = conversation.messages[i + 1].content
                
                turn_eval = self.evaluate_response(user_msg, asst_msg)
                turn_eval["turn_number"] = i // 2 + 1
                turn_evaluations.append(turn_eval)
        
        # Aggregate scores
        if turn_evaluations:
            avg_relevance = sum(t["relevance"]["score"] for t in turn_evaluations) / len(turn_evaluations)
            avg_coherence = sum(t["coherence"]["score"] for t in turn_evaluations) / len(turn_evaluations)
            avg_safety = sum(t["safety"]["score"] for t in turn_evaluations) / len(turn_evaluations)
        else:
            avg_relevance = avg_coherence = avg_safety = 0
        
        context_retention = self.evaluate_context_retention(conversation)
        
        return {
            "conversation_id": conversation.conversation_id,
            "total_turns": len(turn_evaluations),
            "aggregate_scores": {
                "relevance": round(avg_relevance, 3),
                "coherence": round(avg_coherence, 3),
                "safety": round(avg_safety, 3),
                "context_retention": context_retention["score"]
            },
            "turn_details": turn_evaluations,
            "context_retention_details": context_retention
        }


def main():
    """Example usage of chat evaluator."""
    
    # Create a sample conversation
    conversation = ConversationSample(
        conversation_id="conv_001",
        messages=[
            ChatMessage(role="user", content="I need help with my order"),
            ChatMessage(role="assistant", content="I'd be happy to help you with your order! Could you please provide your order number so I can look it up?"),
            ChatMessage(role="user", content="It's ORDER-12345"),
            ChatMessage(role="assistant", content="Thank you! I found your order ORDER-12345. It was placed on January 15th and is currently being shipped. Is there something specific about this order you need help with?"),
            ChatMessage(role="user", content="Yes, I want to change the delivery address"),
            ChatMessage(role="assistant", content="I understand you'd like to change the delivery address for your order. Since the order is already being shipped, let me check if we can still update the address. Could you provide the new delivery address you'd like to use?"),
        ]
    )
    
    # Initialize evaluator
    evaluator = ChatEvaluator()
    
    # Evaluate the conversation
    results = evaluator.evaluate_conversation(conversation)
    
    # Print results
    print("=" * 60)
    print("Chat Conversation Evaluation Results")
    print("=" * 60)
    print(f"\nConversation ID: {results['conversation_id']}")
    print(f"Total Turns: {results['total_turns']}")
    
    print("\nAggregate Scores:")
    print("-" * 40)
    for metric, score in results["aggregate_scores"].items():
        print(f"  {metric:20s}: {score:.3f}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
