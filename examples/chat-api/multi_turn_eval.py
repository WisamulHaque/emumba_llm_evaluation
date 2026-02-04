"""
Multi-Turn Conversation Evaluation

This script demonstrates how to evaluate multi-turn conversations
for consistency, goal completion, and conversation flow.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str
    content: str
    intent: Optional[str] = None
    entities: Optional[Dict[str, str]] = None


@dataclass
class MultiTurnTestCase:
    """Test case for multi-turn conversation evaluation."""
    test_id: str
    conversation: List[ConversationTurn]
    expected_intents: List[str]
    expected_goal_completed: bool
    category: str


class MultiTurnEvaluator:
    """
    Evaluator for multi-turn conversation quality.
    
    Focuses on:
    - Intent detection accuracy
    - Entity extraction consistency
    - Goal completion rate
    - Conversation flow quality
    """
    
    def __init__(self):
        """Initialize the multi-turn evaluator."""
        # Intent keywords mapping (simplified)
        self.intent_keywords = {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "order_inquiry": ["order", "purchase", "bought", "tracking"],
            "change_request": ["change", "update", "modify", "edit"],
            "cancellation": ["cancel", "stop", "remove", "delete"],
            "complaint": ["problem", "issue", "wrong", "broken", "bad"],
            "gratitude": ["thank", "thanks", "appreciate"],
            "farewell": ["bye", "goodbye", "see you", "take care"]
        }
    
    def detect_intent(self, message: str) -> str:
        """
        Detect the intent of a message.
        
        Args:
            message: User message
            
        Returns:
            Detected intent string
        """
        message_lower = message.lower()
        
        for intent, keywords in self.intent_keywords.items():
            if any(kw in message_lower for kw in keywords):
                return intent
        
        return "general_query"
    
    def evaluate_intent_accuracy(
        self, 
        test_case: MultiTurnTestCase
    ) -> Dict[str, Any]:
        """
        Evaluate intent detection accuracy across the conversation.
        
        Args:
            test_case: Multi-turn test case
            
        Returns:
            Dictionary with intent accuracy metrics
        """
        user_turns = [t for t in test_case.conversation if t.role == "user"]
        
        if len(user_turns) != len(test_case.expected_intents):
            return {
                "score": 0.0,
                "error": "Mismatch between turns and expected intents"
            }
        
        correct = 0
        intent_results = []
        
        for i, (turn, expected) in enumerate(zip(user_turns, test_case.expected_intents)):
            detected = self.detect_intent(turn.content)
            is_correct = detected == expected
            if is_correct:
                correct += 1
            
            intent_results.append({
                "turn": i + 1,
                "message": turn.content[:50] + "..." if len(turn.content) > 50 else turn.content,
                "expected": expected,
                "detected": detected,
                "correct": is_correct
            })
        
        accuracy = correct / len(user_turns) if user_turns else 0
        
        return {
            "score": round(accuracy, 3),
            "correct_count": correct,
            "total_turns": len(user_turns),
            "details": intent_results
        }
    
    def evaluate_conversation_flow(
        self, 
        test_case: MultiTurnTestCase
    ) -> Dict[str, Any]:
        """
        Evaluate the natural flow of the conversation.
        
        Args:
            test_case: Multi-turn test case
            
        Returns:
            Dictionary with flow quality metrics
        """
        flow_issues = []
        
        # Check for proper turn-taking
        for i in range(len(test_case.conversation) - 1):
            current = test_case.conversation[i]
            next_turn = test_case.conversation[i + 1]
            
            if current.role == next_turn.role:
                flow_issues.append({
                    "turn": i + 1,
                    "issue": "consecutive_same_role",
                    "description": f"Two consecutive {current.role} messages"
                })
        
        # Check for very short responses
        for i, turn in enumerate(test_case.conversation):
            if turn.role == "assistant" and len(turn.content.split()) < 3:
                flow_issues.append({
                    "turn": i + 1,
                    "issue": "too_short_response",
                    "description": "Assistant response is too brief"
                })
        
        # Calculate flow score
        total_checks = len(test_case.conversation)
        issues_count = len(flow_issues)
        score = max(0, 1 - (issues_count / total_checks))
        
        return {
            "score": round(score, 3),
            "issues_found": len(flow_issues),
            "issues": flow_issues
        }
    
    def evaluate_goal_completion(
        self, 
        test_case: MultiTurnTestCase
    ) -> Dict[str, Any]:
        """
        Evaluate if the conversation achieved its goal.
        
        Args:
            test_case: Multi-turn test case
            
        Returns:
            Dictionary with goal completion assessment
        """
        # Check for resolution indicators in the last assistant message
        assistant_messages = [t for t in test_case.conversation if t.role == "assistant"]
        
        if not assistant_messages:
            return {
                "goal_completed": False,
                "confidence": 0.0,
                "reason": "No assistant responses found"
            }
        
        last_response = assistant_messages[-1].content.lower()
        
        # Resolution indicators
        positive_indicators = [
            "completed", "done", "resolved", "updated", "changed",
            "confirmed", "processed", "successful", "all set"
        ]
        negative_indicators = [
            "cannot", "unable", "sorry", "unfortunately", "failed"
        ]
        
        has_positive = any(ind in last_response for ind in positive_indicators)
        has_negative = any(ind in last_response for ind in negative_indicators)
        
        if has_positive and not has_negative:
            completed = True
            confidence = 0.8
        elif has_negative:
            completed = False
            confidence = 0.7
        else:
            # Neutral - assume ongoing
            completed = False
            confidence = 0.5
        
        # Compare with expected
        matches_expected = completed == test_case.expected_goal_completed
        
        return {
            "goal_completed": completed,
            "expected": test_case.expected_goal_completed,
            "matches_expected": matches_expected,
            "confidence": confidence,
            "score": 1.0 if matches_expected else 0.0
        }
    
    def evaluate(self, test_case: MultiTurnTestCase) -> Dict[str, Any]:
        """
        Run complete evaluation on a multi-turn test case.
        
        Args:
            test_case: Multi-turn test case
            
        Returns:
            Dictionary with all evaluation results
        """
        intent_eval = self.evaluate_intent_accuracy(test_case)
        flow_eval = self.evaluate_conversation_flow(test_case)
        goal_eval = self.evaluate_goal_completion(test_case)
        
        # Calculate overall score
        overall = (
            intent_eval["score"] * 0.3 +
            flow_eval["score"] * 0.3 +
            goal_eval["score"] * 0.4
        )
        
        return {
            "test_id": test_case.test_id,
            "category": test_case.category,
            "overall_score": round(overall, 3),
            "intent_accuracy": intent_eval,
            "conversation_flow": flow_eval,
            "goal_completion": goal_eval
        }


def main():
    """Example usage of multi-turn evaluator."""
    
    # Create a test case
    test_case = MultiTurnTestCase(
        test_id="test_001",
        category="order_modification",
        conversation=[
            ConversationTurn(
                role="user",
                content="Hello, I need help with my order"
            ),
            ConversationTurn(
                role="assistant",
                content="Hello! I'd be happy to help you with your order. Could you please provide your order number?"
            ),
            ConversationTurn(
                role="user",
                content="My order number is ORD-12345"
            ),
            ConversationTurn(
                role="assistant",
                content="Thank you! I found your order ORD-12345. It contains 2 items and is scheduled for delivery on February 10th. What would you like to do with this order?"
            ),
            ConversationTurn(
                role="user",
                content="I want to change the delivery address"
            ),
            ConversationTurn(
                role="assistant",
                content="I can help you change the delivery address. Please provide the new address you'd like to use."
            ),
            ConversationTurn(
                role="user",
                content="123 New Street, City, State 12345"
            ),
            ConversationTurn(
                role="assistant",
                content="I've updated your delivery address to 123 New Street, City, State 12345. The change has been confirmed and your order will be delivered to this new address. Is there anything else I can help you with?"
            ),
            ConversationTurn(
                role="user",
                content="No, thank you for your help!"
            ),
            ConversationTurn(
                role="assistant",
                content="You're welcome! Thank you for choosing us. Have a great day!"
            ),
        ],
        expected_intents=[
            "greeting",
            "order_inquiry", 
            "change_request",
            "general_query",
            "gratitude"
        ],
        expected_goal_completed=True
    )
    
    # Initialize evaluator
    evaluator = MultiTurnEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate(test_case)
    
    # Print results
    print("=" * 60)
    print("Multi-Turn Conversation Evaluation Results")
    print("=" * 60)
    print(f"\nTest ID: {results['test_id']}")
    print(f"Category: {results['category']}")
    print(f"\nOverall Score: {results['overall_score']:.3f}")
    
    print("\n" + "-" * 40)
    print("Intent Accuracy:")
    print(f"  Score: {results['intent_accuracy']['score']:.3f}")
    print(f"  Correct: {results['intent_accuracy']['correct_count']}/{results['intent_accuracy']['total_turns']}")
    
    print("\n" + "-" * 40)
    print("Conversation Flow:")
    print(f"  Score: {results['conversation_flow']['score']:.3f}")
    print(f"  Issues Found: {results['conversation_flow']['issues_found']}")
    
    print("\n" + "-" * 40)
    print("Goal Completion:")
    print(f"  Completed: {results['goal_completion']['goal_completed']}")
    print(f"  Matches Expected: {results['goal_completion']['matches_expected']}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
