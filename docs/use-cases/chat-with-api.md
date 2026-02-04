---
layout: default
title: Chat with API Evaluation
parent: Use Cases
nav_order: 2
---

# Chat with API Evaluation

Evaluating conversational AI systems that interact through APIs requires assessing conversation quality, response appropriateness, and API reliability.

## Overview

Chat API evaluation covers:
1. **Response Quality** - Relevance, coherence, and helpfulness
2. **Conversation Flow** - Multi-turn consistency and context handling
3. **Safety & Compliance** - Content moderation and policy adherence

---

## Key Metrics

### Response Quality Metrics

| Metric | Description | Target Range |
|--------|-------------|--------------|
| **Relevance** | Response addresses the user query | > 0.85 |
| **Coherence** | Logical flow and readability | > 0.9 |
| **Helpfulness** | Provides actionable/useful information | > 0.8 |
| **Conciseness** | Appropriate response length | Context-dependent |

### Conversation Metrics

| Metric | Description | Target Range |
|--------|-------------|--------------|
| **Context Retention** | Maintains conversation history | > 0.9 |
| **Turn Consistency** | Consistent persona/knowledge | > 0.85 |
| **Resolution Rate** | Successfully resolves user intent | > 0.75 |

### Safety Metrics

| Metric | Description | Target Range |
|--------|-------------|--------------|
| **Toxicity Score** | Absence of harmful content | < 0.05 |
| **Bias Detection** | Fairness across demographics | < 0.1 |
| **PII Handling** | Proper handling of sensitive data | 100% |

---

## Evaluation Framework

### Multi-Turn Conversation Evaluation

```
┌─────────────────────────────────────────────────────────────┐
│                  Conversation Flow                           │
├─────────────────────────────────────────────────────────────┤
│  Turn 1  │  Turn 2  │  Turn 3  │  Turn 4  │  Turn N        │
│  ────────┼──────────┼──────────┼──────────┼────────         │
│  Intent  │  Follow  │  Clarify │  Resolve │  Close          │
│  Detect  │  Up      │          │          │                 │
├─────────────────────────────────────────────────────────────┤
│  Metrics: Context Retention, Consistency, Goal Completion   │
└─────────────────────────────────────────────────────────────┘
```

---

## Test Dataset Structure

```json
{
  "conversation_id": "conv_001",
  "turns": [
    {
      "role": "user",
      "content": "I need help with my order"
    },
    {
      "role": "assistant", 
      "content": "I'd be happy to help! Could you provide your order number?",
      "expected_intent": "request_info"
    },
    {
      "role": "user",
      "content": "It's ORDER-12345"
    },
    {
      "role": "assistant",
      "content": "Thank you! I found your order...",
      "expected_intent": "provide_info"
    }
  ],
  "expected_resolution": true,
  "category": "order_support"
}
```

---

## Evaluation Types

### 1. Automated Evaluation

- **LLM-as-Judge**: Use another LLM to evaluate responses
- **Embedding Similarity**: Compare to reference responses
- **Keyword Matching**: Check for required information

### 2. Human Evaluation

- **A/B Testing**: Compare model versions
- **Likert Scale Ratings**: Human quality scores
- **Error Categorization**: Classify failure modes

---

## Best Practices

### ✅ Do

- Test with diverse conversation scenarios
- Include multi-turn context dependencies
- Evaluate edge cases (empty inputs, very long inputs)
- Measure response latency and token usage

### ❌ Don't

- Test only single-turn interactions
- Ignore conversation history handling
- Skip safety and bias testing
- Overlook API error handling

---

## Code Examples

- [Chat API Evaluation](../../examples/chat-api/chat_evaluation.py) - Basic chat evaluation
- [Multi-Turn Testing](../../examples/chat-api/multi_turn_eval.py) - Conversation flow testing

---

**← [RAG Application](./rag-application.md)** | **[Text Summarization →](./text-summarization.md)**
