---
layout: default
title: RAG Application Evaluation
parent: Use Cases
nav_order: 1
permalink: /use-cases/rag-application/
---

# RAG Application Evaluation

Retrieval-Augmented Generation (RAG) applications combine document retrieval with LLM generation. Evaluating RAG systems requires assessing both the retrieval and generation components.

## Overview

RAG evaluation focuses on three main areas:
1. **Retrieval Quality** - How well does the system find relevant documents?
2. **Generation Quality** - How well does the LLM use retrieved context?
3. **End-to-End Performance** - Overall system effectiveness

---

## Key Metrics

### Retrieval Metrics

| Metric | Description | Target Range |
|--------|-------------|--------------|
| **Precision@K** | Proportion of relevant documents in top K results | > 0.8 |
| **Recall@K** | Proportion of relevant documents retrieved | > 0.7 |
| **MRR** | Mean Reciprocal Rank of first relevant result | > 0.85 |
| **NDCG** | Normalized Discounted Cumulative Gain | > 0.75 |

### Generation Metrics

| Metric | Description | Target Range |
|--------|-------------|--------------|
| **Faithfulness** | Response grounded in retrieved context | > 0.9 |
| **Answer Relevance** | Response addresses the query | > 0.85 |
| **Context Precision** | Relevance of retrieved context | > 0.8 |
| **Context Recall** | Coverage of ground truth in context | > 0.75 |

---

## Evaluation Framework

### 1. Component-Level Evaluation

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                              │
├──────────────┬──────────────────────┬───────────────────────┤
│   Retriever  │      Reranker        │      Generator        │
│              │     (optional)       │                       │
├──────────────┼──────────────────────┼───────────────────────┤
│ Precision@K  │  Rerank Accuracy     │  Faithfulness         │
│ Recall@K     │  Position Shift      │  Answer Relevance     │
│ MRR          │  Latency             │  Coherence            │
└──────────────┴──────────────────────┴───────────────────────┘
```

### 2. End-to-End Evaluation

Evaluate the complete pipeline with:
- **Answer Correctness**: Compare against ground truth answers
- **Hallucination Rate**: Detect unsupported claims
- **Response Latency**: Measure total response time

---

## Evaluation Dataset Structure

Your evaluation dataset should include:

```json
{
  "question": "What is the capital of France?",
  "ground_truth": "The capital of France is Paris.",
  "contexts": [
    "Paris is the capital and most populous city of France.",
    "France is a country in Western Europe."
  ],
  "metadata": {
    "category": "geography",
    "difficulty": "easy"
  }
}
```

---

## Best Practices

### ✅ Do

- Create diverse test sets covering edge cases
- Include adversarial queries to test robustness
- Measure latency alongside quality metrics
- Use human evaluation for subjective quality

### ❌ Don't

- Rely solely on automated metrics
- Test only on in-distribution data
- Ignore context window limitations
- Skip failure case analysis

---

## Code Examples

Check out our code examples for RAG evaluation:

- [Basic RAG Evaluation](../../examples/rag/basic_evaluation.py) - Simple evaluation script
- [RAGAS Integration](../../examples/rag/ragas_evaluation.py) - Using RAGAS framework

---

## Tools & Frameworks

| Tool | Description | Link |
|------|-------------|------|
| **RAGAS** | RAG Assessment framework | [GitHub](https://github.com/explodinggradients/ragas) |
| **LangChain Eval** | LangChain evaluation tools | [Docs](https://python.langchain.com/docs/guides/evaluation/) |
| **DeepEval** | LLM evaluation framework | [GitHub](https://github.com/confident-ai/deepeval) |

---

**← [Home](../)** | **[Chat with API →](./chat-with-api)**
