---
layout: default
title: Use Cases
nav_order: 2
has_children: true
permalink: /use-cases/
---

# Use Cases

This section provides detailed evaluation guidelines for specific LLM application types.

## Available Guides

<div class="use-case-grid">

### [RAG Applications](./rag-application.md)
Evaluate Retrieval-Augmented Generation systems for faithfulness, relevance, and retrieval quality.

### [Chat with API](./chat-with-api.md)
Assess conversational AI systems for response quality, context handling, and safety.

### [Text Summarization](./text-summarization.md)
Measure summarization quality using ROUGE, BERTScore, and factual consistency metrics.

### [Code Generation](./code-generation.md)
Evaluate code generation with Pass@k, test execution, and code quality metrics.

</div>

---

## Choosing the Right Evaluation Approach

| If you're building... | Focus on... | Key Metrics |
|----------------------|-------------|-------------|
| **RAG System** | Retrieval + Generation | Faithfulness, Context Recall |
| **Chatbot** | Conversation Flow | Coherence, Safety |
| **Summarizer** | Content Coverage | ROUGE, Factual Consistency |
| **Code Assistant** | Correctness | Pass@k, Test Pass Rate |

---

Select a use case from the navigation to get started.
