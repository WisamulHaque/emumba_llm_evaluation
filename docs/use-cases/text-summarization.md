---
layout: default
title: Text Summarization Evaluation
parent: Use Cases
nav_order: 3
---

# Text Summarization Evaluation

Evaluating text summarization systems requires assessing content coverage, factual accuracy, and linguistic quality.

## Overview

Text summarization evaluation focuses on:
1. **Content Quality** - Coverage of key information
2. **Factual Accuracy** - Faithfulness to source
3. **Linguistic Quality** - Readability and coherence

---

## Key Metrics

### Content Metrics

| Metric | Description | Target Range |
|--------|-------------|--------------|
| **ROUGE-1** | Unigram overlap with reference | > 0.45 |
| **ROUGE-2** | Bigram overlap with reference | > 0.20 |
| **ROUGE-L** | Longest common subsequence | > 0.40 |
| **BERTScore** | Semantic similarity | > 0.85 |

### Factual Accuracy Metrics

| Metric | Description | Target Range |
|--------|-------------|--------------|
| **Factual Consistency** | Claims supported by source | > 0.95 |
| **Hallucination Rate** | Unsupported information | < 0.05 |
| **Entity Accuracy** | Correct entity mentions | > 0.98 |

---

## Evaluation Dataset Structure

```json
{
  "document": "Full source document text...",
  "reference_summary": "Human-written reference summary...",
  "key_points": [
    "Main point 1",
    "Main point 2"
  ],
  "metadata": {
    "domain": "news",
    "length": "long"
  }
}
```

---

## Code Examples

- [Summarization Metrics](../../examples/summarization/summary_evaluation.py) - ROUGE and BERTScore evaluation

---

**← [Chat with API](./chat-with-api.md)** | **[Code Generation →](./code-generation.md)**
