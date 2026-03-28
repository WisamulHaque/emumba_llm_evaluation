---
layout: default
title: "Grounded Accuracy"
parent: "2. Accuracy"
nav_order: 3
permalink: /accuracy/grounded-accuracy/
---

# Grounded Accuracy

Given whatever context was sourced, did the answer correctly use it?

| Metric | Description |
|--------|-------------|
| **Evidence Use Rate** | % of answers whose claims are supported by the provided context |
| **Citation Correctness** | % of citations that truly back the claim (no fabricated refs) |
| **Unsupported Claim Rate** | % of answers containing claims not in the provided context |
| **Hallucination Detection** | Identifying fabricated information absent from source material |

**Manual Examples:**

| Response | Context | Result |
|---|---|---|
| All claims reference provided documents with correct citations | Context contains all referenced information | ✅ PASS — fully grounded |
| Adds a claim about "write overhead" not present in context | Context only covers table scans and B-tree types | ❌ FAIL — unsupported claim / hallucination |

**Code:** [`examples/accuracy/grounded_accuracy/faithfulness_evaluator.py`](https://github.com/WisamulHaque/emumba_llm_evaluation/blob/main/examples/accuracy/grounded_accuracy/faithfulness_evaluator.py) · [`examples/accuracy/grounded_accuracy/citation_accuracy_evaluator.py`](https://github.com/WisamulHaque/emumba_llm_evaluation/blob/main/examples/accuracy/grounded_accuracy/citation_accuracy_evaluator.py)

```python
from examples.accuracy.grounded_accuracy.faithfulness_evaluator import evaluate_faithfulness
from examples.accuracy.grounded_accuracy.citation_accuracy_evaluator import evaluate_citation_accuracy

# Hallucination / unsupported claims (checks against provided context, not ground truth)
result = evaluate_faithfulness(
    judge=judge,
    query="How do database indexes work?",
    generated_response="Indexes speed up retrieval... they also add write overhead...",
    context="Document 1: Indexes speed up data retrieval by avoiding full table scans..."
)

# Citation correctness
result = evaluate_citation_accuracy(
    judge=judge,
    query="What are the health benefits of exercise?",
    generated_response="Exercise improves heart health [Doc A] and reduces stress [Doc B]...",
    citations=[
        {"source": "Doc A", "content": "Regular exercise strengthens the cardiovascular system..."},
        {"source": "Doc B", "content": "Physical activity reduces cortisol levels..."},
    ]
)
```
