---
layout: default
title: "Context Sourcing"
parent: "2. Accuracy"
nav_order: 2
permalink: /accuracy/context-sourcing/
---

# Context Sourcing

Evaluate how well the system retrieves and provides the right context for generating answers.

---

## RAG Sourcing

When evidence comes from vector/keyword indices, measure retrieval quality with two complementary metrics:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Recall** | Of all the chunks that *should* have been retrieved, how many were actually found? | Low recall = the LLM is missing critical information it needs to answer correctly |
| **Precision** | Of all the chunks the system *did* retrieve, how many are actually relevant? | Low precision = noise and off-topic chunks waste the context window and can confuse the LLM |

**Manual Examples:**

| Query | Ground Truth (3 chunks) | Retrieved (2 chunks) | Recall | Precision | Verdict |
|---|---|---|---|---|---|
| What are the health insurance benefits? | ① Eligible after 90 days ② Company covers 80% of premium ③ Dental/vision available for $45/mo | ① Eligible after 90 days + ❌ Cafeteria hours (off-topic) | 1/3 = 0.33 | 1/2 = 0.50 | ❌ FAIL — missed 2 of 3 expected chunks and retrieved noise |
| How do I request annual leave? | ① Submit through self-service portal ② Manager approval 5 days in advance | ① Submit through portal ② Manager approval 5 days in advance | 2/2 = 1.0 | 2/2 = 1.0 | ✅ PASS — all expected chunks found, no noise |
| How do I request annual leave? | ① Submit through self-service portal ② Manager approval 5 days in advance | ① Submit through portal ② Manager approval + ❌ Company founding date + ❌ Parking info | 2/2 = 1.0 | 2/4 = 0.50 | ❌ FAIL — all info found but half the retrieved chunks are noise |

**Code:** [`examples/accuracy/context_sourcing/rag_retrieval_evaluator.py`](https://github.com/WisamulHaque/emumba_llm_evaluation/blob/main/examples/accuracy/context_sourcing/rag_retrieval_evaluator.py)

```python
from examples.accuracy.context_sourcing.rag_retrieval_evaluator import (
    evaluate_recall,
    evaluate_precision,
    evaluate_retrieval,   # runs both in one call
)

# Recall only — did we find all the expected chunks?
recall_result = evaluate_recall(
    retrieved_chunks=["Eligible after 90 days...", "Cafeteria hours..."],
    ground_truth_chunks=["Eligible after 90 days...", "Company covers 80%...", "Dental/vision for $45..."]
)
print(recall_result)  # {"passed": False, "recall": 0.333, "missing_chunks": [...]}

# Precision only — are the retrieved chunks relevant?
precision_result = evaluate_precision(
    retrieved_chunks=["Submit through portal...", "Manager approval...", "Company founded in 2003...", "Parking info..."],
    ground_truth_chunks=["Submit through portal...", "Manager approval..."]
)
print(precision_result)  # {"passed": False, "precision": 0.5, "noise_chunks": [...]}

# Combined — both recall and precision in one call
result = evaluate_retrieval(
    retrieved_chunks=["Submit through portal...", "Manager approval..."],
    ground_truth_chunks=["Submit through portal...", "Manager approval..."]
)
print(result)  # {"passed": True, "recall": 1.0, "precision": 1.0}
```

---

## Non-RAG Sourcing (Tools / DB / API)

Not all systems use RAG. Many fetch data from APIs, databases, or programmatic tools instead. The exact failure modes and evaluation approach depend on your architecture, but three common areas to evaluate are:

| Area | What It Evaluates | Why It Matters |
|------|-------------------|----------------|
| **API Selection** | Did the system pick the correct endpoint? | Wrong endpoint = entirely wrong data, no downstream fix possible |
| **Parameter Accuracy** | Did it pass the correct query params / filters? | Right endpoint + wrong parameters = wrong results (e.g., wrong date, wrong passenger count) |
| **Query Generation** | Did it produce a correct database query (SQL/NoSQL)? | Wrong JOINs, missing WHERE clauses, or bad aggregations silently return incorrect data |

> **Note:** These are starting examples. Non-RAG sourcing can vary widely — GraphQL queries, gRPC calls, multi-step tool chains, etc. For simple cases, API Selection and Parameter checks are straightforward ground-truth comparisons (no LLM needed). For more complex systems with ambiguous routing logic, multiple valid endpoints, or dynamic schemas, you can substitute an LLM-as-judge approach instead.

**Manual Examples — API Selection:**

| Query | Selected | Ground Truth | Result |
|---|---|---|---|
| Find flights from Karachi to Dubai next Friday. | flight_search | flight_search | ✅ PASS — correct endpoint |
| Find flights from Karachi to Dubai next Friday. | weather | flight_search | ❌ FAIL — wrong endpoint entirely |

**Manual Examples — Parameter Accuracy:**

| Query | Generated Params | Ground Truth Params | Result |
|---|---|---|---|
| Economy flights KHI→DXB on March 28 for 2 passengers. | `{origin: KHI, dest: DXB, date: 2026-03-28, pax: 2, class: economy}` | Same | ✅ PASS — all params correct |
| Flights LHE→IST for 3 passengers on April 5. | `{origin: LHE, dest: IST, date: 2026-04-05, pax: 1, class: business}` | `{origin: LHE, dest: IST, date: 2026-04-05, pax: 3}` | ❌ FAIL — wrong passenger count, invented cabin class |

**Manual Examples — Query Generation:**

| Query | Generated SQL | Result |
|---|---|---|
| Show all pending orders from customers in Pakistan. | `SELECT ... FROM orders o JOIN customers c ... WHERE c.country = 'Pakistan' AND o.status = 'pending'` | ✅ PASS — both filters present |
| Show all pending orders from customers in Pakistan. | `SELECT ... FROM orders o JOIN customers c ... WHERE c.country = 'Pakistan'` | ❌ FAIL — missing status filter, returns all orders instead of just pending |

**Code:** [`examples/accuracy/context_sourcing/non_rag_sourcing_evaluator.py`](https://github.com/WisamulHaque/emumba_llm_evaluation/blob/main/examples/accuracy/context_sourcing/non_rag_sourcing_evaluator.py)

```python
from examples.accuracy.context_sourcing.non_rag_sourcing_evaluator import (
    evaluate_api_selection,
    evaluate_parameter_accuracy,
    evaluate_query_generation,
)

# 1. API Selection — simple ground-truth comparison (no LLM needed)
result = evaluate_api_selection(
    selected_api="flight_search",
    ground_truth_api="flight_search",
)
# {"passed": True, "reason": "Correct endpoint selected: 'flight_search'."}

# 2. Parameter Accuracy — simple dict comparison (no LLM needed)
result = evaluate_parameter_accuracy(
    generated_params={"origin": "KHI", "destination": "DXB", "departure_date": "2026-03-28", "passengers": 2},
    ground_truth_params={"origin": "KHI", "destination": "DXB", "departure_date": "2026-03-28", "passengers": 2},
)
# {"passed": True, "reason": "All parameters match the ground truth.", "issues": []}

# 3. Query Generation — LLM-as-judge (user query + generated SQL)
result = evaluate_query_generation(
    judge=judge,
    query="Show all pending orders from customers in Pakistan",
    generated_query="SELECT o.order_id, o.total_amount FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE c.country = 'Pakistan' AND o.status = 'pending'",
)
# {"passed": True, "reason": "...", "issues": []}
```
