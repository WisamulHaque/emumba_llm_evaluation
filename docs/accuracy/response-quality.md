---
layout: default
title: "Response Quality"
parent: "2. Accuracy"
nav_order: 1
permalink: /accuracy/response-quality/
---

# Response Quality

Evaluate the quality of the LLM's generated output regardless of where the context came from.

---

## Task Quality

Does the response actually do what the user asked for? This evaluates **intent fulfillment** — not whether the answer is factually correct (that's Factuality), but whether the response addresses every part of the user's request with a concrete, actionable answer.

**Manual Examples:**

| Query | Generated Response | Result |
|---|---|---|
| Show me all orders placed by customers from Pakistan that are still pending. | SQL query with correct JOIN and WHERE clauses for country and status | ✅ PASS — response delivers the requested artifact with both conditions |
| Show me all orders placed by customers from Pakistan that are still pending. | SQL query filters by country but omits the status filter | ❌ FAIL — misses part of the user's intent |
| Summarize last week's sales and list the top 3 products by revenue. | Provides the sales summary but omits the top 3 products list | ❌ FAIL — only half the request is fulfilled |
| What is the refund policy for cancelled flights? | "Refund policies can vary. I'd recommend checking the airline's website." | ❌ FAIL — vague deflection, doesn't answer the question |

> **Note:** This evaluator does NOT require ground truth. It judges the response solely against the user's query.

**Code:** [`examples/accuracy/response_quality/task_quality_evaluator.py`](https://github.com/WisamulHaque/emumba_llm_evaluation/blob/main/examples/accuracy/response_quality/task_quality_evaluator.py)

```python
from examples.accuracy.response_quality.task_quality_evaluator import evaluate_task_quality

result = evaluate_task_quality(
    judge=judge,
    query="Summarize last week's sales and list the top 3 products by revenue.",
    generated_response="Last week's total sales were $42,350 across 1,204 orders..."
)
print(result)  # {"passed": False, "reason": "...", "missed_intents": ["top 3 products by revenue"]}
```

---

## Instruction Following

Does the response follow the rules and constraints defined in the system prompt?

**Manual Examples:**

| System Prompt Rule | User Message | Response | Result |
|---|---|---|---|
| Never reveal whether a candidate's answer is correct | "I would reverse the linked list iteratively." | "Can you walk me through what happens to the pointers at each step?" | ✅ PASS — withholds judgment, asks follow-up |
| Never reveal whether a candidate's answer is correct | "I would sort the array first." | "That's exactly right! Great job!" | ❌ FAIL — explicitly confirms correctness |

**Code:** [`examples/accuracy/response_quality/instruction_following_evaluator.py`](https://github.com/WisamulHaque/emumba_llm_evaluation/blob/main/examples/accuracy/response_quality/instruction_following_evaluator.py)

```python
from examples.accuracy.response_quality.instruction_following_evaluator import evaluate_response_accuracy

result = evaluate_response_accuracy(
    judge=judge,
    system_prompt="Never reveal whether answers are correct...",
    user_message="I would reverse the linked list iteratively.",
    assistant_response="Can you walk me through your approach?"
)
```

---

## Factuality

Are the claims in the response factually correct when compared against a known ground truth answer? This requires a ground truth — use [Grounded Accuracy](../grounded-accuracy) if you want to check claims against retrieved context instead.

**Manual Examples:**

| Response Claim | Ground Truth | Result |
|---|---|---|
| "B-tree indexes are used for range queries" | Ground truth confirms B-tree for range queries | ✅ PASS — factually supported |
| "Hash indexes are used for range queries" | Ground truth says B-tree for ranges, hash for equality | ❌ FAIL — factual error |

**Code:** [`examples/accuracy/response_quality/factuality_evaluator.py`](https://github.com/WisamulHaque/emumba_llm_evaluation/blob/main/examples/accuracy/response_quality/factuality_evaluator.py)

```python
from examples.accuracy.response_quality.factuality_evaluator import evaluate_factuality

result = evaluate_factuality(
    judge=judge,
    query="How do database indexes improve query performance?",
    generated_response="Hash indexes are used for range queries...",
    ground_truth="B-tree indexes handle range queries; hash indexes handle equality..."
)
print(result)  # {"passed": False, "score": 0.5, "unsupported_claims": ["Hash indexes are used for range queries"]}
```

---

## Consistency

Does the model produce consistent outputs for similar queries? Repeated queries should not yield contradictory answers.

**Manual Examples:**

| Query | Response 1 | Response 2 | Result |
|---|---|---|---|
| What is a Python decorator? | "A function that wraps another function using @syntax" | "A function that modifies behavior of another function via @syntax" | ✅ PASS — consistent |
| Does Python support multi-threading for CPU tasks? | "Yes, threading is effective" | "No, the GIL prevents parallel execution" | ❌ FAIL — contradictory |

**Code:** [`examples/accuracy/response_quality/consistency_evaluator.py`](https://github.com/WisamulHaque/emumba_llm_evaluation/blob/main/examples/accuracy/response_quality/consistency_evaluator.py)

```python
from examples.accuracy.response_quality.consistency_evaluator import evaluate_consistency

result = evaluate_consistency(
    judge=judge,
    query="What is a Python decorator?",
    responses=["A function that wraps...", "A function that modifies..."]
)
```
