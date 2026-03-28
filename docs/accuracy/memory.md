---
layout: default
title: "Memory"
parent: "2. Accuracy"
nav_order: 4
permalink: /accuracy/memory/
---

# Memory

Evaluate whether the system correctly stores, recalls, and updates information across conversations.

| Metric | What It Evaluates | What Goes Wrong |
|--------|-------------------|-----------------|
| **Memory Correctness** | Did the system store exactly what the user said? | Wrong value stored, hallucinated details added, key information missing |
| **Recall Relevance** | When the system retrieves memory, is it the right memory at the right time? | Irrelevant memories surface, useful memories ignored |
| **Update Correctness** | When the user corrects something, does memory update properly? | Old value kept alongside new, update ignored, unrelated memory deleted |

---

## Memory Correctness

**Manual Examples:**

| User Input | Stored Memory | Expected | Result |
|---|---|---|---|
| "My favorite color is blue." | `favorite_color = blue` | `favorite_color = blue` | ✅ PASS — stored exactly what user said |
| "My favorite color is blue." | `favorite_color = green` | `favorite_color = blue` | ❌ FAIL — wrong value stored |
| "My project deadline is May 5." | `deadline = May 5, 2024 at 5:00 PM` | `deadline = May 5` | ❌ FAIL — hallucinated time and year user never said |
| "I have 2 kids, ages 5 and 8." | `kids_count = 2` (ages missing) | `kids_count = 2, kids_ages = [5, 8]` | ❌ FAIL — missing key information |

---

## Recall Relevance

**Manual Examples:**

| Current Query | Retrieved Memory | Full Memory Store | Result |
|---|---|---|---|
| "Suggest outfits I might like." | `favorite_color = blue` | favorite_color, favorite_food, city | ✅ PASS — relevant memory retrieved |
| "Suggest outfits I might like." | `favorite_food = pizza` | favorite_color, favorite_food, city | ❌ FAIL — irrelevant memory; missed favorite_color |

---

## Update Correctness

**Manual Examples:**

| User Says | Memory Before | Memory After | Result |
|---|---|---|---|
| "Change my favorite color to red." | `{favorite_color: blue, city: Karachi}` | `{favorite_color: red, city: Karachi}` | ✅ PASS — updated correctly, unrelated key preserved |
| "Change my favorite color to red." | `{favorite_color: blue, city: Karachi}` | `{favorite_color: blue, favorite_color_new: red, city: Karachi}` | ❌ FAIL — kept both old and new |
| "Change my favorite color to red." | `{favorite_color: blue, city: Karachi}` | `{favorite_color: blue, city: Karachi}` | ❌ FAIL — ignored the update entirely |
| "Change my favorite color to red." | `{favorite_color: blue, city: Karachi}` | `{favorite_color: red}` | ❌ FAIL — updated correctly but deleted unrelated `city` |

---

## Code

**Code:** [`examples/accuracy/memory/memory_context_evaluator.py`](https://github.com/WisamulHaque/emumba_llm_evaluation/blob/main/examples/accuracy/memory/memory_context_evaluator.py)

```python
from examples.accuracy.memory.memory_context_evaluator import (
    evaluate_memory_correctness,
    evaluate_recall_relevance,
    evaluate_update_correctness,
)

# 1. Memory Correctness — LLM-as-judge
result = evaluate_memory_correctness(
    judge=judge,
    user_input="My favorite color is blue.",
    stored_memory={"favorite_color": "green"},
    expected_memory={"favorite_color": "blue"},
)
# {"passed": False, "reason": "...", "issues": ["Wrong value for 'favorite_color': ..."]}

# 2. Recall Relevance — LLM-as-judge
result = evaluate_recall_relevance(
    judge=judge,
    query="Suggest outfits I might like.",
    retrieved_memories=[{"key": "favorite_color", "value": "blue"}],
    all_memories=[
        {"key": "favorite_color", "value": "blue"},
        {"key": "favorite_food", "value": "pizza"},
        {"key": "city", "value": "Karachi"},
    ],
)
# {"passed": True, "reason": "...", "relevant_memories": [...], "irrelevant_memories": [], "missed_memories": []}

# 3. Update Correctness — LLM-as-judge
result = evaluate_update_correctness(
    judge=judge,
    memory_before={"favorite_color": "blue", "city": "Karachi"},
    update_instruction="Actually, change my favorite color to red.",
    memory_after={"favorite_color": "red", "city": "Karachi"},
    expected_memory_after={"favorite_color": "red", "city": "Karachi"},
)
# {"passed": True, "reason": "...", "issues": []}
```
