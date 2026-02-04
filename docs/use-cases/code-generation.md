---
layout: default
title: Code Generation Evaluation
parent: Use Cases
nav_order: 4
---

# Code Generation Evaluation

Evaluating code generation systems requires assessing functional correctness, code quality, and security.

## Overview

Code generation evaluation covers:
1. **Functional Correctness** - Does the code work?
2. **Code Quality** - Readability, efficiency, best practices
3. **Security** - Vulnerability detection

---

## Key Metrics

### Correctness Metrics

| Metric | Description | Target Range |
|--------|-------------|--------------|
| **Pass@k** | Probability of correct solution in k attempts | > 0.7 (k=1) |
| **Test Pass Rate** | Percentage of test cases passed | > 0.9 |
| **Compilation Success** | Code compiles without errors | > 0.95 |

### Quality Metrics

| Metric | Description | Target Range |
|--------|-------------|--------------|
| **Code Similarity** | Similarity to reference solution | > 0.7 |
| **Cyclomatic Complexity** | Code complexity measure | < 10 |
| **Lint Score** | Adherence to style guides | > 0.9 |

---

## Evaluation Dataset Structure

```json
{
  "prompt": "Write a function to calculate fibonacci numbers",
  "test_cases": [
    {"input": [0], "expected": 0},
    {"input": [1], "expected": 1},
    {"input": [10], "expected": 55}
  ],
  "reference_solution": "def fibonacci(n): ...",
  "language": "python"
}
```

---

## Code Examples

- [Code Eval Metrics](../../examples/code-generation/code_evaluation.py) - Pass@k and test execution

---

**← [Text Summarization](./text-summarization.md)** | **[Home](../index.md)**
