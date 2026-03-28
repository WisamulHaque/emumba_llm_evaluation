---
layout: default
title: "1. Strategy"
nav_order: 2
permalink: /strategy/
---

# 1. Evaluation Strategy

Before looking into specific evaluation areas, it's important to first have a high-level strategy. Once the goal is clear, you can work and produce quality results in a more structured manner. This section covers how an end-to-end evaluation plan should look — the approach you should follow to test the credibility of your Gen AI application.

The following are the key areas we cover:

1. **Evaluation Methods** — Manual vs. automated, and which automated approach to pick
2. **Test Set Design** — How to build a test set that covers your application's full surface area
3. **Dataset Generation** — Where the test cases come from
4. **Evaluation Design** — How to turn evaluation results into actionable decisions

---

## Evaluation Methods

There are two broad categories: **manual** (human review) and **automated** (rule-based or LLM-as-judge).

### Manual — Human Review

Manual evaluation cannot be fully replaced. A part of review will always be required, even if you are using automated approaches — a **human feedback loop is necessary**. The person reviewing the LLM output should be well-versed in what they are building.

The downside is that manual QA takes significantly more time and in many cases is simply not feasible. Consider a RAG application with 10,000 documents — a manual reviewer, or even a team of them, won't be able to fully cover it.

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Human Review** | High-stakes decisions, edge cases, calibrating automated judges | Gold-standard quality signal; catches subtle issues machines miss | Expensive, slow, doesn't scale; inter-rater disagreement |

### Automated

To scale evaluation, we need automated approaches. There are two major ones:

#### Rule-Based

Rule-based methods use deterministic checks like **cosine similarity**, regex matching, keyword presence, or schema validation. For example, you compare the LLM output against a ground truth using embedding similarity and threshold it — if the score is above the threshold, it passes; otherwise it fails.

The problem is that the threshold can be unreliable. Because LLM output is non-deterministic, even if the response is accurate, it can still have a low similarity score if the wording is different from the ground truth. And even if you do get results, you'd still need a complete manual review to verify them — there's no reasoning attached to explain *why* something passed or failed.

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Rule-Based** | Deterministic checks — format, schema, keyword presence, regex, similarity | Fast, cheap, reproducible, zero ambiguity | Can't handle open-ended quality; threshold is fragile with non-deterministic outputs; no reasoning |

#### LLM-as-Judge

A better approach to test an LLM application is to **use an LLM itself as the evaluator**. Why? Because it can do verifications based on your provided rules, and it won't just look at things word-by-word — it evaluates by **meaning**. This is critical because the response can be completely correct but worded entirely differently from the ground truth. On top of that, the LLM judge can generate **reasoning** for why it made a certain decision, which helps enormously with debugging.

This approach is scalable. That said, using an LLM as a judge comes at a cost — each evaluation call is an LLM API call.

Throughout this evaluation guide, we have included many prompt templates that you can use for your own use cases.

For a detailed deep-dive on LLM-as-Judge — model selection, prompt design, calibration, bias mitigation, and cost control — refer to: [Why Use LLM as a Judge: A Complete Guide for Software Engineers](https://medium.com/emumba/why-use-llm-as-a-judge-a-complete-guide-for-software-engineers-e79c81cdbc64)

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **LLM as a Judge** | Subjective quality, tone, coherence, faithfulness, any open-ended assessment | Scales to thousands of cases; evaluates by meaning not just words; generates reasoning | Requires calibration; judge model bias; cost per call |

**Code:** This repo includes a shared LLM judge that supports OpenAI, Anthropic, and Groq providers:

[`examples/common/llm_judge.py`](https://github.com/WisamulHaque/emumba_llm_evaluation/blob/main/examples/common/llm_judge.py)

```python
from examples.common.llm_judge import LLMJudge

judge = LLMJudge(provider="openai", model="gpt-4o", api_key="your-api-key")
result = judge.judge("Rate the factual accuracy of this response: ...")
# result: {"score": 1, "reason": "All claims are supported by the ground truth."}
```

---

## Test Set Design

A good test set is the backbone of reliable evaluation. It must cover the full surface area of your application.

### Functional Coverage

| Dimension | Description | Example |
|-----------|-------------|---------|
| **Representative Scenarios** (per capability) | At least one test case per feature/capability your app supports | RAG app → one case per document type; API app → one case per endpoint |
| **Variants / Corners** | Vary difficulty, modality, persona, and prompt style to stress-test generalization | Easy vs. hard queries; formal vs. casual tone; short vs. long context |
| **Negative Controls** | Queries the app *should refuse* — verify graceful refusal | Out-of-scope questions; adversarial prompts; gibberish input |

### Robustness & Safety Suite

| Dimension | Description | Example |
|-----------|-------------|---------|
| **Edge Cases** | Boundary conditions, empty inputs, extremely long context, ambiguous queries | Empty query; 100K-token context; query with no clear intent |
| **Stress Tests** | High-volume, concurrent, or resource-intensive scenarios | 50 parallel requests; context window near max capacity |
| **Adversarial Cases** | Malicious attempts to break the system | Prompt injection; jailbreaking; data exfiltration attempts |

> **Tip:** Store test cases as JSONL files — one JSON object per line. Easy to version, diff, and load.

---

## Dataset Generation

Where do the test cases come from? Use a mix of sources for coverage and realism.

| Source | Description | Best For |
|--------|-------------|----------|
| **Golden Dataset** | Hand-curated, expert-verified input-output pairs | Baseline accuracy measurement; regression testing |
| **Human-Labeled Data** | Real or synthetic inputs annotated by human reviewers | Calibrating LLM judges; subjective quality dimensions |
| **Synthetic Data** | LLM-generated test cases (with human spot-checks) | Scaling coverage quickly; generating edge cases and variants |
| **Real User Data** | Anonymized production logs and user interactions | Ensuring evaluation reflects actual usage patterns |

> **Tip:** Start with a small golden dataset (50–100 cases), expand with synthetic data, and continuously enrich with anonymized real user data as the app matures.

---

## Evaluation Design

How do you turn raw evaluation signals into actionable decisions?

| Approach | Description | When to Use |
|----------|-------------|-------------|
| **Pass / Fail** | Binary threshold — the test case either passes or fails | Deterministic checks (format, schema, security); safety guardrails |
| **Scoring** | Numeric scale (e.g., 1–5, 0–100%) with defined rubrics | Quality dimensions (faithfulness, coherence, completeness) |
| **Human Review** | Qualitative assessment by a human reviewer | Edge cases; calibrating automated scoring; high-stakes decisions |

---

## Example — What an Evaluation Looks Like (RAG)

Here's a concrete example of what an evaluation result looks like for a RAG application using LLM-as-judge:

| Query | Ground Truth | Actual Response | Pass/Fail | Reasoning |
|-------|-------------|-----------------|-----------|-----------|
| What are Python decorators and how do they work? | Python decorators are functions that modify the behavior of other functions using the @syntax. They are used for logging, authentication, and caching. Built-in decorators include @staticmethod, @classmethod, and @property. | Python decorators use the @syntax to modify function behavior. They are commonly used for logging, authentication, and caching. Examples of built-in decorators are @staticmethod, @classmethod, and @property. | ✅ PASS | All claims in the response are supported by the ground truth. The wording differs but the meaning is equivalent. |
| How do database indexes improve query performance? | Database indexes speed up data retrieval by avoiding full table scans. B-tree indexes are the most common type used for range queries and exact matches. | Database indexes speed up data retrieval by avoiding full table scans. B-tree indexes are commonly used. Indexes also add write overhead because they must be updated on every insert, update, or delete. | ❌ FAIL | The response introduces a claim about write overhead that is not present in the ground truth. Two of three claims are supported, but the unsupported claim makes this a failure. |

> This is the kind of output every evaluator in this guide produces — a clear pass/fail verdict with reasoning you can review and debug.

---

**← Home:** [Gen AI Applications Evaluation Guidelines](../) · **Next:** [2. Accuracy →](../accuracy)
