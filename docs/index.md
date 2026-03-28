---
layout: default
title: Home
nav_order: 1
description: "Gen AI Applications Evaluation Guidelines - A comprehensive guide to evaluating Gen AI applications"
permalink: /
---

# Gen AI Applications Evaluation Guidelines

Welcome to the **Gen AI Applications Evaluation Guidelines** repository.

At the core of every Gen AI application is a Large Language Model — and LLMs are **non-deterministic by nature**. In traditional software, you have defined inputs and expected outputs: if X happens, Z should be the result, and you can pass or fail accordingly. With LLMs, the same query can produce different responses every time. This is what makes them powerful, but it's also what makes them unpredictable.

Because of this, **testing becomes even more critical before anything moves to production**. Without proper evaluation, you're shipping software that you can't reliably verify — and that's a risk no team scaling a Gen AI product can afford.

This guide is built for anyone who wants to **scale their Gen AI products with proper evaluation in place**. It covers the areas that should be evaluated, provides a brief and precise introduction to each area with concrete examples, and includes code snippet examples you can view and use as a head start on your own implementation.

<a href="./mindmap.html" target="_blank">🗺️ View the interactive mindmap of all evaluation areas →</a>

---

## How This Guide Is Organized

| Pillar | What It Covers |
|--------|----------------|
| **[1. Strategy](./strategy)** | The high-level "how" of evaluation — what types of evaluation exist, how to build test datasets, how to score and interpret results |
| **[2. Accuracy](./accuracy)** | The specific areas that should be evaluated in a Gen AI application — from simple single-LLM apps to data-source-backed systems (RAG, databases, APIs) and multi-agent pipelines |
| **[3. Performance](./performance)** | A comprehensive writeup on measuring and optimizing latency, cost, throughput, and reliability under real-world conditions |
| **[4. Safety](./safety)** | The responsible AI perspective — guardrails, privacy, bias & fairness, and content safety, covered in detail |

## Quick Start

1. **Start with Strategy**: Define your evaluation methods, test sets, and scoring approach
2. **Measure Accuracy**: Pick the capability areas relevant to your app (RAG, agents, multi-modal, etc.)
3. **Benchmark Performance**: Establish latency, cost, and reliability baselines
4. **Validate Safety**: Run guardrail, privacy, bias, and content safety checks

---

**Start here:** [1. Strategy →](./strategy)
