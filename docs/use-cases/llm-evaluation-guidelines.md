---
layout: default
title: LLM Evaluation Guidelines
parent: Use Cases
nav_order: 1
has_children: true
permalink: /use-cases/llm-evaluation-guidelines/
---

# LLM Evaluation Guidelines

This document defines the core evaluation areas for LLM-powered applications. The evaluation strategy depends on the type of application being built. We categorize LLM applications into **three main categories**, each with its own set of evaluation areas.

---

## Application Categories

### [Category 1: Data Source Applications](./category-1-data-source-apps)
Single-responsibility LLM applications that interact with an external data source to answer user queries — including **Chat with Documents (RAG)**, **Chat with APIs**, and **Chat with Databases**. Covers retrieval accuracy, faithfulness, response completeness, and more.

### [Category 2: Multi-Agent Applications](./category-2-multi-agent-apps)
Complex applications composed of multiple specialized agents, tools, and decision pathways. Covers **journey validation**, **tool call accuracy**, **orchestration efficiency**, and **error recovery**.

### [Category 3: No Ground Truth Applications](./category-3-no-ground-truth-apps)
Applications where no external data source or ground truth exists — including conversational chatbots, interview bots, and coaching assistants. Covers **tonality**, **persona consistency**, **engagement quality**, and **empathy**.

---

## Summary

| Category | Key Focus | Primary Challenge |
|----------|-----------|-------------------|
| **Simple LLM Apps (with Data Sources)** | Accuracy of retrieval and response generation | Grounding responses in external data |
| **Multi-Agent Applications** | Orchestration correctness and tool usage | Maintaining coherence across agent chain |
| **Simple LLM Apps (without Ground Truth)** | Behavioral alignment and persona adherence | Evaluating without reference answers |
