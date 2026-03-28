---
layout: default
title: "3. Performance"
nav_order: 4
permalink: /performance/
---

# 3. Performance

Performance evaluation measures the non-functional characteristics of your LLM application — how fast, how cheap, how scalable, and how resilient it is under real-world conditions.

---

## Latency

Response time is one of the most visible quality signals to end users. Slow responses degrade UX regardless of accuracy.

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Time to First Token (TTFT)** | Time from request submission to the first token appearing in the response stream | Perceived responsiveness — users judge "speed" by when they see the first output, not when the full response completes |
| **End-to-End Latency** | Total time from request to complete response delivery | Overall throughput and SLA compliance; critical for synchronous workflows |
| **Throughput** | Requests processed per second (RPS) under steady-state load | Capacity planning; determines how many concurrent users the system can serve |
| **Streaming Stability** | Consistency of token delivery rate during streamed responses | Choppy streaming (bursts then pauses) feels broken even if total latency is acceptable |
| **Responsiveness** | Time from user action (click, enter) to visible system acknowledgment | Includes network, queue, and pre-processing time — not just LLM inference |

### How to Measure

- **TTFT:** Instrument the streaming callback — timestamp the first `on_token` event minus the request timestamp.
- **E2E Latency:** Timestamp at request send and response complete. For multi-step agents, sum per-step latencies and add orchestration overhead.
- **Throughput:** Run a controlled load test (see Load Testing below) and measure sustained RPS at acceptable latency percentiles (p50, p95, p99).
- **Streaming Stability:** Measure inter-token intervals during streaming. Flag if standard deviation exceeds a threshold (e.g., >500ms gaps).

### Tools

For **end-to-end load and latency testing**, industry-standard tools like [JMeter](https://jmeter.apache.org/) and [k6](https://k6.io/) are well-suited — they can simulate concurrent users, measure p50/p95/p99 latencies, and stress-test your full pipeline including retrieval, orchestration, and LLM inference.

For **chunk-by-chunk streaming latency** (TTFT, inter-token intervals, streaming stability), use [`streamapiperformance`](https://www.npmjs.com/package/streamapiperformance) — an npm package purpose-built for measuring token-level timing in streamed LLM responses.

**Code:** [`examples/performance/latency_evaluator.py`](https://github.com/WisamulHaque/emumba_llm_evaluation/blob/main/examples/performance/latency_evaluator.py) — a simple callable-based timing wrapper for measuring E2E latency of any LLM call.

---

## Cost

At production scale, cost becomes one of the most critical performance dimensions. As user volume grows, every unnecessary token, redundant LLM call, and over-fetched context chunk compounds into significant spend. A single query in a multi-agent RAG pipeline can trigger 3–10+ LLM invocations — if you're not tracking cost per journey, you're flying blind.

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Per Model Call** | Token cost (input + output) for a single LLM invocation | Baseline unit cost; varies dramatically between models (GPT-4o vs. GPT-4o-mini vs. open-source) |
| **End-to-End Journey Cost** | Total cost of resolving a user query, including all LLM calls, retrieval, and tool invocations | Multi-agent and RAG systems often make 3–10+ LLM calls per query; per-call cost alone is misleading |
| **Token Usage Efficiency** | Ratio of useful output tokens to total tokens consumed (including system prompts, retries, and context) | Bloated system prompts, unnecessary retries, and over-fetched context silently inflate costs |

### How to Measure

- **Per Model Call:** Log `prompt_tokens` and `completion_tokens` from the API response. Multiply by the model's per-token pricing.
- **Journey Cost:** Sum all per-call costs across the full agent/RAG pipeline for a single user query. Track as a distribution (p50, p95).
- **Token Efficiency:** `(output_tokens) / (total_input_tokens + output_tokens)`. Low efficiency suggests system prompt bloat or over-retrieval.

---

## Context & Memory Efficiency

For RAG and memory-augmented systems, more context is not always better. There's a quality curve.

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Context Size vs Quality Curve** | How response quality changes as you increase the number of retrieved chunks (K) | Diminishing returns — going from K=3 to K=10 may add noise without improving accuracy |
| **Memory Size vs Relevance Curve** | How memory recall quality degrades as the memory store grows | Older or less relevant memories may pollute the context window |

### How to Evaluate

1. **Sweep K:** Run the same test set with K=1, 3, 5, 10, 20. Plot accuracy (or faithfulness score) vs K.
2. **Find the elbow:** Identify the point where adding more chunks stops improving quality.

---

## Load Testing

Evaluate system behavior under realistic and peak traffic conditions.

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Concurrency Limit** | Maximum number of simultaneous requests before latency degrades beyond acceptable thresholds | Capacity planning; determines infrastructure scaling requirements |
| **Peak Load Behavior** | System behavior at and beyond capacity — does it degrade gracefully or fail catastrophically? | Determines whether the system queues, throttles, or crashes under burst traffic |

### How to Test

- Use load testing tools ([Locust](https://locust.io/), [k6](https://k6.io/), [Artillery](https://www.artillery.io/)) to simulate concurrent users.
- Ramp from 1 → N concurrent requests. Record latency percentiles (p50, p95, p99) and error rates at each level.
- Identify the concurrency at which p95 latency exceeds your SLA — that's your effective concurrency limit.
- Push 20% beyond that limit and observe: does the system queue gracefully, return 429s, or crash?

> **Tool recommendation:** [Locust](https://locust.io/) is Python-native and easy to script custom LLM request patterns. [k6](https://k6.io/) is better for high-volume HTTP benchmarks with built-in dashboards.

---

## Reliability

How does the system behave when things go wrong?

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Retry Mechanism** | Whether failed LLM calls, tool invocations, or retrieval steps are automatically retried with appropriate backoff | Transient failures (rate limits, timeouts) are common; retries prevent unnecessary user-facing errors |
| **Graceful Degradation** | System behavior when a component fails — does it fall back to a simpler path or fail entirely? | Users prefer a partial answer over a cryptic error; fallback chains maintain UX under failure conditions |

### What to Validate

| Scenario | Expected Behavior |
|----------|-------------------|
| LLM API returns 429 (rate limited) | Retry with exponential backoff; succeed within 2–3 retries |
| Primary retrieval service is down | Fall back to cached results or a secondary index |
| One agent in a multi-agent chain times out | Orchestrator detects the timeout, skips or retries the step, and returns a partial result with a disclaimer |
| LLM returns unparseable output (malformed JSON) | Retry with a stricter prompt; if still fails, return a structured error to the user |

---

**← Previous:** [2. Accuracy](../accuracy) · **Next:** [4. Safety →](../safety)
