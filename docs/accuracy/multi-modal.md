---
layout: default
title: "Multi-Modal"
parent: "2. Accuracy"
nav_order: 6
permalink: /accuracy/multi-modal/
---

# Multi-Modal I/O Accuracy

Most Gen AI applications start as text-in, text-out — but as products mature, they expand to voice interfaces, image understanding, and combinations of both. When your application crosses modality boundaries, new failure modes emerge that pure text evaluation won't catch.

The metrics below are **product-agnostic** — they apply regardless of which ASR engine, TTS provider, or vision model you use. The specifics of how you instrument them depend on your stack, but *what* you measure is universal.

---

## A. Text Interaction

Even in text-only systems, two I/O-level quality signals are often overlooked:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Streaming Stability** | Partial outputs don't contradict the final meaning or mislead the user mid-stream | Users read tokens as they arrive — if early tokens commit to a claim the model later reverses, trust breaks even if the final output is correct |
| **Interaction Responsiveness** | System responds promptly to user actions (send, stop, regenerate) | Perceived quality degrades if there's a visible lag between user action and system acknowledgment, regardless of actual LLM inference speed |

**Manual Examples:**

| Scenario | Result |
|----------|--------|
| User sends a query; first token appears within 500ms | ✅ PASS — responsive |
| Streamed tokens say "The answer is yes..." then final output says "...but actually no" | ❌ FAIL — contradictory commitment mid-stream |
| User clicks "Stop Generating"; system continues for 3+ seconds | ❌ FAIL — unresponsive to user action |

---

## B. Voice I/O

Standard metrics for any conversational voice system — whether it's a phone agent, voice assistant, or voice-enabled chatbot:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **ASR Accuracy (WER)** | Word Error Rate of speech-to-text transcription | Transcription errors propagate through the entire pipeline — if the system mishears "cancel" as "handle", everything downstream is wrong |
| **Turn-Taking Latency (UTFT)** | Time from when the user stops speaking to when the system begins its audio response | Long pauses feel like the system is broken; too-short pauses cause the system to cut off the user |
| **Barge-In Handling** | When the user interrupts mid-response, the system stops speaking and listens immediately | If the system keeps talking over the user, it signals a non-conversational, frustrating experience |

**Manual Examples:**

| Scenario | Result |
|----------|--------|
| User says "Cancel my order" → transcribed as "Cancel my order" | ✅ PASS — accurate transcription |
| User says "Cancel my order" → transcribed as "Handle my border" | ❌ FAIL — high WER, downstream pipeline will fail |
| User finishes speaking; system responds within 800ms | ✅ PASS — natural turn-taking |
| User finishes speaking; 4-second silence before response | ❌ FAIL — perceived as broken |
| User interrupts mid-response; system stops and listens | ✅ PASS — proper barge-in |
| User interrupts mid-response; system keeps talking | ❌ FAIL — ignores user input |

> *No code evaluators provided — these metrics require integration with your specific ASR/TTS pipeline. Measure WER by comparing ASR output against human-transcribed reference; measure latency by timestamping audio events.*

---

## C. Vision I/O

Core metrics for any image-based pipeline — document analysis, visual Q&A, screenshot understanding, diagram interpretation:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Visual Task Success** | System correctly performs the requested visual task (e.g., "describe this chart", "extract the table from this receipt") | The fundamental pass/fail — did the system do what the user asked with the image? |
| **Grounded Claim Support** | Claims made about visual content are supported by what's actually in the image | Vision models hallucinate just like text models — claiming a chart shows an upward trend when it shows a decline is dangerous |
| **OCR Accuracy** | Text extracted from images matches the actual text (if text extraction is involved) | OCR errors in receipts, documents, or screenshots propagate to downstream processing |

**Manual Examples:**

| Scenario | Result |
|----------|--------|
| User uploads a bar chart → system correctly identifies the highest bar as "Q3 at $4.2M" | ✅ PASS — correct visual interpretation |
| User uploads a bar chart → system claims "Q1 had the highest revenue" when Q3 is clearly tallest | ❌ FAIL — hallucinated visual claim |
| User uploads a receipt → system extracts "Total: $42.50" matching the actual text | ✅ PASS — accurate OCR |
| User uploads a receipt → system extracts "Total: $425.0" (decimal error) | ❌ FAIL — OCR error |

> *No code evaluators provided — these metrics require integration with your specific vision model. Evaluate by comparing model outputs against human-annotated visual ground truth.*

---

## D. Cross-Modal Stability

When your application handles multiple modalities (e.g., voice input + text output, image input + voice response), additional failure modes emerge at the boundaries:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Graceful Recovery** | If an input fails (corrupted audio, broken image, unsupported format), the system recovers cleanly rather than crashing or returning nonsense | Users will send bad inputs — the system must degrade gracefully, not catastrophically |
| **Safety Compliance** | No unsafe inferences or PII leakage occur when crossing modality boundaries | A vision model might extract PII from an image that the text layer then includes in the response; safety must hold across the full chain |

**Manual Examples:**

| Scenario | Result |
|----------|--------|
| User sends a corrupted audio file → system responds "I couldn't process that audio. Could you try again?" | ✅ PASS — graceful recovery |
| User sends a corrupted audio file → system crashes or returns garbled text | ❌ FAIL — no recovery |
| User uploads an image containing a credit card → system describes the image without revealing the card number | ✅ PASS — PII safety across modalities |
| User uploads an image containing a credit card → system includes the full card number in its text response | ❌ FAIL — PII leakage across modality boundary |

---

## Summary Table

| Area | Metric | What It Measures |
|------|--------|-----------------|
| **Text** | Streaming Stability | Partial outputs don't contradict final meaning or mislead user |
| | Responsiveness | System responds promptly to user actions (start, stop, send) |
| **Voice** | ASR Accuracy (WER) | Speech transcription accuracy |
| | Turn-Taking Latency | How quickly system responds after user stops speaking |
| | Barge-In Handling | If user interrupts, system stops speaking and listens immediately |
| **Vision** | Visual Task Success | System correctly performs the requested visual task |
| | Grounded Claim Support | Claims are supported by the image (no hallucination) |
| | OCR Accuracy | Correct extraction of text from images |
| **Cross-Modal** | Graceful Recovery | If input fails (broken audio/image), system recovers cleanly |
| | Safety Compliance | No unsafe inferences or PII leakage across modalities |
