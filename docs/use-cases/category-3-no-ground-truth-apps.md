---
layout: default
title: "Category 3: No Ground Truth Applications"
parent: LLM Evaluation Guidelines
grand_parent: Use Cases
nav_order: 3
permalink: /use-cases/category-3-no-ground-truth-apps/
---

# Category 3: Simple LLM Applications without Ground Truth

These are applications where no external data source or ground truth exists. The LLM generates responses based solely on its training knowledge and the application's system prompt. Examples include conversational chatbots, interview-taking applications, coaching bots, and creative assistants.

---

## Evaluation Areas

**I. Tonality**
Validate that the tone, style, and language of the application's responses align with the defined purpose and persona of the application. An interview bot should sound professional, a coaching bot should sound supportive, etc.

**II. Response Accuracy**
Ensure the application's responses are factually sound and align with the rules, instructions, and constraints defined in the system prompt. Even without ground truth data, responses should not contradict the application's own guidelines.

**III. Persona Consistency**
Verify that the application maintains its defined character, role, and personality consistently throughout the conversation. It should not break character or contradict its own prior behavior.

**IV. Coherence**
Validate that individual responses are logically structured, well-organized, and easy to understand. Responses should not contain contradictions, non-sequiturs, or disjointed information.

**V. Conversational Flow**
Evaluate the naturalness of the conversation. The application should handle topic transitions smoothly, maintain context across turns, and support natural turn-taking patterns.

**VI. Engagement Quality**
Assess whether the application's responses promote meaningful continued interaction. For applications like interview bots, this means asking relevant follow-up questions; for coaching bots, it means providing actionable guidance.

**VII. Empathy and Emotional Intelligence**
For applications that interact with users in sensitive contexts (interviews, coaching, support), validate that the responses demonstrate appropriate empathy, emotional awareness, and sensitivity.

**VIII. Out-of-Scope Accuracy**
Validate that the application correctly identifies and handles queries outside its defined domain. It should redirect the user back to the application's purpose rather than generating irrelevant or hallucinated responses.

**IX. Guardrails Accuracy**
Ensure the application upholds community guidelines and does not generate harmful, offensive, or inappropriate content regardless of user input.

**X. Prompt Injection Resistance**
Validate that the application rejects all attempts to manipulate its system prompt, override its instructions, or extract its internal configuration through adversarial user inputs.
