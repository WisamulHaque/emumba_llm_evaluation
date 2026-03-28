# Gen AI Applications Evaluation Guidelines

A comprehensive, end-to-end guide to evaluating Gen AI applications.

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://WisamulHaque.github.io/emumba_llm_evaluation/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 📖 Overview

This repository provides a structured evaluation framework for LLM-powered applications, organized around **four pillars**: Strategy, Accuracy, Performance, and Safety. Whether you're building RAG systems, multi-agent pipelines, chatbots, or other LLM applications, you'll find evaluation areas, metrics, worked examples, and code evaluators here.

## 🚀 Quick Start

### Documentation

Visit our [documentation site](https://WisamulHaque.github.io/emumba_llm_evaluation/) for the full guide, or [view the interactive mindmap](https://WisamulHaque.github.io/emumba_llm_evaluation/mindmap.html).

### Running Code Examples

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create a .env file with your API keys
cat > .env << 'EOF'
LLM_JUDGE_PROVIDER=groq
LLM_JUDGE_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
LLM_JUDGE_API_KEY=your-groq-api-key

# Required for guardrails evaluator (OpenAI Moderation API)
GUARDRAILS_PROVIDER=openai_moderation
GUARDRAILS_API_KEY=your-openai-api-key
EOF

# 3. Run any evaluator
python -m examples.accuracy.response_quality.task_quality_evaluator
python -m examples.safety.guardrails.prompt_injection_evaluator
python -m examples.accuracy.context_sourcing.rag_retrieval_evaluator
```

## 📂 Repository Structure

```
emumba_llm_evaluation/
├── docs/                                  # Documentation (GitHub Pages)
│   ├── index.md                           # Home page
│   ├── strategy.md                        # 1. Evaluation Strategy
│   ├── accuracy.md                        # 2. Accuracy (parent page)
│   │   └── accuracy/                      # Accuracy sub-pages
│   │       ├── response-quality.md        #    Response Quality
│   │       ├── context-sourcing.md        #    Context Sourcing
│   │       ├── grounded-accuracy.md       #    Grounded Accuracy
│   │       ├── memory.md                  #    Memory
│   │       ├── agentic.md                 #    Agentic Evaluation
│   │       └── multi-modal.md             #    Multi-Modal I/O
│   ├── performance.md                     # 3. Performance
│   ├── safety.md                          # 4. Safety
│   └── mindmap.html                       # Interactive mindmap
│
├── examples/                              # Code evaluators
│   ├── accuracy/
│   │   ├── response_quality/              # Task quality, instruction following, factuality, consistency
│   │   ├── context_sourcing/              # RAG retrieval, non-RAG sourcing (API, params, query gen)
│   │   ├── grounded_accuracy/             # Faithfulness, citation accuracy
│   │   ├── memory/                        # Memory correctness, recall relevance, update correctness
│   │   └── agentic/                       # Tool calls, task adherence, trajectory (correctness + efficiency), planning
│   ├── safety/
│   │   ├── guardrails/                    # Prompt injection, guardrails (jailbreaking, policy compliance)
│   │   └── privacy/                       # PII protection, data security
│   ├── performance/
│   │   └── latency_evaluator.py           # Latency measurement
│   └── common/                            # Shared utilities (LLM Judge, utils)
│
└── README.md                              # This file
```

## 🗺️ The Four Pillars

| Pillar | Focus | Key Areas |
|--------|-------|-----------|
| **[1. Strategy](docs/strategy.md)** | How to evaluate | Evaluation methods, test set design, dataset generation, scoring |
| **[2. Accuracy](docs/accuracy.md)** | Is the output correct? | Response quality, context sourcing, grounded accuracy, memory, agentic evaluation, multi-modal I/O |
| **[3. Performance](docs/performance.md)** | Is it fast & efficient? | Latency (TTFT, E2E), cost, context efficiency, load testing, reliability |
| **[4. Safety](docs/safety.md)** | Is it safe & trustworthy? | Guardrails, privacy, bias & fairness, content safety |

## 🔧 Key Evaluation Areas

### Accuracy — Capability Evaluation
- **Response Quality**: Task quality, instruction following, factuality, consistency
- **Context Sourcing**: RAG sourcing (recall, precision), non-RAG sourcing (API selection, parameter accuracy, query generation)
- **Grounded Accuracy**: Faithfulness to context, citation correctness, hallucination detection
- **Memory**: Memory correctness, recall relevance, update correctness
- **Agentic Evaluation**: Tool call accuracy, task adherence, trajectory quality (correctness + efficiency), plan quality
- **Multi-Modal I/O**: Text (streaming stability, responsiveness), Voice (ASR, turn-taking, barge-in), Vision (visual tasks, OCR), Cross-modal stability

### Performance
- **Latency**: TTFT, E2E latency, throughput, streaming stability, responsiveness
- **Cost**: Per model call, E2E journey cost, token usage efficiency
- **Context & Memory Efficiency**: K-sweep optimization, memory relevance curve
- **Load Testing**: Concurrency limits, peak load behavior
- **Reliability**: Retry mechanisms, graceful degradation

### Safety
- **Guardrails**: Prompt injection, jailbreaking, policy compliance, ethical constraints
- **Privacy**: PII protection, role-based access
- **Bias & Fairness**: Demographic parity, stereotyping detection
- **Content Safety**: Toxic output detection, harmful content filtering

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ for the LLM community**
