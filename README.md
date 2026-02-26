# LLM Evaluation Guidelines

A comprehensive guide to evaluating Large Language Models across various application types.

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://WisamulHaque.github.io/emumba_llm_evaluation/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 📖 Overview

This repository provides structured evaluation guidelines for LLM-powered applications. Whether you're building RAG systems, multi-agent pipelines, chatbots, or other LLM applications, you'll find evaluation areas and guidance here.

## 🚀 Quick Start

### Documentation

Visit our [documentation site](https://WisamulHaque.github.io/emumba_llm_evaluation/) for comprehensive guides.

## 📂 Repository Structure

```
emumba_llm_evaluation/
├── docs/                           # Documentation (GitHub Pages)
│   ├── index.md                    # Home page
│   └── use-cases/                  # Evaluation guidelines
│       └── llm-evaluation-guidelines.md  # Core evaluation areas
│
├── examples/                       # Code examples
│   ├── rag/                        # RAG evaluation examples
│   ├── chat-api/                   # Chat API evaluation examples
│   └── common/                     # Shared utilities
│
└── README.md                       # This file
```

## 📚 LLM Application Categories

We evaluate LLM applications across three main categories:

| Category | Examples | Documentation |
|----------|----------|---------------|
| **Simple LLM Apps (with Data Sources)** | Chat with Documents, Chat with APIs, Chat with Databases | [Guidelines](docs/use-cases/llm-evaluation-guidelines.md) |
| **Multi-Agent Applications** | Multi-tool orchestration, agent pipelines | [Guidelines](docs/use-cases/llm-evaluation-guidelines.md) |
| **Simple LLM Apps (without Ground Truth)** | Chatbots, interview bots, coaching assistants | [Guidelines](docs/use-cases/llm-evaluation-guidelines.md) |

## 🔧 Key Evaluation Areas

### Simple LLM Apps (with Data Sources)
- **Retrieval Accuracy**: Relevance of retrieved information
- **Response Accuracy & Faithfulness**: Grounded, hallucination-free responses
- **API/Query Selection Accuracy**: Correct tool and parameter selection
- **Privacy & Security**: Data masking and injection prevention

### Multi-Agent Applications
- **Behavioral Accuracy**: Correct journey through agent graph
- **Tool Call Accuracy**: Right tools with right parameters
- **Memory & Context Preservation**: No information loss across agents

### Simple LLM Apps (without Ground Truth)
- **Tonality & Persona Consistency**: Aligned with application purpose
- **Coherence & Engagement**: Logical, meaningful interactions
- **Guardrails & Prompt Injection Resistance**: Safety and robustness

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
