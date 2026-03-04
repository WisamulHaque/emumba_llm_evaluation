"""
Response Evaluators Runner

Loads evaluation samples from a JSON input file, runs the configured
response evaluators, and writes results to a JSONL output file.

Supported LLM providers (configured via .env):
    - Anthropic (claude-*)
    - OpenAI    (gpt-*)
    - Groq      (llama-*, mixtral-*, gemma-*)

Configuration:
    Create a .env file with:
        RESPONSE_EVAL_PROVIDER=groq        # anthropic | openai | groq
        RESPONSE_EVAL_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
        RESPONSE_EVAL_API_KEY=your_api_key
        RESPONSE_EVAL_INPUT=input/answer_sample_input.json
        RESPONSE_EVAL_OUTPUT=results/response_results.jsonl

Usage:
    python response_runner.py

Input JSON format:
    [
        {
            "query_id": 1,
            "query": "What are Python decorators?",
            "retrieved_chunks": ["Python decorators are...", "..."],
            "generated_response": "Python decorators are functions that modify other functions...",
            "ground_truth": "Python decorators wrap other functions using the @syntax..."
        }
    ]
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from examples.common.utils import truncate_text

from response_evaluators import (
    IntentAccuracyEvaluator,
    ResponseEvaluationSample,
    LLMJudge,
    ResponseAccuracyEvaluator,
    FaithfulnessEvaluator,
    ResponseCompletenessEvaluator,
)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass


def get_config() -> dict:
    """Load configuration from environment variables."""
    return {
        "input_path":  os.environ.get("RESPONSE_EVAL_INPUT"),
        "output_path": os.environ.get("RESPONSE_EVAL_OUTPUT", "results/response_results.jsonl"),
        "provider":    os.environ.get("RESPONSE_EVAL_PROVIDER", "groq"),
        "model":       os.environ.get("RESPONSE_EVAL_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        "api_key":     os.environ.get("RESPONSE_EVAL_API_KEY"),
    }


def evaluate_from_file(
    input_path: str,
    output_path: str,
    provider: str,
    model: str,
    api_key: str,
) -> None:
    """
    Run response evaluation on a JSON input file.
    Results are written to a JSONL file immediately after each evaluation.

    Args:
        input_path:  Path to input JSON file with evaluation samples
        output_path: Path to output JSONL file for results
        provider:    LLM provider — "anthropic", "openai", or "groq"
        model:       Model name for the judge LLM
        api_key:     API key for the provider
    """
    judge = LLMJudge(provider=provider, model=model, api_key=api_key)
    accuracy_evaluator = ResponseAccuracyEvaluator(judge=judge)
    faithfulness_evaluator = FaithfulnessEvaluator(judge=judge)
    intent_evaluator = IntentAccuracyEvaluator(judge=judge)
    completeness_evaluator = ResponseCompletenessEvaluator(judge=judge)


    with open(input_path, 'r', encoding='utf-8') as f:
        samples_data = json.load(f)

    print(f"Loaded {len(samples_data)} samples from {input_path}")
    print(f"Provider         : {provider}")
    print(f"Model            : {model}")
    print(f"Writing results to {output_path}")
    print("-" * 60)

    
    accuracy_passed_count = 0
    faithfulness_passed_count = 0
    intent_passed_count = 0
    completeness_passed_count = 0
    total_count = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for i, sample_data in enumerate(samples_data, 1):
            sample = ResponseEvaluationSample.from_dict(sample_data)
            accuracy_result = accuracy_evaluator.evaluate_response_accuracy(sample)
            faithfulness_result = faithfulness_evaluator.evaluate_faithfulness(sample)
            intent_result = intent_evaluator.evaluate_intent_accuracy(sample)
            completeness_result = completeness_evaluator.evaluate_completeness(sample)

            output_record = {
                "query_id": sample.query_id,
                "query": truncate_text(sample.query, 80),
                "generated_response": truncate_text(sample.generated_response, 80),
                "evaluations": {
                    "accuracy": accuracy_result,
                    "faithfulness": faithfulness_result,
                    "intent_accuracy": intent_result,
                    "completeness": completeness_result,
                }
            }

            # Write immediately — safe against crashes mid-run
            out_file.write(json.dumps(output_record) + "\n")
            out_file.flush()

            if accuracy_result["passed"]:
                accuracy_passed_count += 1
            if faithfulness_result["passed"]:
                faithfulness_passed_count += 1
            if intent_result["passed"]:   
                intent_passed_count += 1
            if completeness_result["passed"]:
                completeness_passed_count += 1
            total_count += 1

    print(f"Results saved to : {output_path}")
    print(f"Accuracy         : {accuracy_passed_count}/{total_count} passed ({100*accuracy_passed_count/total_count:.1f}%)")
    print(f"Faithfulness     : {faithfulness_passed_count}/{total_count} passed ({100*faithfulness_passed_count/total_count:.1f}%)")
    print(f"Intent Accuracy  : {intent_passed_count}/{total_count} passed ({100*intent_passed_count/total_count:.1f}%)")
    print(f"Completeness     : {completeness_passed_count}/{total_count} passed ({100*completeness_passed_count/total_count:.1f}%)")

def main():
    """Run response evaluation from input file configured in .env."""
    config = get_config()

    if not config["input_path"]:
        print("No input file configured.")
        print("To run evaluation:")
        print("  1. Copy .env.example to .env")
        print("  2. Set RESPONSE_EVAL_INPUT to your input JSON file path")
        print("  3. Run this script again")
        print()
        print("See answer_sample_input.json for input file format reference.")
        return

    if not Path(config["input_path"]).exists():
        print(f"Input file not found: {config['input_path']}")
        print("Please check the RESPONSE_EVAL_INPUT path in your .env file.")
        return

    if not config["api_key"]:
        print("No API key configured.")
        print("Set RESPONSE_EVAL_API_KEY in your .env file.")
        return

    evaluate_from_file(
        input_path=config["input_path"],
        output_path=config["output_path"],
        provider=config["provider"],
        model=config["model"],
        api_key=config["api_key"],
    )


if __name__ == "__main__":
    main()