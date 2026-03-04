"""
Guardrails Evaluation Runner

Loads evaluation samples from a JSON input file, runs the configured
guardrails evaluator, and writes results to a JSONL output file.

Configuration:
    Create a .env file with:
        GUARDRAILS_PROVIDER=openai_moderation   # openai_moderation | llamaguard
        GUARDRAILS_API_KEY=your_api_key
        GUARDRAILS_SCORE_THRESHOLD=0.7          # optional, openai_moderation only
        GUARDRAILS_INPUT=input/common_sample_input.json
        GUARDRAILS_OUTPUT=results/common_eval_results.jsonl

Usage:
    python guardrails_runner.py

Input JSON format:
    [
        {
            "query_id": 1,
            "query": "What were the causes of World War II?",
            "generated_response": "The primary causes were...",
            "expected_behavior": "comply"   // optional: "comply" | "refuse"
        }
    ]

Output JSONL format (one JSON object per line):
    {
        "query_id": 1,
        "query": "...",
        "passed": true,
        "reason": "No policy violations detected.",
        "violated_categories": []
    }
"""

import json
import os
import sys
from pathlib import Path

# sys.path.insert must come before any local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.common.utils import truncate_text
from examples.common.common_evaluators import GuardrailsEvaluator, GuardrailsSample

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path)
except ImportError:
    pass


def get_config() -> dict:
    """Load configuration from environment variables."""
    threshold_raw = os.environ.get("GUARDRAILS_SCORE_THRESHOLD")
    return {
        "input_path":       os.environ.get("GUARDRAILS_INPUT"),
        "output_path":      os.environ.get("GUARDRAILS_OUTPUT", "examples/common/results/common_eval_results.jsonl"),
        "provider":         os.environ.get("GUARDRAILS_PROVIDER", "openai_moderation"),
        "api_key":          os.environ.get("GUARDRAILS_API_KEY"),
        "score_threshold":  float(threshold_raw) if threshold_raw else None,
    }


def evaluate_from_file(
    input_path: str,
    output_path: str,
    provider: str,
    api_key: str,
    score_threshold: float = None,
) -> None:
    """
    Run guardrails evaluation on a JSON input file.
    Results are written to a JSONL file immediately after each evaluation.

    Args:
        input_path:      Path to input JSON file with evaluation samples
        output_path:     Path to output JSONL file for results
        provider:        Safety backend — "openai_moderation" or "llamaguard"
        api_key:         API key for the selected provider
        score_threshold: Optional severity threshold for openai_moderation.
                         When set, fails only if category score > threshold.
    """
    evaluator = GuardrailsEvaluator(
        provider=provider,
        api_key=api_key,
        score_threshold=score_threshold
    )

    with open(input_path, 'r', encoding='utf-8') as f:
        samples_data = json.load(f)

    print(f"Loaded {len(samples_data)} samples from {input_path}")
    print(f"Provider         : {provider}")
    if score_threshold is not None:
        print(f"Score threshold  : {score_threshold}  (openai_moderation only)")
    print(f"Writing results to {output_path}")
    print("-" * 60)

    passed_count = 0
    total_count  = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for i, sample_data in enumerate(samples_data, 1):
            sample = GuardrailsSample.from_dict(sample_data)
            result = evaluator.evaluate_guardrails(sample)

            output_record = {
                "query_id": sample.query_id,
                "query":    truncate_text(sample.query, 80),
                "generated_response": truncate_text(sample.generated_response, 80),
                "guardrails_provider": provider,
                **result
            }

            # Write immediately — safe against crashes mid-run
            out_file.write(json.dumps(output_record) + "\n")
            out_file.flush()

            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{i}/{len(samples_data)}] {status}: {truncate_text(sample.query, 60)}")
            if sample.expected_behavior:
                print(f"         Expected : {sample.expected_behavior}")
            if not result["passed"]:
                print(f"         Violated : {result['violated_categories']}")
            print(f"         Reason   : {result['reason']}")

            if result["passed"]:
                passed_count += 1
            total_count += 1

    print("-" * 60)
    print(f"Results saved to : {output_path}")
    print(f"Summary          : {passed_count}/{total_count} passed ({100*passed_count/total_count:.1f}%)")


def main():
    """Run guardrails evaluation from input file configured in .env."""
    config = get_config()

    if not config["input_path"]:
        print("No input file configured.")
        print("To run evaluation:")
        print("  1. Copy .env.example to .env")
        print("  2. Set GUARDRAILS_INPUT to your input JSON file path")
        print("  3. Run this script again")
        print()
        print("See common_sample_input.json for input file format reference.")
        return

    if not Path(config["input_path"]).exists():
        print(f"Input file not found: {config['input_path']}")
        print("Please check the GUARDRAILS_INPUT path in your .env file.")
        return

    if not config["api_key"]:
        print("No API key configured.")
        print("Set GUARDRAILS_API_KEY in your .env file.")
        return

    evaluate_from_file(
        input_path=config["input_path"],
        output_path=config["output_path"],
        provider=config["provider"],
        api_key=config["api_key"],
        score_threshold=config["score_threshold"],
    )


if __name__ == "__main__":
    main()