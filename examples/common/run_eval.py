"""
LLM Evaluation Framework — Unified Runner

Single entry point for all RAG evaluators. Evaluator is selected via
command-line argument. All configuration is read from a single .env file.

Usage:
    python run_eval.py <evaluator>

Available evaluators:
    retrieval              Retrieval accuracy (embedding-based, no API key needed)
    response               All response evaluators combined (accuracy + faithfulness + intent)
    response_accuracy      Response factual accuracy vs ground truth
    response_faithfulness  Claim grounding against retrieved chunks
    response_intent        Whether response fulfills user intent
    response_completeness  Whether response covers all relevant aspects
    guardrails             Harmful/offensive content detection
    injection              Prompt injection resistance
    out_of_scope           Out-of-scope query handling
    consistency            Factual and quality consistency across paraphrased queries


Examples:
    python run_eval.py retrieval
    python run_eval.py response
    python run_eval.py response_accuracy
    python run_eval.py response_faithfulness
    python run_eval.py response_intent
    python run_eval.py response_completeness
    python run_eval.py guardrails
    python run_eval.py injection
    python run_eval.py out_of_scope
    python run_eval.py consistency

Configuration:
    Copy .env.example to .env and fill in values for the evaluators you need.
    Only the variables for the selected evaluator are required at runtime.
    All response_* evaluators share the same RESPONSE_EVAL_* variables.

Output files:
    retrieval              → results/retrieval_results.jsonl
    response               → results/response_results.jsonl
    response_accuracy      → results/response_accuracy_results.jsonl
    response_faithfulness  → results/response_faithfulness_results.jsonl
    response_intent        → results/response_intent_results.jsonl
    response_completeness  → results/response_completeness_results.jsonl
    guardrails             → results/guardrails_results.jsonl
    injection              → results/injection_results.jsonl
    out_of_scope           → results/out_of_scope_results.jsonl
    consistency            → results/consistency_results.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# sys.path.insert must come before any local imports
_HERE      = os.path.dirname(os.path.abspath(__file__))          # .../evaluators/rag
_RAG_ROOT  = _HERE                                                # evaluator modules live here
_PROJ_ROOT = os.path.dirname(os.path.dirname(_HERE))             # .../llm_eval

sys.path.insert(0, _RAG_ROOT)
sys.path.insert(0, _PROJ_ROOT)

# Load .env from the same directory as this script
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

from examples.common.utils import truncate_text

# Evaluator registry
EVALUATORS = {
    "retrieval",
    "response",
    "response_accuracy",
    "response_faithfulness",
    "response_intent",
    "response_completeness",
    "guardrails",
    "injection",
    "out_of_scope",
    "consistency",
}


# Shared helpers
def _load_samples(input_path: str) -> list:
    """Load samples from a JSON input file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _open_output(output_path: str):
    """Create output directory if needed and open file for writing."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return open(output_path, 'w', encoding='utf-8')


def _write_record(out_file, record: dict) -> None:
    """Write a single JSONL record and flush immediately."""
    out_file.write(json.dumps(record) + "\n")
    out_file.flush()


def _print_header(evaluator: str, input_path: str, output_path: str, **kwargs) -> None:
    """Print run header with config info."""
    print(f"Evaluator        : {evaluator}")
    print(f"Input            : {input_path}")
    print(f"Output           : {output_path}")
    for key, value in kwargs.items():
        if value is not None:
            print(f"{key:<17}: {value}")
    print("-" * 60)


def _print_summary(output_path: str, passed: int, total: int) -> None:
    """Print final summary line."""
    pct = 100 * passed / total if total else 0
    print("-" * 60)
    print(f"Results saved to : {output_path}")
    print(f"Summary          : {passed}/{total} passed ({pct:.1f}%)")


def _validate_input(input_path: Optional[str], env_var: str) -> None:
    """Exit with a clear message if input path is missing or file not found."""
    if not input_path:
        print(f"Error: {env_var} is not set in .env")
        print("Copy .env.example to .env and set the required variables.")
        sys.exit(1)
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        print(f"Check the {env_var} path in your .env file.")
        sys.exit(1)


def _validate_api_key(api_key: Optional[str], env_var: str) -> None:
    """Exit with a clear message if API key is missing."""
    if not api_key:
        print(f"Error: {env_var} is not set in .env")
        print("Copy .env.example to .env and set the required variables.")
        sys.exit(1)


# Retrieval accuracy
def run_retrieval() -> None:
    """Retrieval accuracy — embedding-based cosine similarity."""
    from rag.retrieval_accuracy_evaluator import RetrievalEvaluator, RetrievalEvaluationSample

    input_path  = os.environ.get("RETRIEVAL_EVAL_INPUT")
    output_path = os.environ.get("RETRIEVAL_EVAL_OUTPUT", "examples/rag/results/retrieval_results.jsonl")
    threshold   = float(os.environ.get("RETRIEVAL_EVAL_THRESHOLD", "0.5"))
    model       = os.environ.get("RETRIEVAL_EVAL_MODEL", "all-MiniLM-L6-v2")

    _validate_input(input_path, "RETRIEVAL_EVAL_INPUT")
    _print_header("retrieval", input_path, output_path, Model=model, Threshold=threshold)

    evaluator    = RetrievalEvaluator(relevance_threshold=threshold, model_name=model)
    samples      = _load_samples(input_path)
    passed_count = 0

    with _open_output(output_path) as out_file:
        for i, sample_data in enumerate(samples, 1):
            sample = RetrievalEvaluationSample.from_dict(sample_data)
            result = evaluator.evaluate_retrieval_accuracy(sample)
            _write_record(out_file, {"query_id": sample.query_id, "query": truncate_text(sample.query, 80), **result})
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{i}/{len(samples)}] {status} | {truncate_text(sample.query, 60)}")
            if result["passed"]: passed_count += 1

    _print_summary(output_path, passed_count, len(samples))


# Response evaluations (shared config loader)
def _response_config() -> tuple:
    """Load shared config for all response evaluators."""
    return (
        os.environ.get("RESPONSE_EVAL_INPUT"),
        os.environ.get("RESPONSE_EVAL_PROVIDER", "groq"),
        os.environ.get("RESPONSE_EVAL_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        os.environ.get("RESPONSE_EVAL_API_KEY"),
    )


def run_response() -> None:
    """All response evaluators combined — accuracy + faithfulness + intent."""
    from rag.response_evaluators import (
        ResponseEvaluationSample, LLMJudge,
        AccuracyEvaluator, FaithfulnessEvaluator, IntentAccuracyEvaluator, CompletenessEvaluator
    )

    input_path, provider, model, api_key = _response_config()
    output_path = os.environ.get("RESPONSE_EVAL_OUTPUT", "examples/rag/results/response_results.jsonl")

    _validate_input(input_path, "RESPONSE_EVAL_INPUT")
    _validate_api_key(api_key, "RESPONSE_EVAL_API_KEY")
    _print_header("response", input_path, output_path, Provider=provider, Model=model)

    judge                  = LLMJudge(provider=provider, model=model, api_key=api_key)
    accuracy_evaluator     = AccuracyEvaluator(judge=judge)
    faithfulness_evaluator = FaithfulnessEvaluator(judge=judge)
    intent_evaluator       = IntentAccuracyEvaluator(judge=judge)
    completeness_evaluator = CompletenessEvaluator(judge=judge)
    samples = _load_samples(input_path)
    accuracy_passed = faithfulness_passed = intent_passed = completeness_passed = 0

    with _open_output(output_path) as out_file:
        for i, sample_data in enumerate(samples, 1):
            sample             = ResponseEvaluationSample.from_dict(sample_data)
            accuracy_result    = accuracy_evaluator.evaluate_response_accuracy(sample)
            faithfulness_result = faithfulness_evaluator.evaluate_faithfulness(sample)
            intent_result      = intent_evaluator.evaluate_intent_accuracy(sample)
            completeness_result = completeness_evaluator.evaluate_completeness(sample)
            _write_record(out_file, {
                "query_id": sample.query_id,
                "query":    truncate_text(sample.query, 80),
                "evaluations": {
                    "accuracy":     accuracy_result,
                    "faithfulness": faithfulness_result,
                    "intent":       intent_result,
                    "completeness": completeness_result
                }
            })

            print(f"[{i}/{len(samples)}] query_id={sample.query_id} | {truncate_text(sample.query, 50)}")
            print(f"  Accuracy     : {'PASS' if accuracy_result['passed'] else 'FAIL'} — {accuracy_result['reason']}")
            print(f"  Faithfulness : {'PASS' if faithfulness_result['passed'] else 'FAIL'} — score={faithfulness_result['score']}")
            print(f"  Intent       : {'PASS' if intent_result['passed'] else 'FAIL'} — {intent_result['reason']}")
            print()

            if accuracy_result["passed"]:     accuracy_passed    += 1
            if faithfulness_result["passed"]: faithfulness_passed += 1
            if intent_result["passed"]:       intent_passed      += 1

    print("-" * 60)
    print(f"Results saved to : {output_path}")
    print(f"Accuracy         : {accuracy_passed}/{len(samples)} passed")
    print(f"Faithfulness     : {faithfulness_passed}/{len(samples)} passed")
    print(f"Intent           : {intent_passed}/{len(samples)} passed")


def run_response_accuracy() -> None:
    """Response accuracy — factual correctness vs ground truth."""
    from rag.response_evaluators import ResponseEvaluationSample, LLMJudge, AccuracyEvaluator

    input_path, provider, model, api_key = _response_config()
    output_path = os.environ.get("RESPONSE_ACCURACY_EVAL_OUTPUT", "examples/rag/results/response_accuracy_results.jsonl")
    output_path = output_path.replace(".jsonl", "_accuracy.jsonl") if "response_results" in output_path else output_path

    _validate_input(input_path, "RESPONSE_EVAL_INPUT")
    _validate_api_key(api_key, "RESPONSE_EVAL_API_KEY")
    _print_header("response_accuracy", input_path, output_path, Provider=provider, Model=model)

    evaluator    = AccuracyEvaluator(judge=LLMJudge(provider=provider, model=model, api_key=api_key))
    samples      = _load_samples(input_path)
    passed_count = 0

    with _open_output(output_path) as out_file:
        for i, sample_data in enumerate(samples, 1):
            sample = ResponseEvaluationSample.from_dict(sample_data)
            result = evaluator.evaluate_response_accuracy(sample)
            _write_record(out_file, {"query_id": sample.query_id, "query": truncate_text(sample.query, 80), **result})
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{i}/{len(samples)}] {status} | {truncate_text(sample.query, 60)}")
            print(f"  Reason: {result['reason']}")
            if result["passed"]: passed_count += 1

    _print_summary(output_path, passed_count, len(samples))


def run_response_faithfulness() -> None:
    """Response faithfulness — claim grounding against retrieved chunks."""
    from rag.response_evaluators import ResponseEvaluationSample, LLMJudge, FaithfulnessEvaluator

    input_path, provider, model, api_key = _response_config()
    output_path = os.environ.get("RESPONSE_FAITHFULNESS_EVAL_OUTPUT", "examples/rag/results/response_faithfulness_results.jsonl")
    output_path = output_path.replace(".jsonl", "_faithfulness.jsonl") if "response_results" in output_path else output_path

    _validate_input(input_path, "RESPONSE_EVAL_INPUT")
    _validate_api_key(api_key, "RESPONSE_EVAL_API_KEY")
    _print_header("response_faithfulness", input_path, output_path, Provider=provider, Model=model)

    evaluator    = FaithfulnessEvaluator(judge=LLMJudge(provider=provider, model=model, api_key=api_key))
    samples      = _load_samples(input_path)
    passed_count = 0

    with _open_output(output_path) as out_file:
        for i, sample_data in enumerate(samples, 1):
            sample = ResponseEvaluationSample.from_dict(sample_data)
            result = evaluator.evaluate_faithfulness(sample)
            _write_record(out_file, {"query_id": sample.query_id, "query": truncate_text(sample.query, 80), **result})
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{i}/{len(samples)}] {status} | score={result['score']} | claims={result['total_claims']} | {truncate_text(sample.query, 50)}")
            if result["unsupported_claims"]:
                for c in result["unsupported_claims"]:
                    print(f"   - {c}")
            if result["passed"]: passed_count += 1

    _print_summary(output_path, passed_count, len(samples))


def run_response_intent() -> None:
    """Response intent — whether response fulfills user intent."""
    from rag.response_evaluators import ResponseEvaluationSample, LLMJudge, IntentAccuracyEvaluator

    input_path, provider, model, api_key = _response_config()
    output_path = os.environ.get("RESPONSE_INTENT_EVAL_OUTPUT", "examples/rag/results/response_intent_results.jsonl")
    output_path = output_path.replace(".jsonl", "_intent.jsonl") if "response_results" in output_path else output_path

    _validate_input(input_path, "RESPONSE_EVAL_INPUT")
    _validate_api_key(api_key, "RESPONSE_EVAL_API_KEY")
    _print_header("response_intent", input_path, output_path, Provider=provider, Model=model)

    evaluator    = IntentAccuracyEvaluator(judge=LLMJudge(provider=provider, model=model, api_key=api_key))
    samples      = _load_samples(input_path)
    passed_count = 0

    with _open_output(output_path) as out_file:
        for i, sample_data in enumerate(samples, 1):
            sample = ResponseEvaluationSample.from_dict(sample_data)
            result = evaluator.evaluate_intent_accuracy(sample)
            _write_record(out_file, {"query_id": sample.query_id, "query": truncate_text(sample.query, 80), **result})
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{i}/{len(samples)}] {status} | {truncate_text(sample.query, 60)}")
            print(f"  Reason: {result['reason']}")
            if result["passed"]: passed_count += 1

    _print_summary(output_path, passed_count, len(samples))

def run_response_completeness() -> None:
    """Response completeness — whether response covers all relevant aspects."""
    from rag.response_evaluators import ResponseEvaluationSample, LLMJudge, CompletenessEvaluator

    input_path, provider, model, api_key = _response_config()
    output_path = os.environ.get("RESPONSE_COMPLETENESS_EVAL_OUTPUT", "examples/rag/results/response_completeness_results.jsonl")

    _validate_input(input_path, "RESPONSE_EVAL_INPUT")
    _validate_api_key(api_key, "RESPONSE_EVAL_API_KEY")
    _print_header("response_completeness", input_path, output_path, Provider=provider, Model=model)

    evaluator    = CompletenessEvaluator(judge=LLMJudge(provider=provider, model=model, api_key=api_key))
    samples      = _load_samples(input_path)
    passed_count = 0

    with _open_output(output_path) as out_file:
        for i, sample_data in enumerate(samples, 1):
            sample = ResponseEvaluationSample.from_dict(sample_data)
            result = evaluator.evaluate_completeness(sample)
            _write_record(out_file, {"query_id": sample.query_id, "query": truncate_text(sample.query, 80), **result})
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{i}/{len(samples)}] {status} | {truncate_text(sample.query, 60)}")
            print(f"  Reason: {result['reason']}")
            if result["passed"]: passed_count += 1

    _print_summary(output_path, passed_count, len(samples))

# Guardrails
def run_guardrails() -> None:
    """Guardrails — harmful/offensive content detection."""
    from guardrails_evaluator import GuardrailsEvaluator, GuardrailsSample

    input_path      = os.environ.get("GUARDRAILS_INPUT")
    output_path     = os.environ.get("GUARDRAILS_OUTPUT", "examples/common/results/guardrails_results.jsonl")
    provider        = os.environ.get("GUARDRAILS_PROVIDER", "openai_moderation")
    api_key         = os.environ.get("GUARDRAILS_API_KEY")
    threshold_raw   = os.environ.get("GUARDRAILS_SCORE_THRESHOLD")
    score_threshold = float(threshold_raw) if threshold_raw else None

    _validate_input(input_path, "GUARDRAILS_INPUT")
    _validate_api_key(api_key, "GUARDRAILS_API_KEY")
    _print_header("guardrails", input_path, output_path, Provider=provider, Threshold=score_threshold)

    evaluator    = GuardrailsEvaluator(provider=provider, api_key=api_key, score_threshold=score_threshold)
    samples      = _load_samples(input_path)
    passed_count = 0

    with _open_output(output_path) as out_file:
        for i, sample_data in enumerate(samples, 1):
            sample = GuardrailsSample.from_dict(sample_data)
            result = evaluator.evaluate_guardrails(sample)
            _write_record(out_file, {"query_id": sample.query_id, "query": truncate_text(sample.query, 80), **result})
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{i}/{len(samples)}] {status} | {truncate_text(sample.query, 60)}")
            if not result["passed"]:
                print(f"  Violated : {result['violated_categories']}")
            print(f"  Reason   : {result['reason']}")
            if result["passed"]: passed_count += 1

    _print_summary(output_path, passed_count, len(samples))


# Prompt injection
def run_injection() -> None:
    """Prompt injection resistance — three-layer evaluation."""
    from prompt_injection_evaluator import PromptInjectionEvaluator, PromptInjectionSample, LLMJudge

    input_path  = os.environ.get("INJECTION_INPUT")
    output_path = os.environ.get("INJECTION_OUTPUT", "examples/common/results/prompt_injection_results.jsonl")
    provider    = os.environ.get("INJECTION_PROVIDER", "groq")
    model       = os.environ.get("INJECTION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    api_key     = os.environ.get("INJECTION_API_KEY")

    _validate_input(input_path, "INJECTION_INPUT")
    # api_key is optional — Layer 3 skipped if not set

    layers_active = "1, 2, 3 (LLM judge)" if api_key else "1, 2 only (no API key — Layer 3 skipped)"
    _print_header("injection", input_path, output_path, Provider=provider, Model=model, Layers=layers_active)

    judge     = LLMJudge(provider=provider, model=model, api_key=api_key) if api_key else None
    evaluator = PromptInjectionEvaluator(judge=judge)
    samples   = _load_samples(input_path)

    passed_count   = 0
    layer_failures = {1: 0, 2: 0, 3: 0}

    with _open_output(output_path) as out_file:
        for i, sample_data in enumerate(samples, 1):
            sample = PromptInjectionSample.from_dict(sample_data)
            result = evaluator.evaluate_prompt_injection(sample)
            _write_record(out_file, {"query_id": sample.query_id, "query": truncate_text(sample.query, 80), **result})
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{i}/{len(samples)}] {status} | attack: {result['attack_type']} | {truncate_text(sample.query, 50)}")
            print(f"  Reason: {result['reason']}")
            if not result["passed"]:
                print(f"  Failed at Layer {result['failed_layer']}")
            if result["leakage_detected"]:
                print(f"  Leaked: {result['leaked_strings']}")
            if result["passed"]:
                passed_count += 1
            elif result["failed_layer"] in layer_failures:
                layer_failures[result["failed_layer"]] += 1

    print("-" * 60)
    print(f"Results saved to : {output_path}")
    print(f"Summary          : {passed_count}/{len(samples)} passed ({100*passed_count/len(samples):.1f}%)")
    print(f"Layer 1 failures : {layer_failures[1]}  (behavioral mismatch)")
    print(f"Layer 2 failures : {layer_failures[2]}  (sensitive content leakage)")
    print(f"Layer 3 failures : {layer_failures[3]}  (instruction fidelity)")


# Out of scope
def run_out_of_scope() -> None:
    """Out-of-scope accuracy — domain boundary enforcement."""
    from out_of_scope_evaluator import OutOfScopeEvaluator, OutOfScopeSample, LLMJudge

    input_path  = os.environ.get("OUT_OF_SCOPE_INPUT")
    output_path = os.environ.get("OUT_OF_SCOPE_OUTPUT", "results/out_of_scope_results.jsonl")
    provider    = os.environ.get("OUT_OF_SCOPE_PROVIDER", "groq")
    model       = os.environ.get("OUT_OF_SCOPE_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    api_key     = os.environ.get("OUT_OF_SCOPE_API_KEY")

    _validate_input(input_path, "OUT_OF_SCOPE_INPUT")
    _validate_api_key(api_key, "OUT_OF_SCOPE_API_KEY")
    _print_header("out_of_scope", input_path, output_path, Provider=provider, Model=model)

    evaluator        = OutOfScopeEvaluator(judge=LLMJudge(provider=provider, model=model, api_key=api_key))
    samples          = _load_samples(input_path)
    passed_count     = 0
    category_stats   = defaultdict(lambda: {"passed": 0, "total": 0})
    behavior_stats   = defaultdict(lambda: {"passed": 0, "total": 0})

    with _open_output(output_path) as out_file:
        for i, sample_data in enumerate(samples, 1):
            sample = OutOfScopeSample.from_dict(sample_data)
            result = evaluator.evaluate_out_of_scope(sample)
            _write_record(out_file, {"query_id": sample.query_id, "query": truncate_text(sample.query, 80), **result})
            status   = "PASS" if result["passed"] else "FAIL"
            print(f"[{i}/{len(samples)}] {status} | expected: {result['expected_behavior']}")
            print(f"  Query  : {truncate_text(sample.query, 70)}")
            print(f"  Reason : {result['reason']}")
            print()

            if result["passed"]: passed_count += 1

            beh = result["expected_behavior"]
            behavior_stats[beh]["total"] += 1
            if result["passed"]: behavior_stats[beh]["passed"] += 1

    print("-" * 60)
    print(f"Results saved to : {output_path}")
    print(f"Summary          : {passed_count}/{len(samples)} passed ({100*passed_count/len(samples):.1f}%)")
    print()
    print("By expected behavior:")
    for beh, stats in sorted(behavior_stats.items()):
        pct = 100 * stats["passed"] / stats["total"] if stats["total"] else 0
        print(f"  {beh:<15} : {stats['passed']}/{stats['total']} ({pct:.0f}%)")
    if any(k != "uncategorized" for k in category_stats):
        print()
        print("By query category:")
        for cat, stats in sorted(category_stats.items()):
            pct = 100 * stats["passed"] / stats["total"] if stats["total"] else 0
            print(f"  {cat:<20} : {stats['passed']}/{stats['total']} ({pct:.0f}%)")


# Consistency
def run_consistency() -> None:
    """Consistency — factual and quality consistency across paraphrased queries."""
    from consistency_evaluator import ConsistencyEvaluator, ConsistencyGroup, LLMJudge

    input_path  = os.environ.get("CONSISTENCY_INPUT")
    output_path = os.environ.get("CONSISTENCY_OUTPUT", "results/consistency_results.jsonl")
    provider    = os.environ.get("CONSISTENCY_PROVIDER", "groq")
    model       = os.environ.get("CONSISTENCY_MODEL", "llama-3.3-70b-versatile")
    api_key     = os.environ.get("CONSISTENCY_API_KEY")

    _validate_input(input_path, "CONSISTENCY_INPUT")
    _validate_api_key(api_key, "CONSISTENCY_API_KEY")
    _print_header("consistency", input_path, output_path, Provider=provider, Model=model)

    evaluator    = ConsistencyEvaluator(judge=LLMJudge(provider=provider, model=model, api_key=api_key))
    groups       = _load_samples(input_path)
    passed_count = 0
    factual_passed = 0
    quality_passed = 0

    with _open_output(output_path) as out_file:
        for i, group_data in enumerate(groups, 1):
            group  = ConsistencyGroup.from_dict(group_data)
            result = evaluator.evaluate_consistency(group)

            _write_record(out_file, {
                "group_id":   group.group_id,
                "intent":     group.intent,
                "query_count": len(group.queries),
                **result,
            })

            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{i}/{len(groups)}] {status} | group_id={group.group_id} | queries={len(group.queries)}")
            print(f"  Intent   : {group.intent}")
            print(f"  Factual  : {'PASS' if result['factual_consistency']['passed'] else 'FAIL'} — {result['factual_consistency']['reason']}")
            print(f"  Quality  : {'PASS' if result['quality_consistency']['passed'] else 'FAIL'} — {result['quality_consistency']['reason']}")
            if result["contradictions"]:
                for c in result["contradictions"]:
                    print(f"   ✗ {c}")
            print()

            if result["passed"]:                              passed_count   += 1
            if result["factual_consistency"]["passed"]:       factual_passed += 1
            if result["quality_consistency"]["passed"]:       quality_passed += 1

    print("-" * 60)
    print(f"Results saved to    : {output_path}")
    print(f"Summary             : {passed_count}/{len(groups)} passed ({100*passed_count/len(groups):.1f}%)")
    print(f"Factual consistency : {factual_passed}/{len(groups)} passed")
    print(f"Quality consistency : {quality_passed}/{len(groups)} passed")

# Dispatch
DISPATCH = {
    "retrieval":             run_retrieval,
    "response":              run_response,
    "response_accuracy":     run_response_accuracy,
    "response_faithfulness": run_response_faithfulness,
    "response_intent":       run_response_intent,
    "response_completeness": run_response_completeness,
    "guardrails":            run_guardrails,
    "injection":             run_injection,
    "out_of_scope":          run_out_of_scope,
    "consistency":           run_consistency,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="LLM Evaluation Framework — run any evaluator from a single entry point.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "evaluator",
        choices=sorted(EVALUATORS),
        metavar="evaluator",
        help=f"Evaluator to run. One of: {', '.join(sorted(EVALUATORS))}"
    )

    args = parser.parse_args()

    print(f"Running: {args.evaluator}")
    print("=" * 60)
    DISPATCH[args.evaluator]()


if __name__ == "__main__":
    main()