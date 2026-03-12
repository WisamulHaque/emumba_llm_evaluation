"""
Parameter Accuracy Evaluator

Validates that the parameters constructed by the LLM for an API call are
correct and match the requirements of the user's query. Evaluation runs in
two sequential layers:

  Layer 1 — Deterministic (structural): checks required field presence, data
  types, and unknown fields against the API schema. Any failure in this layer 
  results in an immediate fail verdict with no need for Layer 2, as semantic 
  evaluation is meaningless if the structure is invalid.

  Layer 2 — LLM-as-judge (semantic): only reached when structure is valid.
  The judge verifies that each parameter value is logically correct and
  consistent with what the user actually asked for (e.g. correct date
  extraction, right numeric value, sensible enum choice).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Any
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the parameter values constructed for an API call are
semantically correct and logically consistent with the user's query.
The structural validity of the parameters (required fields, data types)
has already been confirmed before reaching this evaluation.

Following inputs will be provided:

Inputs:
- Query: The original user query submitted to the application
- API Schema: The definition of the API being called, including each parameter
  name, type, and description
- Generated Parameters: The parameter key-value pairs constructed by the LLM

Query:
{query}

API Schema:
{schema_block}

Generated Parameters:
{params_block}

Optional Parameters Populated:
{optional_block}

Based on the above-provided inputs, use the rules below to evaluate parameter accuracy:

Evaluation Rules:
1. Each parameter value must be logically derived from the query, a value that
   contradicts or is unsupported by what the user asked is a failure
2. Optional parameters must only be populated when the user's query contains explicit wording that directly maps to that parameter's value.
   If the query text does not mention an explicit value for an optional parameter,
   its presence in the generated params is always a failure, regardless of
   whether the value seems reasonable, common, or a sensible default.
   Example: "Book economy flights" -> cabin_class: "economy" is valid.
   Example: "Find flights to Paris" -> cabin_class: "business" is a failure.
3. Numeric and quantity values must exactly match what the user stated, rounding
   or substituting a different number without basis in the query is a failure
4. Date and time values must correctly reflect the temporal expression in the
   query (e.g. "next Friday", "two nights from this Saturday"), a misresolved
   date is a failure
5. String values such as city names, identifiers, and categories must match the
   query's intent exactly, a substituted or paraphrased value that changes
   meaning is a failure
6. Evaluate solely based on the query text and schema descriptions provided,
   do not use outside knowledge or assumptions beyond what is stated

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if all parameter values are accurate and logically correct, false otherwise>,
    "reason": "<one or two sentences explaining the overall verdict>",
    "issues": ["<describe each identified problem with a parameter value, or empty list if none>"]
}}"""


# Python type map used for deterministic type checking in Layer 1
_TYPE_MAP = {
    "string":  str,
    "integer": int,
    "float":   float,
    "boolean": bool,
    "list":    list,
    "dict":    dict,
}


def evaluate_parameter_accuracy(
    judge: LLMJudge,
    query: str,
    api_schema: List[Dict[str, Any]],
    generated_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate whether the generated API parameters are structurally and
    semantically correct for the user's query.

    Args:
        judge:             Configured LLMJudge instance
        query:             The original user query
        api_schema:        List of parameter definitions, each a dict with:
                             - name (str): parameter key
                             - type (str): expected type — "string", "integer",
                               "float", "boolean", "list", or "dict"
                             - required (bool): whether the field is mandatory
                             - description (str): what the parameter represents
        generated_params:  The parameter dict constructed by the LLM

    Returns a dict with:
        - passed:          True if all checks pass, False otherwise
        - reason:          Summary of the verdict
        - issues:          List of identified problems (empty if passed)
        - layer:           "structural" if failed in Layer 1, "semantic" if
                           failed in Layer 2, "none" if fully passed
    """

    # --- Layer 1: Structural Validation (deterministic, no LLM) ---

    structural_issues = []
    schema_keys = {param["name"] for param in api_schema}

    # Check for unknown fields not defined in the schema
    for key in generated_params:
        if key not in schema_keys:
            structural_issues.append(
                f"Unknown parameter '{key}' is not defined in the API schema."
            )

    # Check required fields and data types
    for param in api_schema:
        name = param["name"]
        expected_type_str = param.get("type", "string")
        is_required = param.get("required", False)

        if name not in generated_params:
            if is_required:
                structural_issues.append(
                    f"Required parameter '{name}' is missing from the generated parameters."
                )
            continue  # optional and absent — nothing more to check

        value = generated_params[name]
        expected_type = _TYPE_MAP.get(expected_type_str)

        if expected_type is not None and not isinstance(value, expected_type):
            structural_issues.append(
                f"Parameter '{name}' has type {type(value).__name__!r} "
                f"but expected {expected_type_str!r}."
            )

    # Structural failures make semantic evaluation meaningless
    if structural_issues:
        return {
            "passed": False,
            "reason": f"{len(structural_issues)} structural issue(s) found in the generated parameters.",
            "issues": structural_issues,
            "layer": "structural",
        }

    # --- Layer 2: Semantic Validation (LLM-as-judge) ---
    # Only reached when all structural checks pass.

    schema_block = "\n".join(
        f"  - {p['name']} ({p['type']}, {'required' if p.get('required') else 'optional'}): {p['description']}"
        for p in api_schema
    )
    params_block = "\n".join(
        f"  - {k}: {v!r}"
        for k, v in generated_params.items()
    )

    optional_params_present = [
        p for p in api_schema
        if not p.get("required") and p["name"] in generated_params
    ]
    optional_block = "\n".join(
        f"  - {p['name']}: {generated_params[p['name']]!r}  — verify this value appears explicitly in the query"
        for p in optional_params_present
    ) or "  (none)"

    prompt = EVAL_PROMPT.format(
        query=query,
        schema_block=schema_block,
        params_block=params_block,
        optional_block=optional_block,
    )
    result = judge.judge(prompt)

    passed = result.get("passed", False)
    if isinstance(passed, str):
        passed = passed.lower() == "true"

    issues = result.get("issues", [])

    return {
        "passed": bool(passed),
        "reason": result.get("reason", ""),
        "issues": issues if isinstance(issues, list) else [],
        "layer": "none" if passed else "semantic",
    }


# API schema shared across test samples
FLIGHT_SCHEMA = [
    {"name": "origin",           "type": "string",  "required": True,  "description": "IATA airport code for the departure airport"},
    {"name": "destination",      "type": "string",  "required": True,  "description": "IATA airport code for the arrival airport"},
    {"name": "departure_date",   "type": "string",  "required": True,  "description": "Departure date in YYYY-MM-DD format"},
    {"name": "passenger_count",  "type": "integer", "required": True,  "description": "Number of passengers travelling"},
    {"name": "cabin_class",      "type": "string",  "required": False, "description": "Cabin class: 'economy', 'business', or 'first'"},
]

# Test data
SAMPLES = [
    {
        # All required params present with correct types and values (expected: PASS)
        "query_id": 1,
        "query": "Book a flight from Karachi to Dubai for 2 passengers on March 13th, 2026.",
        "api_schema": FLIGHT_SCHEMA,
        "generated_params": {
            "origin":          "KHI",
            "destination":     "DXB",
            "departure_date":  "2026-03-13",
            "passenger_count": 2,
        },
    },
    {
        # Required field departure_date is missing entirely (expected: FAIL)
        "query_id": 2,
        "query": "Search for flights from Dubai to Singapore for 1 passenger on March 20th, 2026.",
        "api_schema": FLIGHT_SCHEMA,
        "generated_params": {
            "origin":          "DXB",
            "destination":     "SIN",
            "passenger_count": "1",
        },
    },
    {
        # Structure is valid but passenger_count is 1 while query says 4 (expected: FAIL)
        "query_id": 3,
        "query": "Book economy flight from Tokyo to Sydney for 4 passengers on June 1st, 2026.",
        "api_schema": FLIGHT_SCHEMA,
        "generated_params": {
            "origin":          "NRT",
            "destination":     "SYD",
            "departure_date":  "2026-06-01",
            "passenger_count": 1,
            "cabin_class":     "economy",
        },
    },
    {
        # Structure valid; optional cabin_class invented — query never mentioned cabin preference (expected: FAIL)
        "query_id": 4,
        "query": "Find flights from Karachi to Istanbul for 2 passengers on July 15th, 2026.",
        "api_schema": FLIGHT_SCHEMA,
        "generated_params": {
            "origin":          "KHI",
            "destination":     "IST",
            "departure_date":  "2026-07-15",
            "passenger_count": 2,
            "cabin_class":     "business",
        },
    },
]


def main():
    from dotenv import load_dotenv
    load_dotenv()

    judge = LLMJudge(
        provider=os.environ.get("LLM_JUDGE_PROVIDER", "groq"),
        model=os.environ.get("LLM_JUDGE_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        api_key=os.environ["LLM_JUDGE_API_KEY"],
    )

    for sample in SAMPLES:
        result = evaluate_parameter_accuracy(
            judge,
            query=sample["query"],
            api_schema=sample["api_schema"],
            generated_params=sample["generated_params"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}  (layer: {result['layer']})")
        print(f"  Query  : {sample['query']}")
        print(f"  Reason : {result['reason']}")
        if result["issues"]:
            print(f"  Issues:")
            for i, issue in enumerate(result["issues"], 1):
                print(f"    {i}. {issue}")
    print()


if __name__ == "__main__":
    main()
