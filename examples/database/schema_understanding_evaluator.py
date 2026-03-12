"""
Schema Understanding Evaluator

Validates that the LLM correctly identifies and uses the right tables,
columns, relationships, and joins relevant to the user's natural language
query. Focuses exclusively on schema element selection, whether the correct
schema identifiers were chosen rather than overall query correctness.
Uses LLM-as-judge to reason over schema adherence without requiring a live
database connection.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any, List
from examples.common.llm_judge import LLMJudge


EVAL_PROMPT = """Purpose:
Evaluate whether the generated database query correctly identifies and uses
the right tables/collections, columns/fields, relationships, and joins from
the provided schema to answer the user's natural language query.

Following inputs will be provided:

Inputs:
- Natural Language Query: The original request submitted by the user
- Schema: Full description of available tables/collections, their columns/fields,
  data types, and relationships
- Generated Query: The database query produced by the application

Natural Language Query:
{query}

Schema:
{schema}

Generated Query:
{generated_query}

Based on the above-provided inputs, use the rules below to evaluate schema understanding:

Evaluation Rules:
1. Table/collection selection; the query must reference only the tables or
   collections from the Schema that are actually needed to answer the Natural Language Query;
   including irrelevant tables or omitting a required table is a failure
2. Column/field selection; every column or field referenced in SELECT, WHERE,
   GROUP BY, ORDER BY, or projection clauses must exist in the Schema under the
   table/collection it is attributed to; referencing a non-existent or
   misattributed column is a failure
3. Join correctness; when the query joins tables, it must use the correct
   foreign key to primary key pairs as defined in the Schema; joining on the
   wrong columns even if the syntax is valid is a failure
4. Relationship modeling; the query must correctly model multi-table or
   multi-collection relationships implied by the Natural Language Query; missing
   a required join or lookup that the Schema clearly supports is a failure
5. Hallucinated identifiers; any table, collection, column, or field that does
   not exist in the provided Schema is a hallucination and is a failure
6. Evaluate solely against the provided Schema; do not use outside knowledge
   about what tables or fields might exist in a real-world database

Output Format:
Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "passed": <true if schema usage is correct, false otherwise>,
    "reason": "<one or two sentences summarising the overall verdict>",
    "issues": ["<describe each misused, missing, or hallucinated schema element, or empty list if none>"]
}}"""


def evaluate_schema_understanding(
    judge: LLMJudge,
    query: str,
    generated_query: str,
    schema: str,
) -> Dict[str, Any]:
    """
    Evaluate whether the generated query correctly identifies and uses the
    right schema elements (tables, columns, joins, relationships).

    Args:
        judge:           Configured LLMJudge instance
        query:           The original natural language user request
        generated_query: The SQL/NoSQL query produced by the application
        schema:          Plain-text description of all available tables/collections,
                         columns/fields, data types, and relationships

    Returns a dict with:
        - passed:           True if schema usage is correct, False otherwise
        - reason:           LLM judge explanation
        - issues: List of misused, missing, or hallucinated schema
                            elements (empty list if passed)
    """
    if not generated_query or not generated_query.strip():
        return {
            "passed": False,
            "reason": "No query was generated.",
            "issues": ["The generated query is empty."],
        }
    if not schema or not schema.strip():
        return {
            "passed": False,
            "reason": "No schema was provided.",
            "issues": ["Schema is required to evaluate schema understanding."],
        }

    prompt = EVAL_PROMPT.format(
        query=query,
        schema=schema,
        generated_query=generated_query,
    )
    result = judge.judge(prompt)

    passed = result.get("passed", 0)
    if isinstance(passed, str):
        passed = int(passed) if passed.isdigit() else 0

    issues = result.get("issues", [])

    return {
        "passed": int(passed) == 1,
        "reason": result.get("reason", ""),
        "issues": issues if isinstance(issues, list) else [],
    }


# Test data
SQL_SCHEMA = """
Table: employees
  - employee_id   INTEGER   PRIMARY KEY
  - name          VARCHAR(100)
  - department_id INTEGER   FOREIGN KEY -> departments.department_id
  - salary        DECIMAL(10,2)
  - hire_date     DATE

Table: departments
  - department_id INTEGER   PRIMARY KEY
  - dept_name     VARCHAR(100)
  - location      VARCHAR(100)
"""

NOSQL_SCHEMA = """
Collection: products
  - _id          ObjectId  PRIMARY KEY
  - name         String
  - category     String
  - price        Number
  - in_stock     Boolean

Collection: reviews
  - _id          ObjectId  PRIMARY KEY
  - product_id   ObjectId  REFERENCE -> products._id
  - rating       Number    (1–5)
  - review_text  String
  - created_at   Date
"""

SAMPLES = [
    {
        # SQL — correct tables, columns, and join on the right FK/PK pair (expected: PASS)
        "query_id": 1,
        "query": "List the names of all employees in the Engineering department along with their salaries.",
        "schema": SQL_SCHEMA,
        "generated_query": (
            "SELECT e.name, e.salary "
            "FROM employees e "
            "JOIN departments d ON e.department_id = d.department_id "
            "WHERE d.dept_name = 'Engineering';"
        ),
    },
    {
        # SQL — joins on a non-existent column 'dept_code' instead of 'department_id' (expected: FAIL)
        "query_id": 2,
        "query": "Show the location of the department each employee belongs to.",
        "schema": SQL_SCHEMA,
        "generated_query": (
            "SELECT e.name, d.location "
            "FROM employees e "
            "JOIN departments d ON e.dept_code = d.dept_code;"
        ),
    },
    {
        # NoSQL — correct collection lookup using the right reference field (expected: PASS)
        "query_id": 3,
        "query": "Find all reviews with a rating above 4 and include the name of the reviewed product.",
        "schema": NOSQL_SCHEMA,
        "generated_query": (
            "db.reviews.aggregate(["
            "  { $match: { rating: { $gt: 4 } } },"
            "  { $lookup: { from: 'products', localField: 'product_id', foreignField: '_id', as: 'product' } },"
            "  { $unwind: '$product' },"
            "  { $project: { rating: 1, review_text: 1, 'product.name': 1 } }"
            "]);"
        ),
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
        result = evaluate_schema_understanding(
            judge,
            query=sample["query"],
            generated_query=sample["generated_query"],
            schema=sample["schema"],
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{sample['query_id']}] {status}")
        print(f"  Query  : {sample['query']}")
        print(f"  Reason : {result['reason']}")
        if result["issues"]:
            print(f"  Issues:")
            for i, elem in enumerate(result["issues"], 1):
                print(f"    {i}. {elem}")
    print()


if __name__ == "__main__":
    main()
