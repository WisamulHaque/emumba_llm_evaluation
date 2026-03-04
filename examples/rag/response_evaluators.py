"""
Response Evaluators for RAG Applications

This module contains evaluators for assessing the quality of LLM-generated
responses in RAG (Retrieval-Augmented Generation) applications. All evaluators
here operate on the generated response — they are post-generation checks,
complementing the pre-generation retrieval_evaluation.py checks.

Evaluators included:
    - ResponseAccuracyEvaluator: Checks factual accuracy of response against ground truth
    - FaithfulnessEvaluator:        Every claim in response is supported by retrieved chunks
    - IntentAccuracyEvaluator:      Response fulfills the actual intent of the query
    - ResponseCompletenessEvaluator: Response covers all relevant aspects of the query

Evaluators to be added:
    - CitationAccuracyEvaluator:    Cited sources actually contain the attributed information

All evaluators use LLM-as-judge since response quality requires nuanced reasoning
that embedding similarity or rule-based methods cannot reliably provide.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import os
import re

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# LLM provider imports — each is optional, error raised only if used
try:
    import anthropic as anthropic_sdk
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import openai as openai_sdk
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


# Data model
@dataclass
class ResponseEvaluationSample:
    """
    Input sample for response-level RAG evaluations.

    Attributes:
        query_id:           Unique identifier for the query
        query:              The user's original question
        retrieved_chunks:   Chunks retrieved from the document store
        generated_response: The LLM's response to evaluate
        ground_truth:       The expected correct answer (used by ResponseAccuracyEvaluator)
        metadata:           Optional additional metadata
    """
    query_id: int
    query: str
    retrieved_chunks: List[str]
    generated_response: str
    ground_truth: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResponseEvaluationSample":
        """Create sample from dictionary (for JSON input)."""
        return cls(
            query_id=data.get("query_id", 0),
            query=data["query"],
            retrieved_chunks=data.get("retrieved_chunks", []),
            generated_response=data["generated_response"],
            ground_truth=data.get("ground_truth"),
            metadata=data.get("metadata", {})
        )


# LLM Judge base
class LLMJudge:
    """
    Provider-agnostic LLM judge for evaluation.

    Supports Anthropic, OpenAI, and Groq. Provider and model are configured
    via .env — no code changes needed to switch between them.

    All judge calls expect the LLM to return a JSON object with:
        {
            "score": int,   // 0 or 1
            "reason": str     // explanation of the verdict
        }
    """

    SUPPORTED_PROVIDERS = ("anthropic", "openai", "groq")

    def __init__(self, provider: str, model: str, api_key: str):
        """
        Initialize the LLM judge.

        Args:
            provider: LLM provider — "anthropic", "openai", or "groq"
            model:    Model name, e.g. "claude-3-haiku-20240307", "gpt-4o-mini", "llama3-8b-8192"
            api_key:  API key for the provider
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self._client = None

        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Choose from: {self.SUPPORTED_PROVIDERS}"
            )
        self._validate_provider_installed()

    def _validate_provider_installed(self) -> None:
        """Raise a clear error if the required SDK is not installed."""
        if self.provider == "anthropic" and not _ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic SDK not installed.\nRun: pip install anthropic")
        if self.provider == "openai" and not _OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK not installed.\nRun: pip install openai")
        if self.provider == "groq" and not _GROQ_AVAILABLE:
            raise ImportError("Groq SDK not installed.\nRun: pip install groq")

    def judge(self, prompt: str) -> Dict[str, Any]:
        """
        Send a prompt to the configured LLM and parse the JSON response.

        Args:
            prompt: Full evaluation prompt — should instruct the model to
                    return only a JSON object with "score" and "reason" keys
        Returns:
            Parsed dict with "score" (int) and "reason" (str)
        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON
        """
        raw_response = self._call_provider(prompt)
        return self._parse_response(raw_response)
    
    def _get_client(self):
        """Get or create the LLM client (lazy initialization)."""
        if self._client is None:
            if self.provider == "anthropic":
                self._client = anthropic_sdk.Anthropic(api_key=self.api_key)
            elif self.provider == "openai":
                self._client = openai_sdk.OpenAI(api_key=self.api_key)
            elif self.provider == "groq":
                self._client = Groq(api_key=self.api_key)
        return self._client

    def _call_provider(self, prompt: str, max_tokens: int = 256) -> str:
        """Call the configured provider and return raw text response."""
        if self.provider == "anthropic":
            client = self._get_client()
            message = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        if self.provider == "openai":
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        if self.provider == "groq":
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

    def _parse_response(self, raw: str) -> Dict[str, Any]:

        # Strip markdown code fences
        clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        if not clean:
            raise ValueError(f"LLM returned empty response after stripping fences.\nRaw: {raw}")

        # Use regex to find JSON object
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in LLM response.\nCleaned: {clean}")

        json_text = match.group(0)

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM response was not valid JSON.\nExtracted text:\n{json_text}\nOriginal raw:\n{raw}\nError: {e}"
            )

        # Support expected evaluation formats
        if ("score" in parsed and "reason" in parsed) or \
        ("claims" in parsed) or \
        ("results" in parsed) or \
        ("passed" in parsed and "reason" in parsed):
            return parsed

        raise ValueError(f"LLM response missing required keys.\nParsed: {parsed}")

# Response Accuracy Evaluator
class ResponseAccuracyEvaluator:
    """
    Evaluator for RAG response accuracy.

    Checks whether the LLM's generated response is factually accurate when
    compared against the ground truth answer and uses LLM-as-judge 
    """

    EVAL_PROMPT_TEMPLATE = """You are an expert evaluator assessing the factual accuracy of an AI-generated response.

Compare the generated response against the ground truth and evaluate whether the response is factually accurate.

Query: {query}

Ground Truth: {ground_truth}

Generated Response: {generated_response}

Instructions:
- Focus only on factual correctness — not style, tone, or completeness
- A response is accurate if its factual claims are consistent with the ground truth
- A response fails if it contains any factual errors, contradictions, or misleading information
- Minor wording differences are acceptable as long as the facts are correct
- Score 1 if the response is factually accurate, 0 if it contains any factual errors

Respond with ONLY a JSON object in this exact format, no explanation outside the JSON:
{{
    "score": <1 or 0>,
    "reason": "<one or two sentences explaining the verdict>"
}}"""

    def __init__(self, judge: LLMJudge):
        """
        Initialize the evaluator.

        Args:
            judge: Configured LLMJudge instance
        """
        self.judge = judge

    def evaluate_response_accuracy(
        self,
        sample: ResponseEvaluationSample
    ) -> Dict[str, Any]:
        """
        Evaluate whether the generated response is factually accurate.

        Args:
            sample: ResponseEvaluationSample containing query, response, and ground truth

        Returns:
            Dictionary with:
                - passed: True if response is accurate, False otherwise
                - reason: Explanation of the verdict from the LLM judge
        """
        if not sample.ground_truth:
            return {
                "passed": False,
                "reason": "No ground truth provided — cannot evaluate response accuracy",
            }

        if not sample.generated_response:
            return {
                "passed": False,
                "reason": "No generated response provided",
            }

        prompt = self.EVAL_PROMPT_TEMPLATE.format(
            query=sample.query,
            ground_truth=sample.ground_truth,
            generated_response=sample.generated_response
        )

        result = self.judge.judge(prompt)
        passed = int(result["score"]) == 1

        return {
            "passed": passed,
            "reason": result["reason"]
        }
    
class FaithfulnessEvaluator:
    """
    Evaluator for RAG faithfulness (hallucination detection) with optimized batching.
    """

    CLAIM_EXTRACTION_PROMPT = """You are extracting factual claims from an AI-generated response.

Extract all atomic factual claims from the response below.

Rules:
- Only extract statements that assert factual information.
- Ignore opinions or stylistic commentary.
- Break compound statements into separate claims.
- Return ONLY a JSON object in this format:

{{
  "claims": ["claim 1", "claim 2", "..."]
}}

Response:
{generated_response}
"""

    CLAIM_VERIFICATION_PROMPT = """You are evaluating whether claims are supported by retrieved context.

Retrieved Context:
{retrieved_context}

Claims:
{claims_list}

Instructions:
- For each claim, determine whether it is explicitly supported by the retrieved context.
- If the context does not clearly support a claim, mark it as unsupported.
- Even partially supported claims should be marked as unsupported.
- Do NOT use outside knowledge.

Return ONLY a JSON object in this format:

{{
  "results": [
    {{
      "claim": "claim text",
      "score": 1 or 0,
      "reason": "short explanation"
    }}
  ]
}}
"""

    def __init__(self, judge: LLMJudge, batch_size: int = 5):
        self.judge = judge
        self.batch_size = batch_size

    def _extract_claims(self, response: str) -> List[str]:
        if not response.strip():
            return []

        prompt = self.CLAIM_EXTRACTION_PROMPT.format(
            generated_response=response
        )

        result = self.judge.judge(prompt)
        claims = result.get("claims")

        if not isinstance(claims, list):
            raise ValueError("Claim extraction failed: 'claims' must be a list.")

        return [c.strip() for c in claims if c.strip()]

    def _verify_claim_batch(self, claims_batch: List[str], retrieved_context: str) -> List[Dict[str, Any]]:
        claims_list = "\n".join([f"- {c}" for c in claims_batch])
        prompt = self.CLAIM_VERIFICATION_PROMPT.format(
            retrieved_context=retrieved_context,
            claims_list=claims_list
        )
        result = self.judge.judge(prompt)
        results = result.get("results")

        if not isinstance(results, list):
            raise ValueError("Batch verification failed: 'results' must be a list.")
        return results

    def evaluate_faithfulness(self, sample: ResponseEvaluationSample) -> Dict[str, Any]:
        if not sample.generated_response:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "No generated response provided",
                "total_claims": 0,
                "unsupported_claims": []
            }

        if not sample.retrieved_chunks:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "No retrieved context provided",
                "total_claims": 0,
                "unsupported_claims": []
            }

        # Step 1: Extract claims
        claims = self._extract_claims(sample.generated_response)
        if not claims:
            return {
                "passed": True,
                "score": 1.0,
                "reason": "No factual claims detected in response",
                "total_claims": 0,
                "unsupported_claims": []
            }

        # Step 2: Batch verification
        retrieved_context = "\n\n".join(sample.retrieved_chunks)
        unsupported_claims = []
        supported_count = 0

        # Split claims into batches
        for i in range(0, len(claims), self.batch_size):
            batch = claims[i:i+self.batch_size]
            try:
                results = self._verify_claim_batch(batch, retrieved_context)
            except (ValueError, json.JSONDecodeError) as e:
                # If batch fails, mark all claims in batch as unsupported
                unsupported_claims.extend(batch)
                continue

            for item in results:
                if int(item.get("score", 0)) == 1:
                    supported_count += 1
                else:
                    unsupported_claims.append(item.get("claim", "Unknown claim"))

        total_claims = len(claims)
        score = supported_count / total_claims
        passed = len(unsupported_claims) == 0
        reason = f"{supported_count} out of {total_claims} claims are supported by the retrieved context."

        return {
            "passed": passed,
            "score": round(score, 2),
            "reason": reason,
            "total_claims": total_claims,
            "unsupported_claims": unsupported_claims
        }
    
# Intent Accuracy Evaluator
class IntentAccuracyEvaluator:
    """
    Evaluator for RAG intent accuracy.

    Ensures that the generated response fulfills the actual intent of the user's query,
    rather than just superficially matching keywords.
    """

    EVAL_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether an AI-generated response
fulfills the actual intent behind a user's query.

Query: {query}

Generated Response: {generated_response}

Instructions:
- Determine whether the response fully addresses the user's true intent.
- Do not rely on keyword matches; consider the semantic meaning of the query.
- A response passes if it correctly answers what the user is asking, covering all necessary aspects.
- A response fails if it misunderstands the intent, omits important parts, or is off-topic.

Respond with ONLY a JSON object in this exact format:
{{
    "passed": true or false,
    "reason": "<one or two sentences explaining the verdict>"
}}
"""

    def __init__(self, judge: LLMJudge):
        """
        Initialize the evaluator.

        Args:
            judge: Configured LLMJudge instance
        """
        self.judge = judge

    def evaluate_intent_accuracy(
        self,
        sample: ResponseEvaluationSample
    ) -> Dict[str, Any]:
        """
        Evaluate whether the generated response fulfills the user's intent.

        Args:
            sample: ResponseEvaluationSample containing query and response

        Returns:
            Dictionary with:
                - passed: True if intent is fulfilled, False otherwise
                - reason: Explanation of the verdict from the LLM judge
        """
        if not sample.generated_response:
            return {
                "passed": False,
                "reason": "No generated response provided — cannot evaluate intent accuracy",
            }

        prompt = self.EVAL_PROMPT_TEMPLATE.format(
            query=sample.query,
            generated_response=sample.generated_response
        )

        result = self.judge.judge(prompt)

        # Ensure JSON contains required keys
        if "passed" not in result or "reason" not in result:
            raise ValueError(
                f"Intent accuracy evaluation failed. LLM returned invalid JSON: {result}"
            )

        return {
            "passed": result["passed"],
            "reason": result["reason"]
        }
    
# Response Completeness Evaluator
class ResponseCompletenessEvaluator:
    """
    Evaluator for RAG response completeness.

    Ensures that the generated response covers all relevant aspects of the query
    and does not omit critical information present in the retrieved chunks.
    """

    EVAL_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether an AI-generated response
fully covers all relevant information from the retrieved context to answer a user's query.

Query: {query}

Retrieved Chunks:
{retrieved_context}

Generated Response:
{generated_response}

Instructions:
- Check if the response covers all critical points relevant to the query.
- A response passes if it addresses all necessary aspects from the retrieved chunks.
- A response fails if it omits important details or leaves out key information.

Respond with ONLY a JSON object in this exact format:
{{
    "passed": true or false,
    "reason": "<one or two sentences explaining what's missing or confirming completeness>"
}}
"""

    def __init__(self, judge: LLMJudge):
        """
        Initialize the evaluator.

        Args:
            judge: Configured LLMJudge instance
        """
        self.judge = judge

    def evaluate_completeness(
        self,
        sample: ResponseEvaluationSample
    ) -> dict:
        """
        Evaluate whether the generated response covers all relevant information.

        Args:
            sample: ResponseEvaluationSample containing query, response, and retrieved chunks

        Returns:
            Dictionary with:
                - passed: True if response is complete, False otherwise
                - reason: Explanation from LLM judge
        """
        if not sample.generated_response:
            return {
                "passed": False,
                "reason": "No generated response provided — cannot evaluate completeness",
            }

        if not sample.retrieved_chunks:
            return {
                "passed": False,
                "reason": "No retrieved context provided — cannot evaluate completeness",
            }

        retrieved_context = "\n\n".join(sample.retrieved_chunks)

        prompt = self.EVAL_PROMPT_TEMPLATE.format(
            query=sample.query,
            retrieved_context=retrieved_context,
            generated_response=sample.generated_response
        )

        result = self.judge.judge(prompt)

        # Ensure JSON contains required keys
        if "passed" not in result or "reason" not in result:
            raise ValueError(
                f"Response completeness evaluation failed. LLM returned invalid JSON: {result}"
            )

        return {
            "passed": result["passed"],
            "reason": result["reason"]
        }