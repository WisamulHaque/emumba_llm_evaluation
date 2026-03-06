"""
LLM Judge — shared provider-agnostic judge for all evaluators.

All evaluators that use LLM-as-judge import from here:

    from examples.common.llm_judge import LLMJudge

Supported providers (configured per evaluator via .env):
    anthropic  — claude-* models
    openai     — gpt-* models
    groq       — llama-*, mixtral-*, gemma-* models

"""

import json
import re
from typing import Dict, Any

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.common.utils import (
    anthropic_sdk, openai_sdk, Groq,
    _ANTHROPIC_AVAILABLE, _OPENAI_AVAILABLE, _GROQ_AVAILABLE,
)

class LLMJudge:
    """
    Provider-agnostic LLM judge for all evaluators.

    Handles provider selection, client initialization, API calls, and
    JSON response parsing. All evaluator-specific logic stays in the
    evaluator — the judge only handles the LLM interaction layer.

    Usage:
        judge = LLMJudge(provider="groq", model="llama-3.3-70b-versatile", api_key="...")
        result = judge.judge("Your evaluation prompt here...")
        # result: {"score": 1, "reason": "..."}
    """

    SUPPORTED_PROVIDERS = ("anthropic", "openai", "groq")

    def __init__(self, provider: str, model: str, api_key: str):
        """
        Initialize the LLM judge.

        Args:
            provider: LLM provider — "anthropic", "openai", or "groq"
            model:    Model name, e.g. "claude-3-haiku-20240307",
                      "gpt-4o-mini", "llama-3.3-70b-versatile"
            api_key:  API key for the provider
        """
        self.provider = provider.lower()
        self.model    = model
        self.api_key  = api_key
        self._client  = None

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

    def _get_client(self):
        """Get or create the provider client (lazy initialization)."""
        if self._client is None:
            if self.provider == "anthropic":
                self._client = anthropic_sdk.Anthropic(api_key=self.api_key)
            elif self.provider == "openai":
                self._client = openai_sdk.OpenAI(api_key=self.api_key)
            elif self.provider == "groq":
                self._client = Groq(api_key=self.api_key)
        return self._client

    def judge(self, prompt: str) -> Dict[str, Any]:
        """
        Send prompt to the configured LLM and return parsed JSON response.

        Args:
            prompt: Full evaluation prompt. Should instruct the model to
                    return only a JSON object with the expected keys.

        Returns:
            Parsed dict — format depends on the evaluator's prompt design.
            Common formats: {"score": 1, "reason": "..."} or
                            {"passed": true, "reason": "..."} or
                            {"claims": [...]}

        Raises:
            ValueError: If the response cannot be parsed as valid JSON,
                        or if required keys are missing.
        """
        raw = self._call_provider(prompt)
        return self._parse_response(raw)

    def _call_provider(self, prompt: str) -> str:
        """Call the configured provider and return raw text response."""
        client = self._get_client()

        if self.provider == "anthropic":
            message = client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        if self.provider in ("openai", "groq"):
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        Parse the LLM's JSON response.

        Strips markdown code fences (```json ... ```) if present — some
        models wrap JSON in fences despite being instructed not to.
        Uses regex to extract the JSON object in case the model adds
        preamble or trailing text.

        Supported response formats:
            {"score": 1, "reason": "..."}         — binary/scored evaluation
            {"passed": true, "reason": "..."}     — explicit pass/fail
            {"claims": [...]}                     — faithfulness evaluation
            {"results": [...]}                    — multi-result evaluation
        """
        # Strip markdown code fences
        clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        if not clean:
            raise ValueError(
                f"LLM returned empty response after stripping fences.\n"
                f"Raw: {raw}"
            )

        # Extract JSON object — handles preamble/trailing text from the model
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            raise ValueError(
                f"No JSON object found in LLM response.\n"
                f"Cleaned response: {clean}"
            )

        json_text = match.group(0)

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM response was not valid JSON.\n"
                f"Extracted text:\n{json_text}\n"
                f"Original raw:\n{raw}\n"
                f"Error: {e}"
            )

        # Accept any recognized evaluation response format
        if (
            ("score"  in parsed and "reason" in parsed) or
            ("passed" in parsed and "reason" in parsed) or
            ("claims"  in parsed) or
            ("results" in parsed)
        ):
            return parsed

        raise ValueError(
            f"LLM response missing required keys.\n"
            f"Expected one of: score+reason, passed+reason, claims, results.\n"
            f"Parsed: {parsed}"
        )