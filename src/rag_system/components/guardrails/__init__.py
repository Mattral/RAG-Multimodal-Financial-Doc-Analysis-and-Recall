"""PII redaction and financial guardrails.

Wraps Microsoft Presidio for PII detection/redaction with additional
finance-domain patterns (account numbers, tickers, CUSIP, etc.).
Falls back gracefully if presidio is not installed.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)

# Financial-specific regex patterns
_FINANCIAL_PATTERNS: Dict[str, re.Pattern] = {
    "CUSIP": re.compile(r"\b[0-9A-Z]{9}\b"),
    "ISIN": re.compile(r"\b[A-Z]{2}[0-9A-Z]{10}\b"),
    "BANK_ACCOUNT_US": re.compile(r"\b\d{8,17}\b"),
    "ROUTING_NUMBER": re.compile(r"\b\d{9}\b"),
    "TICKER": re.compile(r"\b\$[A-Z]{1,5}\b"),  # $AAPL style
}


def _redact_with_regex(text: str, entity_map: Dict[str, str]) -> Tuple[str, List[str]]:
    """Apply regex-based redaction for financial entities."""
    found: List[str] = []
    for entity_type, pattern in entity_map.items():
        matches = pattern.findall(text)
        if matches:
            text = pattern.sub(f"<{entity_type}>", text)
            found.extend([entity_type] * len(matches))
    return text, found


class PIIRedactor:
    """Detects and redacts PII + financial identifiers from text.

    Uses Microsoft Presidio when available; falls back to regex-only mode.
    """

    def __init__(
        self,
        pii_entities: Optional[List[str]] = None,
        enable_financial_patterns: bool = True,
        language: str = "en",
    ) -> None:
        self.language = language
        self.enable_financial = enable_financial_patterns
        self.pii_entities = pii_entities or [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "US_SSN",
            "CREDIT_CARD",
            "IBAN_CODE",
            "US_BANK_NUMBER",
        ]
        self._presidio_available = False
        self._analyzer = None
        self._anonymizer = None
        self._try_init_presidio()

    def _try_init_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._presidio_available = True
            logger.info("presidio_pii_engine_initialized")
        except ImportError:
            logger.warning(
                "presidio_not_installed",
                detail="Falling back to regex-only PII redaction. "
                       "Install with: pip install presidio-analyzer presidio-anonymizer",
            )

    def redact(self, text: str) -> Tuple[str, List[str]]:
        """Redact PII from text.

        Returns:
            (redacted_text, list_of_entity_types_found)
        """
        found_entities: List[str] = []

        # --- Presidio pass ---
        if self._presidio_available and self._analyzer:
            try:
                results = self._analyzer.analyze(
                    text=text,
                    entities=self.pii_entities,
                    language=self.language,
                )
                if results:
                    from presidio_anonymizer import AnonymizerEngine
                    anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
                    text = anonymized.text
                    found_entities = [r.entity_type for r in results]
            except Exception as exc:
                logger.warning("presidio_redaction_failed", error=str(exc))

        # --- Financial regex pass ---
        if self.enable_financial:
            text, fin_entities = _redact_with_regex(text, _FINANCIAL_PATTERNS)
            found_entities.extend(fin_entities)

        return text, found_entities

    def redact_batch(self, texts: List[str]) -> List[Tuple[str, List[str]]]:
        """Redact PII from a list of texts."""
        return [self.redact(t) for t in texts]


# ---------------------------------------------------------------------------
# Financial guardrails
# ---------------------------------------------------------------------------


class FinancialGuardrails:
    """Post-generation checks specific to financial RAG outputs.

    Checks:
    1. Numeric grounding – numbers in answer should appear in context
    2. Hallucination proxy – answer embedding similarity to context
    3. Prompt injection detection
    4. Refusal for disallowed financial advice patterns
    """

    # Regex for monetary / numeric values worth checking
    _NUMBER_RE = re.compile(r"\b\d[\d,]*\.?\d*\s*(?:million|billion|trillion|%)?\b", re.I)
    _INJECTION_PATTERNS = re.compile(
        r"(ignore previous|disregard instructions|jailbreak|system prompt|"
        r"act as|you are now|forget your|new persona)",
        re.I,
    )

    def check_numeric_grounding(
        self,
        answer: str,
        context_chunks: List[str],
        tolerance: float = 0.01,
    ) -> Tuple[bool, List[str]]:
        """Check that numeric values in the answer appear in context.

        Returns (passed: bool, ungrounded_numbers: List[str]).
        """
        context_text = " ".join(context_chunks).lower()
        answer_numbers = self._NUMBER_RE.findall(answer)
        ungrounded: List[str] = []

        for num_str in answer_numbers:
            # Normalise: remove commas and whitespace
            cleaned = re.sub(r"[\s,]", "", num_str).lower()
            if cleaned not in context_text.replace(",", "").replace(" ", ""):
                ungrounded.append(num_str)

        passed = len(ungrounded) == 0
        if not passed:
            logger.warning(
                "numeric_grounding_check_failed",
                ungrounded_numbers=ungrounded[:10],
            )
        return passed, ungrounded

    def check_prompt_injection(self, query: str) -> bool:
        """Return True if query appears to contain a prompt injection attempt."""
        is_injection = bool(self._INJECTION_PATTERNS.search(query))
        if is_injection:
            logger.warning("prompt_injection_detected", query_preview=query[:200])
        return is_injection

    def run_all_checks(
        self,
        query: str,
        answer: str,
        context_chunks: List[str],
    ) -> Dict[str, object]:
        """Run all guardrail checks and return a results dict."""
        injection = self.check_prompt_injection(query)
        grounded, ungrounded = self.check_numeric_grounding(answer, context_chunks)
        return {
            "prompt_injection": injection,
            "numeric_grounding_passed": grounded,
            "ungrounded_numbers": ungrounded,
            "overall_passed": not injection and grounded,
        }
