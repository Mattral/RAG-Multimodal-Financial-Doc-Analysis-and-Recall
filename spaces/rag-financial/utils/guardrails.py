"""utils/guardrails.py — Numeric grounding, PII detection, and injection protection.

Mirrors src/rag_system/components/guardrails/ logic in standalone form.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GuardrailResult:
    overall_passed: bool
    numeric_grounding_passed: bool
    pii_detected: bool
    injection_detected: bool
    ungrounded_numbers: List[str]
    pii_entities: List[str]
    redacted_query: Optional[str]
    details: List[str]
    warnings: List[str]


_NUMBER_RE = re.compile(
    r"""
    (?:
        \$\s*[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion|B|M|T|K))?
        | [\d,]+(?:\.\d+)?\s*(?:billion|million|trillion|B|M|T|K)\b
        | [\d,]+(?:\.\d+)?%
        | \d+\.\d+(?:\s*x\b)?
        | (?<!\w)\d{4}(?!\w)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _normalise_number(raw: str) -> str:
    return re.sub(r"\s+", "", raw.lower().replace(",", ""))


def check_numeric_grounding(answer: str, context_texts: List[str]) -> Tuple[bool, List[str], List[str]]:
    details = []
    answer_numbers = _NUMBER_RE.findall(answer)
    if not answer_numbers:
        details.append("No numeric claims found in answer - grounding check not applicable")
        return True, [], details

    context_combined = " ".join(context_texts)
    context_numbers_raw = _NUMBER_RE.findall(context_combined)
    context_numbers_norm = {_normalise_number(n) for n in context_numbers_raw}

    ungrounded = []
    for num in answer_numbers:
        norm = _normalise_number(num)
        found = norm in context_numbers_norm or any(
            norm in cn or cn in norm for cn in context_numbers_norm
        )
        if not found:
            ungrounded.append(num)

    if not ungrounded:
        details.append(f"All {len(answer_numbers)} numeric values grounded in source context")
    else:
        details.append(
            f"{len(ungrounded)} of {len(answer_numbers)} numbers not found in context: "
            + ", ".join(f"`{n}`" for n in ungrounded[:5])
        )

    return len(ungrounded) == 0, ungrounded, details


_PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    (r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]{1,3}\b", "IBAN"),
    (r"\b[A-Z]\d{9}\b", "CUSIP"),
    (r"\b[A-Z]{2}[A-Z0-9]{10}\b", "ISIN"),
    (r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b", "Card number"),
    (r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "Email", re.IGNORECASE),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "Phone"),
    (r"\bACC\d{8,12}\b", "Account number"),
]


def check_pii(text: str) -> Tuple[bool, List[str], str]:
    found_entities = []
    redacted = text
    for pattern_def in _PII_PATTERNS:
        pattern, label = pattern_def[0], pattern_def[1]
        flags = pattern_def[2] if len(pattern_def) > 2 else 0
        matches = re.findall(pattern, text, flags)
        if matches:
            found_entities.extend([f"{label}: {m}" for m in matches[:3]])
            redacted = re.sub(pattern, f"[{label.upper()}_REDACTED]", redacted, flags=flags)
    return bool(found_entities), found_entities, redacted


_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(previous|all|prior)\s+instructions?|"
    r"disregard\s+(your|all|the)\s+(instructions?|guidelines?|system\s+prompt)|"
    r"jailbreak|act\s+as\s+(an?\s+)?unrestricted|"
    r"you\s+are\s+now\s+(?:a\s+)?(?:dan|evil|uncensored)|"
    r"bypass\s+(safety|guardrails?|filters?)|"
    r"pretend\s+you\s+(have\s+no|don't\s+have)\s+(restrictions?|limits?)|"
    r"developer\s+mode\s+enabled|"
    r"new\s+system\s+prompt|override\s+instructions?)",
    re.IGNORECASE,
)


def check_injection(query: str) -> Tuple[bool, Optional[str]]:
    match = _INJECTION_PATTERNS.search(query)
    if match:
        return True, match.group(0)
    return False, None


def run_guardrails(query: str, answer: str, context_texts: List[str]) -> GuardrailResult:
    details = ["**Guardrail Check Results**", ""]
    warnings = []

    is_injection, injection_match = check_injection(query)
    if is_injection:
        details.append(f"Injection blocked: Pattern `{injection_match}` detected in query")
        return GuardrailResult(
            overall_passed=False, numeric_grounding_passed=False,
            pii_detected=False, injection_detected=True,
            ungrounded_numbers=[], pii_entities=[],
            redacted_query=None, details=details, warnings=warnings,
        )
    details.append("Injection check: No adversarial patterns detected")

    pii_found, pii_entities, redacted_query = check_pii(query)
    if pii_found:
        details.append(
            f"PII detected & redacted: {', '.join(pii_entities[:3])}"
            + (" (+more)" if len(pii_entities) > 3 else "")
        )
        warnings.append("Query contained PII - redacted before processing")
    else:
        details.append("PII check: No sensitive identifiers detected in query")

    numeric_passed, ungrounded, numeric_details = check_numeric_grounding(answer, context_texts)
    details.extend(numeric_details)
    if not numeric_passed:
        warnings.append(
            f"{len(ungrounded)} numeric value(s) in the answer could not be verified "
            f"against the retrieved source context. This may indicate the model "
            f"extrapolated beyond the document."
        )

    overall = numeric_passed and not is_injection
    if overall:
        details.append("\nOverall: All guardrails passed")
    else:
        details.append("\nOverall: One or more guardrails flagged - review warnings above")

    return GuardrailResult(
        overall_passed=overall,
        numeric_grounding_passed=numeric_passed,
        pii_detected=pii_found,
        injection_detected=False,
        ungrounded_numbers=ungrounded,
        pii_entities=pii_entities,
        redacted_query=redacted_query if pii_found else None,
        details=details,
        warnings=warnings,
    )
