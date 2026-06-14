# ADR 004: Sandboxed Program-of-Thought for Financial Calculations

**Status:** Accepted  
**Date:** 2024-07  
**Deciders:** Core team

## Context

LLMs frequently hallucinate financial calculations — especially multi-step ones like CAGR, margin percentages, or YoY growth. We needed a way to produce verifiably correct numerical results without trusting the LLM to do arithmetic.

## Decision

Implement a Program-of-Thought (PoT) executor: the LLM generates Python code expressing the calculation, we extract it from the response, validate it against an AST allowlist, execute it in a restricted namespace with a hard timeout, and return the computed result.

Security constraints:
- AST node allowlist (only arithmetic, assignment, builtins like `round`, `abs`, `sum`)
- Blocked identifiers: `import`, `exec`, `eval`, `open`, `__builtins__`, `os`, `sys`, network libs
- Safe builtins dict — no access to `__builtins__` directly
- `asyncio.wait_for` timeout: 5 seconds
- No persistent side effects: fresh namespace per execution

## Rationale

- **PoT vs chain-of-thought arithmetic:** PoT delegates arithmetic to Python (exact) vs asking LLM to compute in text (approximate and error-prone). For financial figures, exactness matters.
- **AST whitelist over string filtering:** String filtering is bypassable. AST inspection at parse time catches obfuscated attacks before any execution.
- **`asyncio.to_thread` + `asyncio.wait_for`:** Prevents blocking the event loop and enforces hard timeout even for tight infinite loops (which the AST won't catch since `while`/`for` are in the allowlist for simple cases).

## Consequences

- **Positive:** CAGR, ROI, margin calculations are now exact. No arithmetic hallucinations for templated financial formulas.
- **Negative:** LLM must generate syntactically correct Python. If the LLM generates invalid code, PoT returns an error and falls back to direct answer. Adds ~200ms latency for the code extraction + execution round-trip.

## Security Note

The sandbox is defense-in-depth, not a security boundary. Do not expose PoT to untrusted user input without additional rate limiting and monitoring. The AST allowlist is regularly reviewed as Python adds new syntax.
