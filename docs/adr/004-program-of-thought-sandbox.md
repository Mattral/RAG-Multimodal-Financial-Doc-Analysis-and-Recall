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

## Update (2026-10): exec() replaced by a restricted interpreter

A review found that the allowlist above could be escaped. `ast.Attribute` was allowed, and
the blocked-name check only looked at bare names, so
`round.__self__.__import__('os').getpid()` passed validation and ran with the real `os`
module. The timeout could not stop runaway arithmetic either: `asyncio.wait_for` abandons the
worker thread but cannot kill it, so `9 ** 9 ** 9 ** 9` kept a core busy.

The executor no longer calls `exec()`. `pot_executor._Evaluator` walks the validated AST and
implements a small arithmetic language: numeric literals, variables, arithmetic, comparison
and boolean operators, `if`/`else`, assignment, list/tuple literals as arguments, and calls to
`abs round min max sum len float int pow`. Attribute access, subscripts, strings (other than
a leading docstring), loops, comprehensions, lambdas and function definitions are rejected by
the validator, so Python's object model is unreachable. Size limits bound the work: at most
2,000 AST nodes, exponents up to 1,000 and integers up to 4,096 bits. With no loops, every
program finishes in bounded time, and the timeout remains as a backstop.

Regression tests in `tests/unit/test_pot_executor.py` cover the traversal escape, other
dunder and attribute probes, and huge-exponent inputs.
