"""Program-of-Thought (PoT) execution engine for financial calculations.

Extracts Python code blocks from LLM responses, validates them against
a financial operation whitelist, executes in a sandboxed environment with
timeout protection, and returns grounded numerical results.

Security model:
  - AST allowlist: only math operations, print, float, int, round, abs, sum, min, max, list
  - Blocked: import, open, exec, eval, __builtins__ access, network calls
  - Hard timeout: 5 seconds via asyncio.wait_for
  - Memory isolation: execution context is a fresh dict each call
"""
from __future__ import annotations

import ast
import asyncio
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)

# ── AST Allowlist ─────────────────────────────────────────────────────────────

_ALLOWED_AST_NODES = {
    # Expressions
    ast.Expression, ast.Expr, ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Compare, ast.IfExp, ast.Call, ast.Constant,
    ast.Name, ast.Load, ast.Attribute,
    # Arithmetic operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    # Comparisons
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    # Control flow (limited)
    ast.If, ast.Return,
    # Assignments
    ast.Assign, ast.AugAssign, ast.AnnAssign,
    # Module
    ast.Module,
    # Collections (literal, no comprehensions)
    ast.List, ast.Tuple, ast.Dict, ast.Store,
    # Function defs (simple single-function scripts only)
    ast.FunctionDef, ast.arguments, ast.arg,
}

_BLOCKED_NAMES = frozenset({
    "import", "__import__", "exec", "eval", "compile",
    "open", "input", "print", "__builtins__", "globals",
    "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "hasattr", "object", "type", "super", "classmethod",
    "staticmethod", "property", "subprocess", "os", "sys",
    "socket", "urllib", "requests", "httpx",
})

_SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "float": float, "int": int,
    "str": str, "bool": bool, "list": list, "tuple": tuple,
    "range": range, "enumerate": enumerate, "zip": zip,
    "sorted": sorted, "reversed": reversed,
    # Financial helpers
    "pow": pow,
}

# ── Financial templates ────────────────────────────────────────────────────────

FINANCIAL_TEMPLATES = {
    "percentage_change": textwrap.dedent("""
        v_old = {v_old}
        v_new = {v_new}
        result = ((v_new - v_old) / v_old) * 100
    """),
    "cagr": textwrap.dedent("""
        v_initial = {v_initial}
        v_final = {v_final}
        n_years = {n_years}
        result = ((v_final / v_initial) ** (1 / n_years) - 1) * 100
    """),
    "roi": textwrap.dedent("""
        gain = {gain}
        cost = {cost}
        result = ((gain - cost) / cost) * 100
    """),
    "gross_margin": textwrap.dedent("""
        revenue = {revenue}
        cogs = {cogs}
        result = ((revenue - cogs) / revenue) * 100
    """),
    "ebitda_margin": textwrap.dedent("""
        ebitda = {ebitda}
        revenue = {revenue}
        result = (ebitda / revenue) * 100
    """),
    "debt_to_equity": textwrap.dedent("""
        total_debt = {total_debt}
        total_equity = {total_equity}
        result = total_debt / total_equity
    """),
}


@dataclass
class PoTResult:
    """Result of a Program-of-Thought execution."""
    success: bool
    result: Optional[float] = None
    code: str = ""
    template_used: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    variables: Dict[str, Any] = field(default_factory=dict)

    def formatted(self, decimals: int = 2) -> str:
        """Return result as a formatted string."""
        if not self.success or self.result is None:
            return f"Error: {self.error}"
        return f"{self.result:.{decimals}f}"


class ASTSandboxValidator:
    """Validates Python AST against allowlist before execution."""

    def validate(self, code: str) -> Optional[str]:
        """Return error string if code is unsafe, else None."""
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            return f"SyntaxError: {exc}"

        for node in ast.walk(tree):
            node_type = type(node)
            if node_type not in _ALLOWED_AST_NODES:
                return f"Disallowed AST node: {node_type.__name__}"

            # Block dangerous name access
            if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                return f"Blocked identifier: {node.id}"

            # Block import statements (belt-and-suspenders)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "Import statements are not allowed"

        return None


class PoTExecutor:
    """Secure Program-of-Thought executor for financial calculations.

    Usage::

        executor = PoTExecutor()

        # Execute LLM-generated code
        result = await executor.execute_code(
            "v_old = 42.3\nv_new = 51.7\nresult = (v_new - v_old) / v_old * 100"
        )

        # Use a template directly
        result = await executor.execute_template(
            "cagr", v_initial=100, v_final=161, n_years=5
        )
    """

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self._timeout = timeout_seconds
        self._validator = ASTSandboxValidator()

    def _extract_code_block(self, text: str) -> Optional[str]:
        """Extract the first ```python ... ``` block from LLM text."""
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
            r"`{1,3}(.*?)`{1,3}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # No fences — treat whole text as code if it looks numeric
        if re.search(r"\bresult\s*=", text):
            return text.strip()
        return None

    def _run_in_sandbox(self, code: str) -> Dict[str, Any]:
        """Execute code in restricted namespace, return local variables."""
        namespace: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        exec(compile(code, "<pot>", "exec"), namespace)  # nosec B102
        return {k: v for k, v in namespace.items() if not k.startswith("__")}

    async def execute_code(self, code: str) -> PoTResult:
        """Validate and execute a Python code string."""
        import time

        error = self._validator.validate(code)
        if error:
            return PoTResult(success=False, code=code, error=f"Validation failed: {error}")

        start = time.perf_counter()
        try:
            variables = await asyncio.wait_for(
                asyncio.to_thread(self._run_in_sandbox, code),
                timeout=self._timeout,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            result_value = variables.get("result")
            if result_value is None:
                return PoTResult(
                    success=False, code=code, variables=variables,
                    error="Code executed but 'result' variable not set",
                    execution_time_ms=elapsed_ms,
                )
            return PoTResult(
                success=True,
                result=float(result_value),
                code=code,
                variables=variables,
                execution_time_ms=elapsed_ms,
            )
        except TimeoutError:
            return PoTResult(success=False, code=code, error=f"Execution timed out after {self._timeout}s")
        except Exception as exc:
            return PoTResult(success=False, code=code, error=f"RuntimeError: {exc}")

    async def execute_from_llm_response(self, llm_response: str) -> PoTResult:
        """Extract code block from LLM response and execute it."""
        code = self._extract_code_block(llm_response)
        if not code:
            return PoTResult(
                success=False,
                code="",
                error="No executable code block found in LLM response",
            )
        result = await self.execute_code(code)
        logger.info(
            "pot_execution_complete",
            success=result.success,
            result=result.result,
            elapsed_ms=round(result.execution_time_ms, 1),
        )
        return result

    async def execute_template(self, template_name: str, **kwargs: Any) -> PoTResult:
        """Execute a named financial calculation template.

        Args:
            template_name: One of percentage_change, cagr, roi, gross_margin, etc.
            **kwargs: Template variables (e.g. v_old=42.3, v_new=51.7).

        Returns:
            PoTResult with the computed result.
        """
        template = FINANCIAL_TEMPLATES.get(template_name)
        if not template:
            return PoTResult(
                success=False,
                error=f"Unknown template: {template_name}. "
                      f"Available: {list(FINANCIAL_TEMPLATES.keys())}",
            )
        try:
            code = template.format(**kwargs).strip()
        except KeyError as exc:
            return PoTResult(success=False, error=f"Missing template variable: {exc}")

        result = await self.execute_code(code)
        result.template_used = template_name
        return result
