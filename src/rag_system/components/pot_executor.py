"""
Program-of-Thought (PoT) Executor for secure Python code execution.

This module provides a sandboxed execution environment for LLM-generated Python code,
specifically designed for financial calculations and data analysis. It includes:
- Secure code execution with timeout protection
- Markdown code block parsing
- Comprehensive error handling and logging
- Type-safe result reporting
- Rate limiting and retry policy support
"""

import asyncio
import re
import sys
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, Optional

import structlog

from ..config import get_config
from ..utils.logger import get_logger
from ..utils.rate_limiter import TokenBucket
from ..utils.retry_policy import RetryPolicy


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ExecutionResult:
    """
    Result of code execution containing status, output, and computed values.

    Attributes:
        success: Whether execution completed without errors
        output: Captured stdout from execution
        result_value: The computed result (if any)
        error_message: Error details (if execution failed)
        execution_time_ms: Time taken for execution in milliseconds
    """

    success: bool
    output: str
    result_value: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0

    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "success": self.success,
            "output": self.output,
            "result_value": str(self.result_value) if self.result_value is not None else None,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
        }


# ============================================================================
# Financial Formula Library
# ============================================================================


class FinancialFormulas:
    """Collection of verified financial calculation formulas."""

    @staticmethod
    def cagr(beginning_value: float, ending_value: float, num_years: float) -> float:
        """
        Calculate Compound Annual Growth Rate.

        Args:
            beginning_value: Starting value
            ending_value: Final value
            num_years: Number of years

        Returns:
            CAGR as a decimal (e.g., 0.15 for 15%)

        Raises:
            ValueError: If num_years <= 0 or beginning_value <= 0
        """
        if num_years <= 0:
            raise ValueError("num_years must be positive")
        if beginning_value <= 0:
            raise ValueError("beginning_value must be positive")
        return (ending_value / beginning_value) ** (1 / num_years) - 1

    @staticmethod
    def percentage_change(old_value: float, new_value: float) -> float:
        """
        Calculate percentage change from old to new value.

        Args:
            old_value: Starting value
            new_value: Ending value

        Returns:
            Percentage change as a decimal (e.g., 0.25 for 25%)

        Raises:
            ValueError: If old_value is zero
        """
        if old_value == 0:
            raise ValueError("old_value cannot be zero for percentage change")
        return (new_value - old_value) / old_value

    @staticmethod
    def roi(profit: float, investment: float) -> float:
        """
        Calculate Return on Investment.

        Args:
            profit: Net profit
            investment: Initial investment

        Returns:
            ROI as a decimal (e.g., 0.50 for 50%)

        Raises:
            ValueError: If investment <= 0
        """
        if investment <= 0:
            raise ValueError("investment must be positive")
        return profit / investment

    @staticmethod
    def compound_interest(principal: float, rate: float, periods: int) -> float:
        """
        Calculate compound interest.

        Args:
            principal: Initial amount
            rate: Interest rate per period (as decimal, e.g., 0.05 for 5%)
            periods: Number of compounding periods

        Returns:
            Final amount

        Raises:
            ValueError: If principal <= 0 or periods < 0
        """
        if principal <= 0:
            raise ValueError("principal must be positive")
        if periods < 0:
            raise ValueError("periods cannot be negative")
        return principal * ((1 + rate) ** periods)


# ============================================================================
# Code Parser
# ============================================================================


class CodeBlockParser:
    """Parse and extract Python code from markdown and text."""

    @staticmethod
    def extract_python_code(text: str) -> Optional[str]:
        """
        Extract Python code from markdown code block or raw text.

        Supports formats:
        - ```python
          code here
          ```
        - ```
          code here
          ```
        - Raw Python code

        Args:
            text: Input text potentially containing markdown code blocks

        Returns:
            Extracted Python code or None if no valid code block found
        """
        # Try to match markdown code block with python language hint
        python_match = re.search(
            r"```python\s*(.*?)```", text, re.DOTALL
        )
        if python_match:
            return python_match.group(1).strip()

        # Try to match generic markdown code block
        generic_match = re.search(
            r"```\s*(.*?)```", text, re.DOTALL
        )
        if generic_match:
            code = generic_match.group(1).strip()
            # Basic heuristic: if it looks like Python, return it
            if any(
                keyword in code
                for keyword in ["def ", "import ", "=", "return", "for ", "if "]
            ):
                return code

        # If no code block found, assume entire text is code
        if text.strip():
            return text.strip()

        return None


# ============================================================================
# PoT Executor
# ============================================================================


class PoTExecutor:
    """
    Secure Program-of-Thought (PoT) executor for Python code.

    Provides a sandboxed environment for executing LLM-generated Python code
    with comprehensive error handling, timeout protection, and logging.
    """

    def __init__(self) -> None:
        """Initialize the PoT executor with configuration and utilities."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        # Initialize token bucket for rate limiting: 100 requests per second with burst of 10
        self.rate_limiter = TokenBucket(capacity=10, refill_rate=100.0)
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.5,
            max_delay_seconds=10.0,
            backoff_factor=2.0,
        )

        # Create safe execution environment with whitelisted imports
        self._safe_globals: Dict[str, Any] = {
            "__builtins__": {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
                "range": range,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "print": print,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
            },
            "FinancialFormulas": FinancialFormulas,
        }

        structlog.get_logger("pot_executor").debug(
            "pot_executor_initialized",
            timeout_seconds=3.0,
            safe_builtins_count=len(self._safe_globals["__builtins__"]),
        )

    async def execute(
        self,
        code: str,
        timeout_seconds: float = 3.0,
        extract_from_markdown: bool = True,
    ) -> ExecutionResult:
        """
        Execute Python code in a secure, isolated environment.

        Args:
            code: Python code to execute
            timeout_seconds: Maximum execution time (default 3.0)
            extract_from_markdown: Whether to extract code from markdown blocks

        Returns:
            ExecutionResult with status, output, and computed values

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout
        """
        import time

        start_time = time.perf_counter()
        logger = structlog.get_logger("pot_executor")

        try:
            # Parse code if needed
            if extract_from_markdown:
                parsed_code = CodeBlockParser.extract_python_code(code)
                if not parsed_code:
                    return ExecutionResult(
                        success=False,
                        output="",
                        error_message="No valid Python code found in input",
                        execution_time_ms=0.0,
                    )
                code = parsed_code

            logger.debug(
                "execution_starting",
                code_length=len(code),
                timeout_seconds=timeout_seconds,
            )

            # Check rate limit
            await self.rate_limiter.acquire(tokens=1)

            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_in_sandbox(code),
                timeout=timeout_seconds,
            )

            execution_time_ms = (time.perf_counter() - start_time) * 1000
            result.execution_time_ms = execution_time_ms

            logger.info(
                "execution_succeeded",
                execution_time_ms=execution_time_ms,
                output_length=len(result.output),
            )

            return result

        except asyncio.TimeoutError as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "execution_timeout",
                timeout_seconds=timeout_seconds,
                execution_time_ms=execution_time_ms,
            )
            return ExecutionResult(
                success=False,
                output="",
                error_message=f"Execution timeout after {timeout_seconds} seconds",
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.warning(
                "execution_failed",
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
            )
            return ExecutionResult(
                success=False,
                output="",
                error_message=error_msg,
                execution_time_ms=execution_time_ms,
            )

    async def _execute_in_sandbox(self, code: str) -> ExecutionResult:
        """
        Execute code in sandbox with captured output.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with execution status and captured output
        """
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        local_scope: Dict[str, Any] = {}

        try:
            # Compile code
            compiled_code = compile(code, "<sandbox>", "exec")

            # Execute in isolated environment
            exec(compiled_code, self._safe_globals, local_scope)

            # Capture output
            output = sys.stdout.getvalue()

            # Extract result_value if a 'result' or 'output' variable exists
            result_value = local_scope.get("result") or local_scope.get("output")

            return ExecutionResult(
                success=True,
                output=output,
                result_value=result_value,
                error_message=None,
            )

        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                output=sys.stdout.getvalue(),
                error_message=f"SyntaxError: {e.msg} at line {e.lineno}",
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output=sys.stdout.getvalue(),
                error_message=f"{type(e).__name__}: {str(e)}",
            )

        finally:
            sys.stdout = old_stdout

    async def execute_financial_formula(
        self,
        formula_name: str,
        **kwargs: Any,
    ) -> ExecutionResult:
        """
        Execute a pre-verified financial formula with kwargs.

        Args:
            formula_name: Name of formula (e.g., 'cagr', 'roi', 'percentage_change')
            **kwargs: Arguments to pass to the formula

        Returns:
            ExecutionResult with computation result
        """
        logger = structlog.get_logger("pot_executor")

        try:
            formula = getattr(FinancialFormulas, formula_name, None)
            if not formula:
                return ExecutionResult(
                    success=False,
                    output="",
                    error_message=f"Unknown formula: {formula_name}",
                )

            logger.debug("formula_execution_starting", formula_name=formula_name, kwargs=kwargs)

            # Execute formula
            result_value = formula(**kwargs)

            logger.info(
                "formula_execution_succeeded",
                formula_name=formula_name,
                result=result_value,
            )

            return ExecutionResult(
                success=True,
                output=f"{formula_name}({kwargs}) = {result_value}",
                result_value=result_value,
            )

        except TypeError as e:
            logger.warning("formula_execution_type_error", formula_name=formula_name, error=str(e))
            return ExecutionResult(
                success=False,
                output="",
                error_message=f"Invalid arguments for {formula_name}: {str(e)}",
            )

        except Exception as e:
            logger.warning("formula_execution_failed", formula_name=formula_name, error=str(e))
            return ExecutionResult(
                success=False,
                output="",
                error_message=f"{type(e).__name__}: {str(e)}",
            )


# ============================================================================
# Public Interface
# ============================================================================


async def execute_pot_code(
    code: str,
    timeout_seconds: float = 3.0,
    extract_from_markdown: bool = True,
) -> ExecutionResult:
    """
    Convenience function to execute code using a singleton PoT executor.

    Args:
        code: Python code to execute
        timeout_seconds: Maximum execution time
        extract_from_markdown: Whether to extract code from markdown blocks

    Returns:
        ExecutionResult with execution status and output
    """
    executor = PoTExecutor()
    return await executor.execute(
        code=code,
        timeout_seconds=timeout_seconds,
        extract_from_markdown=extract_from_markdown,
    )


async def execute_financial_formula(
    formula_name: str,
    **kwargs: Any,
) -> ExecutionResult:
    """
    Convenience function to execute a financial formula.

    Args:
        formula_name: Name of formula to execute
        **kwargs: Formula arguments

    Returns:
        ExecutionResult with computation result
    """
    executor = PoTExecutor()
    return await executor.execute_financial_formula(formula_name=formula_name, **kwargs)
