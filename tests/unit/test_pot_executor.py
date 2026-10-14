"""Unit tests for the sandboxed Program-of-Thought executor."""

import pytest

from src.rag_system.components.pot_executor import (
    ASTSandboxValidator,
    PoTExecutor,
    PoTResult,
)

# ── AST Validator ─────────────────────────────────────────────────────────────


def test_validator_allows_basic_arithmetic():
    v = ASTSandboxValidator()
    assert v.validate("result = (42.3 * 1.5) - 10") is None


def test_validator_allows_assignment_chain():
    v = ASTSandboxValidator()
    code = "revenue = 23.35\ncogs = 15.2\nresult = (revenue - cogs) / revenue * 100"
    assert v.validate(code) is None


def test_validator_blocks_import():
    v = ASTSandboxValidator()
    err = v.validate("import os\nresult = os.getcwd()")
    assert err is not None
    assert "Import" in err or "Disallowed" in err


def test_validator_blocks_exec():
    v = ASTSandboxValidator()
    err = v.validate("exec('import os')\nresult = 1")
    assert err is not None


def test_validator_blocks_open():
    v = ASTSandboxValidator()
    err = v.validate("f = open('/etc/passwd')\nresult = f.read()")
    assert err is not None


def test_validator_blocks_builtins_access():
    v = ASTSandboxValidator()
    err = v.validate("result = __builtins__")
    assert err is not None


def test_validator_allows_financial_math():
    v = ASTSandboxValidator()
    code = "v_initial=100.0\nv_final=161.05\nn=5\nresult=((v_final/v_initial)**(1/n)-1)*100"
    assert v.validate(code) is None


def test_validator_blocks_syntax_error():
    v = ASTSandboxValidator()
    err = v.validate("result = (1 + 2")  # unmatched paren
    assert err is not None
    assert "SyntaxError" in err


# ── PoT Executor ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_simple_calculation():
    executor = PoTExecutor()
    result = await executor.execute_code("result = (100 - 42) / 100 * 100")
    assert result.success
    assert abs(result.result - 58.0) < 0.001


@pytest.mark.asyncio
async def test_execute_missing_result_variable():
    executor = PoTExecutor()
    result = await executor.execute_code("x = 42")
    assert not result.success
    assert "result" in result.error.lower()


@pytest.mark.asyncio
async def test_execute_validation_failure():
    executor = PoTExecutor()
    result = await executor.execute_code("import sys\nresult = sys.version")
    assert not result.success
    assert "Validation" in result.error


@pytest.mark.asyncio
async def test_execute_timeout():
    executor = PoTExecutor(timeout_seconds=0.01)
    # Infinite loop — should timeout
    result = await executor.execute_code("i=0\nwhile True:\n    i+=1\nresult=i")
    # May fail validation (while loop not in allowlist) or timeout
    assert not result.success


@pytest.mark.asyncio
async def test_execute_from_llm_response_fenced():
    executor = PoTExecutor()
    llm_resp = """
To calculate percentage change:
```python
v_old = 21.45
v_new = 23.35
result = (v_new - v_old) / v_old * 100
```
The answer is approximately 8.9%.
    """
    result = await executor.execute_from_llm_response(llm_resp)
    assert result.success
    assert abs(result.result - 8.857) < 0.01


@pytest.mark.asyncio
async def test_execute_from_llm_response_no_code():
    executor = PoTExecutor()
    result = await executor.execute_from_llm_response("The answer is approximately 8.9%.")
    assert not result.success
    assert "No executable code" in result.error


# ── Financial Templates ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_template_percentage_change():
    executor = PoTExecutor()
    result = await executor.execute_template("percentage_change", v_old=21.45, v_new=23.35)
    assert result.success
    assert abs(result.result - 8.857) < 0.01
    assert result.template_used == "percentage_change"


@pytest.mark.asyncio
async def test_template_cagr():
    executor = PoTExecutor()
    result = await executor.execute_template("cagr", v_initial=100.0, v_final=161.05, n_years=5)
    assert result.success
    assert abs(result.result - 10.0) < 0.1  # ~10% CAGR


@pytest.mark.asyncio
async def test_template_roi():
    executor = PoTExecutor()
    result = await executor.execute_template("roi", gain=1500.0, cost=1000.0)
    assert result.success
    assert abs(result.result - 50.0) < 0.001


@pytest.mark.asyncio
async def test_template_gross_margin():
    executor = PoTExecutor()
    result = await executor.execute_template("gross_margin", revenue=23.35, cogs=19.18)
    assert result.success
    assert abs(result.result - 17.87) < 0.1


@pytest.mark.asyncio
async def test_template_unknown():
    executor = PoTExecutor()
    result = await executor.execute_template("unknown_formula", x=1)
    assert not result.success
    assert "Unknown template" in result.error


@pytest.mark.asyncio
async def test_template_missing_variable():
    executor = PoTExecutor()
    result = await executor.execute_template("cagr", v_initial=100.0)  # missing v_final and n_years
    assert not result.success


def test_pot_result_formatted():
    r = PoTResult(success=True, result=17.856789, code="x=1\nresult=17.856789")
    assert r.formatted(2) == "17.86"
    assert r.formatted(0) == "18"


def test_pot_result_formatted_error():
    r = PoTResult(success=False, error="Division by zero")
    assert "Error" in r.formatted()


# ── Sandbox escape regressions ────────────────────────────────────────────────

# Each of these reached Python's object model under the old exec()-based sandbox,
# which allowed attribute access. `round.__self__` is the builtins module.
_ESCAPE_ATTEMPTS = [
    "result = round.__self__.__import__('os').getpid()",
    "result = abs.__self__.open('/etc/passwd')",
    "x = (1).__class__.__mro__[-1].__subclasses__()\nresult = 1",
    "result = (1.0).real",
    "f = lambda: 1\nresult = f()",
    "def f():\n    return 1\nresult = f()",
    "result = [x for x in (1, 2)][0]",
    "result = [1, 2][0]",
    "for i in (1, 2):\n    pass\nresult = 1",
    "result = __name__",
    "result = len('abc')",
    "result = getattr(1, 'real')",
    "result = type(1)",
    "result = sorted([1])",
]


@pytest.mark.parametrize("code", _ESCAPE_ATTEMPTS)
def test_validator_rejects_object_traversal(code):
    assert ASTSandboxValidator().validate(code) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("code", _ESCAPE_ATTEMPTS)
async def test_execute_rejects_object_traversal_before_running(code, monkeypatch):
    executor = PoTExecutor()

    def _must_not_run(self, code):  # pragma: no cover - failure path
        raise AssertionError("sandbox ran an invalid program")

    monkeypatch.setattr(PoTExecutor, "_run_in_sandbox", _must_not_run)
    result = await executor.execute_code(code)
    assert not result.success
    assert "Validation failed" in result.error


def test_run_in_sandbox_revalidates():
    with pytest.raises(ValueError):
        PoTExecutor()._run_in_sandbox("result = round.__self__")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        "result = 9 ** 9 ** 9 ** 9",
        "result = 10 ** 100000",
        "x = 2 ** 4000\nresult = x * x",
        "result = pow(7, 99999)",
    ],
)
async def test_execute_rejects_huge_numbers_quickly(code):
    import time

    start = time.perf_counter()
    result = await PoTExecutor(timeout_seconds=5).execute_code(code)
    assert not result.success
    assert "too large" in result.error
    assert time.perf_counter() - start < 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, expected",
    [
        ('"""Docstring."""\nx = 10\nresult = x * 2', 20.0),
        ("a = 3\na += 4\nresult = a", 7.0),
        ("x: float = 2.5\nresult = x * 2", 5.0),
        ("x = 5\nif x > 3:\n    result = 1\nelse:\n    result = 0", 1.0),
        ("result = round(sum([1.234, 2.345]), 2)", 3.58),
        ("result = max(1, 7, 3) - min([4, 2])", 5.0),
        ("result = abs(-3) if 1 < 2 <= 2 else 0", 3.0),
        ("result = pow(2, 10)", 1024.0),
        ("result = round(10 / 3, ndigits=1)", 3.3),
        ("result = len([1, 2, 3])", 3.0),
        ("result = 17 // 5 + 17 % 5", 5.0),
    ],
)
async def test_execute_supported_language(code, expected):
    result = await PoTExecutor().execute_code(code)
    assert result.success, result.error
    assert result.result == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, fragment",
    [
        ("result = 1 / 0", "RuntimeError"),
        ("result = y + 1", "not defined"),
        ("result = 10.0 ** 400", "RuntimeError"),
        ("result = 1e308 * 10", "finite"),
    ],
)
async def test_execute_runtime_errors(code, fragment):
    result = await PoTExecutor().execute_code(code)
    assert not result.success
    assert fragment in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, expected",
    [
        ("result = pow(2, 10, 7)", 2.0),
        ("result = pow(3, 4, mod=5)", 1.0),
        ("result = pow(2, -1, 7)", 4.0),
        ("result = pow(base=2, exp=10)", 1024.0),
        ("result = pow(1.05, exp=3)", 1.157625),
    ],
)
async def test_pow_call_forms(code, expected):
    result = await PoTExecutor().execute_code(code)
    assert result.success, result.error
    assert result.result == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, fragment",
    [
        # Keyword arguments used to skip the exponent pre-check and compute first.
        ("result = pow(base=3, exp=20000000)", "Exponent too large"),
        ("result = pow(3, exp=20000000)", "Exponent too large"),
        ("result = pow(2.0, 3, 5)", "integer arguments"),
        ("result = pow(2, 3, 5, 7)", "at most 3"),
        ("result = pow(2, base=3)", "Unsupported pow() argument"),
        ("result = pow(2)", "needs base and exp"),
    ],
)
async def test_pow_rejections(code, fragment):
    import time

    start = time.perf_counter()
    result = await PoTExecutor().execute_code(code)
    assert not result.success
    assert fragment in result.error
    assert time.perf_counter() - start < 1.0
