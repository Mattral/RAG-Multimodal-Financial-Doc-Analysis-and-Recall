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
