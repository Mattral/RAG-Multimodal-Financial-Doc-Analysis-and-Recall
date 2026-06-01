"""Tests for pipeline components."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.rag_system.components.pdf_parser import PDFParser, DocumentElement
from src.rag_system.components.vision_processor import VisionProcessor
from src.rag_system.components.vector_indexer import VectorIndexer
from src.rag_system.components.pot_executor import (
    PoTExecutor,
    ExecutionResult,
    CodeBlockParser,
    FinancialFormulas,
    execute_pot_code,
    execute_financial_formula,
)
from src.rag_system.components.layout_parser import (
    LayoutParser,
    LayoutElement,
    LayoutGroup,
    BoundingBox,
    ElementType,
    parse_document_layout,
    elements_to_markdown,
)


class TestDocumentElement:
    """Test DocumentElement model."""

    def test_creation(self) -> None:
        """Test creating a document element."""
        element = DocumentElement(
            type="text",
            text="Sample text",
            source_document="test.pdf",
        )
        assert element.type == "text"
        assert element.text == "Sample text"
        assert element.source_document == "test.pdf"

    def test_with_metadata(self) -> None:
        """Test creating element with metadata."""
        element = DocumentElement(
            type="table",
            text="Table content",
            source_document="test.pdf",
            metadata={"rows": 5, "columns": 3},
        )
        assert element.metadata["rows"] == 5


@pytest.mark.asyncio
async def test_pdf_parser_initialization() -> None:
    """Test PDF parser initialization."""
    parser = PDFParser()
    assert parser.config is not None
    assert parser.retry_policy is not None


@pytest.mark.asyncio
async def test_vision_processor_initialization() -> None:
    """Test vision processor initialization."""
    processor = VisionProcessor()
    assert processor.config is not None
    assert processor.rate_limiter is not None


@pytest.mark.asyncio
async def test_vector_indexer_initialization() -> None:
    """Test vector indexer initialization."""
    indexer = VectorIndexer()
    assert indexer.config is not None
    assert indexer.vector_store is None  # Not initialized yet


# ============================================================================
# PoT Executor Tests
# ============================================================================


class TestCodeBlockParser:
    """Test CodeBlockParser functionality."""

    def test_extract_python_markdown(self) -> None:
        """Test extracting Python code from markdown block."""
        text = """Here's some code:
        ```python
        result = 5 + 5
        print(result)
        ```
        """
        code = CodeBlockParser.extract_python_code(text)
        assert code is not None
        assert "result = 5 + 5" in code
        assert "print(result)" in code

    def test_extract_generic_markdown(self) -> None:
        """Test extracting code from generic markdown block."""
        text = """Code example:
        ```
        x = 10
        y = 20
        sum_val = x + y
        ```
        """
        code = CodeBlockParser.extract_python_code(text)
        assert code is not None
        assert "x = 10" in code

    def test_extract_raw_code(self) -> None:
        """Test extracting raw Python code without markdown."""
        code_text = "x = 42\nprint(x)"
        code = CodeBlockParser.extract_python_code(code_text)
        assert code == code_text

    def test_extract_no_code(self) -> None:
        """Test handling of text with no code."""
        text = "Just plain text"
        code = CodeBlockParser.extract_python_code(text)
        assert code == "Just plain text"


class TestFinancialFormulas:
    """Test financial formula calculations."""

    def test_cagr(self) -> None:
        """Test CAGR calculation."""
        # Example: $100 to $200 over 2 years = ~41.42% CAGR
        result = FinancialFormulas.cagr(
            beginning_value=100,
            ending_value=200,
            num_years=2,
        )
        assert 0.40 < result < 0.45  # ~41.42%

    def test_cagr_invalid_years(self) -> None:
        """Test CAGR with invalid num_years."""
        with pytest.raises(ValueError, match="num_years must be positive"):
            FinancialFormulas.cagr(100, 200, 0)

    def test_cagr_invalid_beginning_value(self) -> None:
        """Test CAGR with invalid beginning_value."""
        with pytest.raises(ValueError, match="beginning_value must be positive"):
            FinancialFormulas.cagr(0, 200, 2)

    def test_percentage_change(self) -> None:
        """Test percentage change calculation."""
        # $100 to $150 = 50% increase
        result = FinancialFormulas.percentage_change(old_value=100, new_value=150)
        assert result == 0.5

    def test_percentage_change_negative(self) -> None:
        """Test negative percentage change."""
        # $200 to $100 = -50% change
        result = FinancialFormulas.percentage_change(old_value=200, new_value=100)
        assert result == -0.5

    def test_percentage_change_zero_old_value(self) -> None:
        """Test percentage change with zero old_value."""
        with pytest.raises(ValueError, match="old_value cannot be zero"):
            FinancialFormulas.percentage_change(old_value=0, new_value=100)

    def test_roi(self) -> None:
        """Test ROI calculation."""
        # $1000 profit on $10000 investment = 10% ROI
        result = FinancialFormulas.roi(profit=1000, investment=10000)
        assert result == 0.1

    def test_roi_invalid_investment(self) -> None:
        """Test ROI with invalid investment."""
        with pytest.raises(ValueError, match="investment must be positive"):
            FinancialFormulas.roi(profit=1000, investment=0)

    def test_compound_interest(self) -> None:
        """Test compound interest calculation."""
        # $1000 at 10% for 2 periods = $1210
        result = FinancialFormulas.compound_interest(
            principal=1000,
            rate=0.1,
            periods=2,
        )
        assert abs(result - 1210.0) < 0.01  # Allow for floating point precision

    def test_compound_interest_invalid_principal(self) -> None:
        """Test compound interest with invalid principal."""
        with pytest.raises(ValueError, match="principal must be positive"):
            FinancialFormulas.compound_interest(
                principal=0,
                rate=0.1,
                periods=2,
            )


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_success_result(self) -> None:
        """Test creating a successful result."""
        result = ExecutionResult(
            success=True,
            output="10",
            result_value=10,
        )
        assert result.success is True
        assert result.output == "10"
        assert result.result_value == 10
        assert result.error_message is None

    def test_failure_result(self) -> None:
        """Test creating a failure result."""
        result = ExecutionResult(
            success=False,
            output="",
            error_message="Division by zero",
        )
        assert result.success is False
        assert result.error_message == "Division by zero"

    def test_model_dump(self) -> None:
        """Test model_dump serialization."""
        result = ExecutionResult(
            success=True,
            output="result",
            result_value=42,
            execution_time_ms=123.45,
        )
        dumped = result.model_dump()
        assert dumped["success"] is True
        assert dumped["result_value"] == "42"
        assert dumped["execution_time_ms"] == 123.45


@pytest.mark.asyncio
async def test_pot_executor_initialization() -> None:
    """Test PoT executor initialization."""
    executor = PoTExecutor()
    assert executor.config is not None
    assert executor.rate_limiter is not None
    assert executor.logger is not None
    assert len(executor._safe_globals["__builtins__"]) > 0


@pytest.mark.asyncio
async def test_pot_execute_simple_arithmetic() -> None:
    """Test executing simple arithmetic code."""
    executor = PoTExecutor()
    code = """
x = 5
y = 10
result = x + y
"""
    result = await executor.execute(code, extract_from_markdown=False)
    assert result.success is True
    assert result.result_value == 15


@pytest.mark.asyncio
async def test_pot_execute_with_output() -> None:
    """Test executing code that produces output."""
    executor = PoTExecutor()
    code = """
for i in range(3):
    print(f"Value: {i}")
result = "done"
"""
    result = await executor.execute(code, extract_from_markdown=False)
    assert result.success is True
    assert "Value: 0" in result.output
    assert "Value: 1" in result.output
    assert "Value: 2" in result.output


@pytest.mark.asyncio
async def test_pot_execute_markdown_extraction() -> None:
    """Test executing code extracted from markdown."""
    executor = PoTExecutor()
    code = """
Please solve this:
```python
x = 100
y = 200
result = x + y
```
"""
    result = await executor.execute(code, extract_from_markdown=True)
    assert result.success is True
    assert result.result_value == 300


@pytest.mark.asyncio
async def test_pot_execute_syntax_error() -> None:
    """Test executing code with syntax error."""
    executor = PoTExecutor()
    code = "x = \nresult = x"  # Invalid syntax
    result = await executor.execute(code, extract_from_markdown=False)
    assert result.success is False
    assert "SyntaxError" in result.error_message


@pytest.mark.asyncio
async def test_pot_execute_runtime_error() -> None:
    """Test executing code with runtime error."""
    executor = PoTExecutor()
    code = "result = 1 / 0"  # ZeroDivisionError
    result = await executor.execute(code, extract_from_markdown=False)
    assert result.success is False
    assert "ZeroDivisionError" in result.error_message


@pytest.mark.asyncio
async def test_pot_execute_timeout() -> None:
    """Test executing code that times out."""
    executor = PoTExecutor()
    code = """
while True:
    pass
"""
    result = await executor.execute(code, timeout_seconds=0.1, extract_from_markdown=False)
    assert result.success is False
    assert "timeout" in result.error_message.lower()


@pytest.mark.asyncio
async def test_pot_execute_forbidden_code() -> None:
    """Test that dangerous functions are blocked."""
    executor = PoTExecutor()
    code = """
import os
result = os.system('echo danger')
"""
    result = await executor.execute(code, extract_from_markdown=False)
    assert result.success is False  # Should fail - os not available


@pytest.mark.asyncio
async def test_pot_execute_financial_formula_cagr() -> None:
    """Test executing CAGR formula."""
    executor = PoTExecutor()
    result = await executor.execute_financial_formula(
        formula_name="cagr",
        beginning_value=100,
        ending_value=200,
        num_years=2,
    )
    assert result.success is True
    assert result.result_value is not None
    assert 0.40 < result.result_value < 0.45


@pytest.mark.asyncio
async def test_pot_execute_financial_formula_roi() -> None:
    """Test executing ROI formula."""
    executor = PoTExecutor()
    result = await executor.execute_financial_formula(
        formula_name="roi",
        profit=1000,
        investment=10000,
    )
    assert result.success is True
    assert result.result_value == 0.1


@pytest.mark.asyncio
async def test_pot_execute_financial_formula_invalid() -> None:
    """Test executing non-existent formula."""
    executor = PoTExecutor()
    result = await executor.execute_financial_formula(
        formula_name="nonexistent",
    )
    assert result.success is False
    assert "Unknown formula" in result.error_message


@pytest.mark.asyncio
async def test_pot_execute_financial_formula_type_error() -> None:
    """Test formula with wrong argument types."""
    executor = PoTExecutor()
    result = await executor.execute_financial_formula(
        formula_name="cagr",
        beginning_value="not_a_number",
        ending_value=200,
        num_years=2,
    )
    assert result.success is False
    assert "Invalid arguments" in result.error_message or "TypeError" in result.error_message


@pytest.mark.asyncio
async def test_execute_pot_code_convenience() -> None:
    """Test convenience function for code execution."""
    code = """
a = 10
b = 20
result = a * b
"""
    result = await execute_pot_code(code, extract_from_markdown=False)
    assert result.success is True
    assert result.result_value == 200


@pytest.mark.asyncio
async def test_execute_financial_formula_convenience() -> None:
    """Test convenience function for formula execution."""
    result = await execute_financial_formula(
        formula_name="percentage_change",
        old_value=100,
        new_value=150,
    )
    assert result.success is True
    assert result.result_value == 0.5


# ============================================================================
# Layout Parser Tests
# ============================================================================


class TestBoundingBox:
    """Test BoundingBox functionality."""

    def test_area_calculation(self) -> None:
        """Test bbox area calculation."""
        bbox = BoundingBox(x0=0, y0=0, x1=10, y1=20)
        assert bbox.area() == 200.0

    def test_overlaps_true(self) -> None:
        """Test bbox overlap detection (overlapping)."""
        bbox1 = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        bbox2 = BoundingBox(x0=5, y0=5, x1=15, y1=15)
        assert bbox1.overlaps(bbox2)

    def test_overlaps_false(self) -> None:
        """Test bbox overlap detection (non-overlapping)."""
        bbox1 = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        bbox2 = BoundingBox(x0=20, y0=20, x1=30, y1=30)
        assert not bbox1.overlaps(bbox2)

    def test_contains_true(self) -> None:
        """Test bbox containment (contains)."""
        bbox1 = BoundingBox(x0=0, y0=0, x1=100, y1=100)
        bbox2 = BoundingBox(x0=10, y0=10, x1=20, y1=20)
        assert bbox1.contains(bbox2)

    def test_contains_false(self) -> None:
        """Test bbox containment (doesn't contain)."""
        bbox1 = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        bbox2 = BoundingBox(x0=5, y0=5, x1=20, y1=20)
        assert not bbox1.contains(bbox2)

    def test_is_above(self) -> None:
        """Test bbox above detection."""
        bbox1 = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        bbox2 = BoundingBox(x0=0, y0=50, x1=10, y1=60)
        assert bbox1.is_above(bbox2)

    def test_is_below(self) -> None:
        """Test bbox below detection."""
        bbox1 = BoundingBox(x0=0, y0=50, x1=10, y1=60)
        bbox2 = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        assert bbox1.is_below(bbox2)

    def test_is_left_of(self) -> None:
        """Test bbox left-of detection."""
        bbox1 = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        bbox2 = BoundingBox(x0=50, y0=0, x1=60, y1=10)
        assert bbox1.is_left_of(bbox2)

    def test_is_right_of(self) -> None:
        """Test bbox right-of detection."""
        bbox1 = BoundingBox(x0=50, y0=0, x1=60, y1=10)
        bbox2 = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        assert bbox1.is_right_of(bbox2)


class TestLayoutElement:
    """Test LayoutElement functionality."""

    def test_creation(self) -> None:
        """Test creating a layout element."""
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        elem = LayoutElement(
            element_type=ElementType.PARAGRAPH,
            text="Sample paragraph",
            bbox=bbox,
        )
        assert elem.element_type == ElementType.PARAGRAPH
        assert elem.text == "Sample paragraph"
        assert elem.confidence == 1.0

    def test_to_markdown_paragraph(self) -> None:
        """Test paragraph markdown conversion."""
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        elem = LayoutElement(
            element_type=ElementType.PARAGRAPH,
            text="This is a paragraph.",
            bbox=bbox,
        )
        md = elem.to_markdown()
        assert md == "This is a paragraph."

    def test_to_markdown_heading(self) -> None:
        """Test heading markdown conversion."""
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        elem = LayoutElement(
            element_type=ElementType.HEADING,
            text="Section Title",
            bbox=bbox,
        )
        md = elem.to_markdown()
        assert md == "## Section Title"

    def test_to_markdown_title(self) -> None:
        """Test title markdown conversion."""
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        elem = LayoutElement(
            element_type=ElementType.TITLE,
            text="Document Title",
            bbox=bbox,
        )
        md = elem.to_markdown()
        assert md == "# Document Title"

    def test_to_markdown_caption(self) -> None:
        """Test caption markdown conversion."""
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        elem = LayoutElement(
            element_type=ElementType.CAPTION,
            text="Figure caption",
            bbox=bbox,
        )
        md = elem.to_markdown()
        assert md == "*Figure caption*"

    def test_model_dump(self) -> None:
        """Test model_dump serialization."""
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        elem = LayoutElement(
            element_type=ElementType.PARAGRAPH,
            text="Test",
            bbox=bbox,
            metadata={"test_key": "test_value"},
        )
        dumped = elem.model_dump()
        assert dumped["element_type"] == "paragraph"
        assert dumped["text"] == "Test"
        assert dumped["metadata"] == {"test_key": "test_value"}


class TestLayoutGroup:
    """Test LayoutGroup functionality."""

    def test_creation(self) -> None:
        """Test creating a layout group."""
        group = LayoutGroup()
        assert group.elements == []
        assert group.bbox is None

    def test_add_element(self) -> None:
        """Test adding elements to group."""
        group = LayoutGroup()
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        elem = LayoutElement(
            element_type=ElementType.PARAGRAPH,
            text="Test",
            bbox=bbox,
        )
        group.add_element(elem)
        assert len(group.elements) == 1
        assert group.bbox is not None

    def test_add_multiple_elements_bbox_expansion(self) -> None:
        """Test bbox expansion when adding multiple elements."""
        group = LayoutGroup()
        bbox1 = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        bbox2 = BoundingBox(x0=50, y0=40, x1=150, y1=90)

        elem1 = LayoutElement(
            element_type=ElementType.PARAGRAPH,
            text="First",
            bbox=bbox1,
        )
        elem2 = LayoutElement(
            element_type=ElementType.PARAGRAPH,
            text="Second",
            bbox=bbox2,
        )

        group.add_element(elem1)
        group.add_element(elem2)

        assert len(group.elements) == 2
        assert group.bbox.x0 == 0
        assert group.bbox.y0 == 0
        assert group.bbox.x1 == 150
        assert group.bbox.y1 == 90

    def test_to_markdown(self) -> None:
        """Test group markdown conversion."""
        group = LayoutGroup()
        bbox1 = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        bbox2 = BoundingBox(x0=0, y0=60, x1=100, y1=110)

        elem1 = LayoutElement(
            element_type=ElementType.HEADING,
            text="Title",
            bbox=bbox1,
        )
        elem2 = LayoutElement(
            element_type=ElementType.PARAGRAPH,
            text="Content",
            bbox=bbox2,
        )

        group.add_element(elem1)
        group.add_element(elem2)

        md = group.to_markdown()
        assert "## Title" in md
        assert "Content" in md
        assert "\n\n" in md


@pytest.mark.asyncio
async def test_layout_parser_initialization() -> None:
    """Test LayoutParser initialization."""
    parser = LayoutParser()
    assert parser.config is not None
    assert parser.proximity_threshold == 20.0
    assert parser.page_height > 0
    assert parser.page_width > 0


@pytest.mark.asyncio
async def test_layout_parser_element_classification() -> None:
    """Test element type classification."""
    parser = LayoutParser()
    
    # Test table classification
    table_elem = DocumentElement(
        type="table",
        text="| Col1 | Col2 |\n|------|------|",
        source_document="test.pdf",
    )
    elem_type = parser._classify_element(table_elem)
    assert elem_type == ElementType.TABLE

    # Test figure classification
    figure_elem = DocumentElement(
        type="image",
        text="Figure content",
        source_document="test.pdf",
    )
    elem_type = parser._classify_element(figure_elem)
    assert elem_type == ElementType.CHART


@pytest.mark.asyncio
async def test_layout_parser_parse_elements() -> None:
    """Test parsing document elements."""
    parser = LayoutParser()
    
    # Create sample elements
    elements = [
        DocumentElement(
            type="title",
            text="Document Title",
            source_document="test.pdf",
            metadata={"x0": 0, "y0": 0, "x1": 100, "y1": 50},
        ),
        DocumentElement(
            type="paragraph",
            text="First paragraph.",
            source_document="test.pdf",
            metadata={"x0": 0, "y0": 60, "x1": 100, "y1": 100},
        ),
    ]
    
    groups = await parser.parse_elements(elements)
    assert len(groups) > 0
    assert all(isinstance(g, LayoutGroup) for g in groups)


@pytest.mark.asyncio
async def test_layout_parser_grouping_heading_with_content() -> None:
    """Test grouping heading with following content."""
    parser = LayoutParser()
    
    elements = [
        DocumentElement(
            type="heading",
            text="Section Heading",
            source_document="test.pdf",
            metadata={"x0": 0, "y0": 0, "x1": 100, "y1": 30},
        ),
        DocumentElement(
            type="paragraph",
            text="Content under heading",
            source_document="test.pdf",
            metadata={"x0": 0, "y0": 40, "x1": 100, "y1": 80},
        ),
    ]
    
    groups = await parser.parse_elements(elements)
    # Should group heading with content
    assert any(g.group_type == "heading_with_content" for g in groups)


@pytest.mark.asyncio
async def test_parse_document_layout_convenience() -> None:
    """Test convenience function for layout parsing."""
    elements = [
        DocumentElement(
            type="title",
            text="Title",
            source_document="test.pdf",
            metadata={"x0": 0, "y0": 0, "x1": 100, "y1": 50},
        ),
    ]
    
    groups = await parse_document_layout(elements)
    assert len(groups) > 0
    assert isinstance(groups[0], LayoutGroup)


@pytest.mark.asyncio
async def test_elements_to_markdown_convenience() -> None:
    """Test convenience function to convert elements to markdown."""
    elements = [
        DocumentElement(
            type="title",
            text="Title",
            source_document="test.pdf",
            metadata={"x0": 0, "y0": 0, "x1": 100, "y1": 50},
        ),
        DocumentElement(
            type="paragraph",
            text="Content",
            source_document="test.pdf",
            metadata={"x0": 0, "y0": 60, "x1": 100, "y1": 100},
        ),
    ]
    
    markdown = await elements_to_markdown(elements)
    assert "# Title" in markdown or "Title" in markdown
    assert "Content" in markdown
