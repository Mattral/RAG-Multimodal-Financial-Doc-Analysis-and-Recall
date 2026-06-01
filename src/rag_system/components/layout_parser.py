"""
Layout-Aware Document Parser for preserving spatial structure and hierarchies.

This module provides layout-aware parsing of documents, preserving spatial 
coordinates, element relationships, and hierarchical structures to maintain 
semantic context from tables, figures, captions, and surrounding narrative text.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import structlog

from ..config import get_config
from ..utils.logger import get_logger
from .pdf_parser import DocumentElement


# ============================================================================
# Data Models
# ============================================================================


class ElementType(Enum):
    """Enumeration of document element types."""

    TITLE = "title"
    HEADING = "heading"
    SUBHEADING = "subheading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    LIST = "list"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    CODE_BLOCK = "code_block"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box coordinates for an element on the page."""

    x0: float  # Left coordinate
    y0: float  # Top coordinate
    x1: float  # Right coordinate
    y1: float  # Bottom coordinate
    page: int = 0  # Page number (0-indexed)

    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this bbox overlaps with another."""
        return not (
            self.x1 < other.x0
            or self.x0 > other.x1
            or self.y1 < other.y0
            or self.y0 > other.y1
        )

    def contains(self, other: "BoundingBox") -> bool:
        """Check if this bbox contains another."""
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def is_above(self, other: "BoundingBox", threshold: float = 0.1) -> bool:
        """Check if this element is above another."""
        return self.y1 < (other.y0 - threshold)

    def is_below(self, other: "BoundingBox", threshold: float = 0.1) -> bool:
        """Check if this element is below another."""
        return self.y0 > (other.y1 + threshold)

    def is_left_of(self, other: "BoundingBox", threshold: float = 0.1) -> bool:
        """Check if this element is left of another."""
        return self.x1 < (other.x0 - threshold)

    def is_right_of(self, other: "BoundingBox", threshold: float = 0.1) -> bool:
        """Check if this element is right of another."""
        return self.x0 > (other.x1 + threshold)

    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page": self.page,
        }


@dataclass
class LayoutElement:
    """An element with layout information."""

    element_type: ElementType
    text: str
    bbox: BoundingBox
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_elements: List["LayoutElement"] = field(default_factory=list)
    confidence: float = 1.0

    def to_markdown(self, include_metadata: bool = False) -> str:
        """
        Convert element to markdown format.

        Args:
            include_metadata: Whether to include metadata as HTML comments

        Returns:
            Markdown-formatted string
        """
        prefix = ""

        if self.element_type == ElementType.TITLE:
            prefix = "# "
        elif self.element_type == ElementType.HEADING:
            prefix = "## "
        elif self.element_type == ElementType.SUBHEADING:
            prefix = "### "
        elif self.element_type == ElementType.TABLE:
            # Tables are already in markdown format from parser
            return self.text
        elif self.element_type == ElementType.FIGURE:
            return f"![Figure]({self.metadata.get('image_path', 'image')})"
        elif self.element_type == ElementType.LIST:
            # Lists should already be formatted
            return self.text
        elif self.element_type == ElementType.CAPTION:
            return f"*{self.text}*"
        elif self.element_type == ElementType.FOOTNOTE:
            return f"<sup>{self.text}</sup>"
        elif self.element_type == ElementType.CODE_BLOCK:
            lang = self.metadata.get("language", "")
            return f"```{lang}\n{self.text}\n```"

        result = prefix + self.text

        if include_metadata and self.metadata:
            meta_str = ", ".join(f"{k}: {v}" for k, v in self.metadata.items())
            result += f"\n<!-- Metadata: {meta_str} -->"

        return result

    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "element_type": self.element_type.value,
            "text": self.text,
            "bbox": self.bbox.model_dump(),
            "metadata": self.metadata,
            "confidence": self.confidence,
            "related_elements_count": len(self.related_elements),
        }


@dataclass
class LayoutGroup:
    """A group of related layout elements."""

    elements: List[LayoutElement] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    group_type: str = "mixed"  # title-content, figure-caption, table-description, etc.

    def add_element(self, element: LayoutElement) -> None:
        """Add an element to the group."""
        self.elements.append(element)

        if self.bbox is None:
            self.bbox = element.bbox
        else:
            # Expand bbox to contain new element
            self.bbox.x0 = min(self.bbox.x0, element.bbox.x0)
            self.bbox.y0 = min(self.bbox.y0, element.bbox.y0)
            self.bbox.x1 = max(self.bbox.x1, element.bbox.x1)
            self.bbox.y1 = max(self.bbox.y1, element.bbox.y1)

    def to_markdown(self) -> str:
        """Convert entire group to markdown."""
        lines = [element.to_markdown() for element in self.elements]
        return "\n\n".join(lines)


# ============================================================================
# Layout Parser
# ============================================================================


class LayoutParser:
    """
    Layout-aware document parser that preserves spatial structure.

    Implements semantic chunking by grouping related elements based on:
    - Spatial proximity and overlap
    - Visual hierarchy (headings, titles, captions)
    - Content type relationships (tables with captions, figures with descriptions)
    - Multi-page constraints
    """

    def __init__(self) -> None:
        """Initialize the layout parser."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.proximity_threshold = 20.0  # Points to consider elements related
        self.page_height = 792.0  # Standard letter page height in points
        self.page_width = 612.0  # Standard letter page width in points

        structlog.get_logger("layout_parser").debug(
            "layout_parser_initialized",
            proximity_threshold=self.proximity_threshold,
            page_height=self.page_height,
            page_width=self.page_width,
        )

    async def parse_elements(
        self,
        elements: List[DocumentElement],
    ) -> List[LayoutGroup]:
        """
        Parse document elements and group them by layout relationships.

        Args:
            elements: List of DocumentElement objects from pdf_parser

        Returns:
            List of LayoutGroup objects with semantic relationships
        """
        logger = structlog.get_logger("layout_parser")

        logger.debug("parse_elements_starting", element_count=len(elements))

        # Convert to LayoutElement with type inference
        layout_elements = await self._infer_element_types(elements)

        # Group related elements
        groups = await self._group_elements(layout_elements)

        logger.info(
            "parse_elements_completed",
            element_count=len(elements),
            layout_element_count=len(layout_elements),
            group_count=len(groups),
        )

        return groups

    async def _infer_element_types(self, elements: List[DocumentElement]) -> List[LayoutElement]:
        """
        Infer element types from text and metadata.

        Args:
            elements: List of DocumentElement objects

        Returns:
            List of LayoutElement objects with inferred types
        """
        layout_elements: List[LayoutElement] = []

        for elem in elements:
            element_type = self._classify_element(elem)
            bbox = self._extract_bbox(elem)

            layout_elem = LayoutElement(
                element_type=element_type,
                text=elem.text,
                bbox=bbox,
                metadata=elem.metadata or {},
            )

            layout_elements.append(layout_elem)

        return layout_elements

    def _classify_element(self, element: DocumentElement) -> ElementType:
        """
        Classify an element based on its properties.

        Args:
            element: DocumentElement to classify

        Returns:
            ElementType classification
        """
        elem_type = element.type.lower()

        # Direct classification by type
        if elem_type == "title":
            return ElementType.TITLE
        elif elem_type in ["heading", "h1", "h2", "h3"]:
            return ElementType.HEADING
        elif elem_type == "table":
            return ElementType.TABLE
        elif elem_type == "figure":
            return ElementType.FIGURE
        elif elem_type == "image":
            return ElementType.CHART
        elif elem_type == "list":
            return ElementType.LIST
        elif elem_type == "code":
            return ElementType.CODE_BLOCK

        # Heuristic-based classification for generic text
        text_lower = element.text.lower()

        # Check for common caption indicators
        if any(
            indicator in text_lower
            for indicator in [
                "figure ",
                "fig. ",
                "table ",
                "caption:",
                "source:",
            ]
        ):
            return ElementType.CAPTION

        # Check for footnote indicators
        if text_lower.startswith("*") or len(element.text) < 50:
            if any(marker in text_lower for marker in ["source", "note:", "footer"]):
                return ElementType.FOOTNOTE

        # Check text length and formatting for heading detection
        if len(element.text) < 100 and element.text.isupper():
            return ElementType.HEADING

        # Default classification
        return ElementType.PARAGRAPH

    def _extract_bbox(self, element: DocumentElement) -> BoundingBox:
        """
        Extract bounding box from element metadata.

        Args:
            element: DocumentElement

        Returns:
            BoundingBox with coordinates
        """
        metadata = element.metadata or {}

        x0 = float(metadata.get("x0", 0.0))
        y0 = float(metadata.get("y0", 0.0))
        x1 = float(metadata.get("x1", self.page_width))
        y1 = float(metadata.get("y1", self.page_height))
        page = int(metadata.get("page", 0))

        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page)

    async def _group_elements(self, elements: List[LayoutElement]) -> List[LayoutGroup]:
        """
        Group related elements based on layout proximity and type relationships.

        Args:
            elements: List of LayoutElement objects

        Returns:
            List of LayoutGroup objects
        """
        if not elements:
            return []

        # Sort by page and vertical position
        sorted_elements = sorted(
            elements, key=lambda e: (e.bbox.page, e.bbox.y0)
        )

        groups: List[LayoutGroup] = []
        current_group = LayoutGroup()

        for element in sorted_elements:
            if not current_group.elements:
                current_group.add_element(element)
            else:
                last_element = current_group.elements[-1]

                # Check if element should be added to current group
                if self._should_group(element, last_element, current_group):
                    current_group.add_element(element)
                else:
                    # Start new group
                    if current_group.elements:
                        current_group.group_type = self._determine_group_type(
                            current_group.elements
                        )
                        groups.append(current_group)

                    current_group = LayoutGroup()
                    current_group.add_element(element)

        # Add final group
        if current_group.elements:
            current_group.group_type = self._determine_group_type(current_group.elements)
            groups.append(current_group)

        return groups

    def _should_group(
        self,
        element: LayoutElement,
        last_element: LayoutElement,
        current_group: LayoutGroup,
    ) -> bool:
        """
        Determine if an element should be grouped with current group.

        Args:
            element: Element to consider
            last_element: Last element in current group
            current_group: Current LayoutGroup

        Returns:
            True if elements should be grouped together
        """
        # Don't group across pages
        if element.bbox.page != last_element.bbox.page:
            return False

        # Always group titles with following content
        if last_element.element_type == ElementType.TITLE:
            return True

        # Always group headings with following content
        if last_element.element_type in [
            ElementType.HEADING,
            ElementType.SUBHEADING,
        ]:
            return element.element_type not in [
                ElementType.TITLE,
                ElementType.HEADING,
            ]

        # Group captions with figures/tables
        if last_element.element_type in [ElementType.FIGURE, ElementType.TABLE]:
            return element.element_type == ElementType.CAPTION

        # Group based on vertical proximity
        vertical_gap = element.bbox.y0 - last_element.bbox.y1
        if 0 <= vertical_gap <= self.proximity_threshold:
            return True

        return False

    def _determine_group_type(self, elements: List[LayoutElement]) -> str:
        """
        Determine the group type based on element composition.

        Args:
            elements: List of LayoutElement objects in group

        Returns:
            Group type string
        """
        types = [e.element_type for e in elements]
        has_heading = any(t in [ElementType.HEADING, ElementType.TITLE] for t in types)
        has_table = ElementType.TABLE in types
        has_figure = ElementType.FIGURE in types
        has_caption = ElementType.CAPTION in types

        if has_figure and has_caption:
            return "figure_with_caption"
        elif has_table and has_caption:
            return "table_with_caption"
        elif has_heading:
            return "heading_with_content"
        elif has_table:
            return "table"
        elif has_figure:
            return "figure"
        else:
            return "mixed"

    async def to_markdown(
        self,
        groups: List[LayoutGroup],
        include_metadata: bool = False,
    ) -> str:
        """
        Convert layout groups to markdown format.

        Args:
            groups: List of LayoutGroup objects
            include_metadata: Whether to include metadata

        Returns:
            Markdown string
        """
        markdowns = [
            group.to_markdown() for group in groups
        ]

        return "\n\n---\n\n".join(markdowns)

    async def extract_table_content(
        self,
        table_element: LayoutElement,
    ) -> Dict[str, Any]:
        """
        Extract structured content from a table element.

        Args:
            table_element: LayoutElement of type TABLE

        Returns:
            Dictionary with table structure (rows, columns, data)
        """
        if table_element.element_type != ElementType.TABLE:
            return {}

        # Return markdown table content
        return {
            "type": "table",
            "content": table_element.text,
            "metadata": table_element.metadata,
        }

    async def extract_figure_content(
        self,
        figure_element: LayoutElement,
    ) -> Dict[str, Any]:
        """
        Extract content from a figure/chart element.

        Args:
            figure_element: LayoutElement of type FIGURE or CHART

        Returns:
            Dictionary with figure structure
        """
        if figure_element.element_type not in [ElementType.FIGURE, ElementType.CHART]:
            return {}

        return {
            "type": "figure",
            "image_path": figure_element.metadata.get("image_path"),
            "description": figure_element.text,
            "metadata": figure_element.metadata,
        }


# ============================================================================
# Public Interface
# ============================================================================


async def parse_document_layout(
    elements: List[DocumentElement],
) -> List[LayoutGroup]:
    """
    Convenience function to parse document layout using LayoutParser.

    Args:
        elements: List of DocumentElement objects

    Returns:
        List of LayoutGroup objects
    """
    parser = LayoutParser()
    return await parser.parse_elements(elements)


async def elements_to_markdown(
    elements: List[DocumentElement],
    include_metadata: bool = False,
) -> str:
    """
    Convert document elements to markdown preserving layout.

    Args:
        elements: List of DocumentElement objects
        include_metadata: Whether to include metadata

    Returns:
        Markdown string
    """
    parser = LayoutParser()
    groups = await parser.parse_elements(elements)
    return await parser.to_markdown(groups, include_metadata=include_metadata)
