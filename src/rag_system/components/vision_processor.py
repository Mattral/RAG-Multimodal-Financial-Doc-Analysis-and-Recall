"""Vision processor compatibility shim — v2.0.

The canonical vision describer is now in `components/vision/__init__.py`.
This module re-exports for backward compatibility.
"""
from src.rag_system.components.vision import (
    OpenAIVisionDescriber,
    Qwen2VLDescriber,
    build_vision_describer,
    FINANCIAL_CHART_PROMPT,
)
from src.rag_system.components.vision.gemini_adapter import GeminiVisionDescriber

# Backward-compatible alias
VisionProcessor = OpenAIVisionDescriber

__all__ = [
    "VisionProcessor",
    "OpenAIVisionDescriber",
    "Qwen2VLDescriber",
    "GeminiVisionDescriber",
    "build_vision_describer",
    "FINANCIAL_CHART_PROMPT",
]
