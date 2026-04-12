"""
llm_api/ — VLM-Based Diagnostic Agent
======================================
Disease classification using Large Multimodal Models.
"""

from .service import classify_disease, classify_tray_disease
from .config import is_available, _configure_gemini

__all__ = ["classify_disease", "classify_tray_disease", "is_available", "_configure_gemini"]
