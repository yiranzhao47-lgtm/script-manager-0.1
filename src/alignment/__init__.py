"""Alignment layer: time-axis overlap algorithm and OCR-timeline pass-through."""

from src.alignment.overlap_aligner import AlignedSegment, OCRCandidate, OverlapAligner

__all__ = ["OverlapAligner", "AlignedSegment", "OCRCandidate"]
