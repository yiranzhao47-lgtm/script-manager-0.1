"""Ingestion layer: ASR (stable-whisper) and OCR (PaddleOCR) runners."""

from src.ingestion.asr_runner import ASRRunner, ASRSegment, ASRWord
from src.ingestion.ocr_dedup import OCRDedup, OCRSegment
from src.ingestion.ocr_runner import OCRBlock, OCRFrame, OCRLine, OCRRunner

__all__ = [
    "ASRRunner",
    "ASRSegment",
    "ASRWord",
    "OCRRunner",
    "OCRFrame",
    "OCRLine",
    "OCRBlock",
    "OCRDedup",
    "OCRSegment",
]
