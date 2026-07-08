"""Metadata layer: MapReduce entity extraction and meta.json canonicalization."""

from src.metadata.map_phase import MapPhase
from src.metadata.reduce_phase import ReducePhase

__all__ = ["MapPhase", "ReducePhase"]
