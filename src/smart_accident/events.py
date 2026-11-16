"""Tracks -> incidents. The whole analysis step, replayable from a saved tracks file."""

from .config import RuleConfig
from .rules import IncidentDetector


def detect_incidents(frames, rules: RuleConfig | None = None):
    """``frames``: dicts with ``t`` and ``tracks``. Returns the incidents as JSON-ready dicts."""
    detector = IncidentDetector(rules)
    incidents = []
    for frame in frames:
        incidents.extend(i.to_dict() for i in detector.update(frame["t"], frame["tracks"]))
    return incidents
