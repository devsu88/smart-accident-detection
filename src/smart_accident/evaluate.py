"""Score predicted incidents against hand-labelled incident intervals.

Labels file (JSON)::

    {"videos": [{"name": "cam1", "duration_s": 600, "incidents": [{"start": 120.0, "end": 128.0}]}]}

A video with an empty ``incidents`` list is normal footage: every event on it is a false alarm. An
event counts for an incident if it falls in ``[start - tolerance_before, end + tolerance_after]``.
"""

from dataclasses import dataclass


@dataclass
class Score:
    detected: int
    missed: int
    false_alarms: int
    precision: float
    recall: float
    f1: float
    false_alarms_per_hour: float
    mean_time_to_detect_s: float | None


def _ratio(num, den):
    return num / den if den else 0.0


def score_video(incidents, event_times, before=2.0, after=5.0):
    """Returns (detected, missed, false_alarms, delays) for one video."""
    matched, delays, false_alarms = set(), [], 0
    for t in event_times:
        hit = next((n for n, w in enumerate(incidents) if w["start"] - before <= t <= w["end"] + after), None)
        if hit is None:
            false_alarms += 1
        elif hit not in matched:
            matched.add(hit)
            delays.append(max(0.0, t - incidents[hit]["start"]))
    return len(matched), len(incidents) - len(matched), false_alarms, delays


def evaluate(labels, events_by_video, before=2.0, after=5.0):
    """``events_by_video``: name -> list of event times. Every labelled video must have an entry."""
    detected = missed = false_alarms = 0
    delays, hours = [], 0.0
    for video in labels["videos"]:
        name = video["name"]
        if name not in events_by_video:
            raise KeyError(f"No events for labelled video '{name}'")
        d, m, f, dl = score_video(video["incidents"], events_by_video[name], before, after)
        detected, missed, false_alarms = detected + d, missed + m, false_alarms + f
        delays += dl
        hours += video["duration_s"] / 3600.0
    precision = _ratio(detected, detected + false_alarms)
    recall = _ratio(detected, detected + missed)
    return Score(
        detected, missed, false_alarms, precision, recall,
        _ratio(2 * precision * recall, precision + recall),
        _ratio(false_alarms, hours),
        sum(delays) / len(delays) if delays else None,
    )
