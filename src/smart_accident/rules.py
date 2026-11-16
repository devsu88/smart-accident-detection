"""Incident rules over tracked vehicles. Pure numpy/cv2: no detector, no torch.

Two rules, both built to avoid firing on a single noisy signal:

- **collision**: two vehicles make *fresh* contact and, within a short time, *both* lose most of
  the speed they had before. Overlap alone is not enough (perspective occlusion overlaps boxes all
  the time) and neither is slowing down alone (traffic does that).
- **stopped vehicle**: a vehicle sits still inside the configured road polygon for a long time,
  unless several others are stopped too (a jam).

Speeds are measured in box-heights per second so they do not depend on distance from the camera,
and use video timestamps, so results do not depend on how fast the video is processed.
"""

import math
from collections import deque
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .config import RuleConfig

HISTORY_S = 6.0
FORGET_S = 2.0


@dataclass
class Incident:
    type: str  # "collision" | "stopped_vehicle"
    t: float  # video time in seconds
    ids: list[int]
    classes: list[int]
    score: float = 1.0
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class _Track:
    def __init__(self, track_id, class_id, t, box):
        self.id = track_id
        self.class_id = class_id
        self.first_t = t
        self.last_t = t
        self.box = box
        self.speed = 0.0
        self.samples = deque()  # (t, smoothed speed)
        self.stopped_since = None
        self.alerted = False

    def update(self, t, box, ema):
        dt = t - self.last_t
        if dt > 0:
            height = max(1.0, (box[3] - box[1] + self.box[3] - self.box[1]) / 2.0)
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            px, py = (self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0
            raw = math.hypot(cx - px, cy - py) / dt / height
            self.speed = raw if not self.samples else ema * raw + (1 - ema) * self.speed
            self.samples.append((t, self.speed))
            while self.samples and t - self.samples[0][0] > HISTORY_S:
                self.samples.popleft()
        self.last_t, self.box = t, box

    def speed_between(self, t0, t1):
        values = [s for ts, s in self.samples if t0 <= ts <= t1]
        return float(np.median(values)) if values else None

    def centre(self):
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)


def overlap_ratio(a, b):
    """Intersection area divided by the smaller box: 1.0 when one box sits inside the other."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (w * h) / smaller if smaller > 0 else 0.0


class IncidentDetector:
    """Feed it one frame of tracks at a time; it returns the incidents raised on that frame."""

    def __init__(self, rules: RuleConfig | None = None):
        self.rules = rules or RuleConfig()
        self._polygon = (
            np.array(self.rules.road_polygon, dtype=np.float32) if self.rules.road_polygon else None
        )
        self._tracks: dict[int, _Track] = {}
        self._contact_since: dict[tuple, float] = {}
        self._last_fired: dict[tuple, float] = {}

    def update(self, t, tracks):
        """``tracks``: rows of [x1, y1, x2, y2, track_id, class_id] for the frame at time ``t``."""
        r = self.rules
        seen = set()
        for row in tracks:
            x1, y1, x2, y2, track_id, class_id = row[:6]
            if int(class_id) not in r.vehicle_classes:
                continue
            track_id, box = int(track_id), (float(x1), float(y1), float(x2), float(y2))
            seen.add(track_id)
            track = self._tracks.get(track_id)
            if track is None:
                self._tracks[track_id] = _Track(track_id, int(class_id), t, box)
            else:
                track.update(t, box, r.speed_ema)

        for track_id in [i for i, tr in self._tracks.items() if t - tr.last_t > FORGET_S]:
            del self._tracks[track_id]
        mature = [tr for tr in self._tracks.values() if tr.id in seen and t - tr.first_t >= r.min_track_s]

        return self._collisions(t, mature) + self._stopped(t, mature)

    def _collisions(self, t, tracks):
        r = self.rules
        incidents, touching = [], set()
        for i, a in enumerate(tracks):
            for b in tracks[i + 1:]:
                if overlap_ratio(a.box, b.box) < r.overlap_min:
                    continue
                key = (min(a.id, b.id), max(a.id, b.id))
                touching.add(key)
                since = self._contact_since.setdefault(key, t)
                if t - since > r.react_s or t - self._last_fired.get(key, -math.inf) < r.cooldown_s:
                    continue
                drops = [self._speed_drop(t, tr) for tr in (a, b)]
                if all(d is not None and d >= r.drop_ratio for d in drops):
                    self._last_fired[key] = t
                    incidents.append(
                        Incident("collision", round(t, 3), [a.id, b.id], [a.class_id, b.class_id],
                                 score=round(min(drops), 3), details={"contact_since": round(since, 3)})
                    )
        for key in [k for k in self._contact_since if k not in touching]:
            del self._contact_since[key]
        return incidents

    def _speed_drop(self, t, track):
        """Share of its earlier speed the vehicle has lost, or None if it was not really moving."""
        r = self.rules
        before = track.speed_between(t - r.baseline_s[1], t - r.baseline_s[0])
        now = track.speed_between(t - r.now_s, t)
        if before is None or now is None or before < r.min_speed:
            return None
        return max(0.0, 1.0 - now / before)

    def _stopped(self, t, tracks):
        r = self.rules
        if self._polygon is None:
            return []
        for tr in tracks:
            now = tr.speed_between(t - 1.0, t)
            inside = cv2.pointPolygonTest(self._polygon, tr.centre(), False) >= 0
            if inside and now is not None and now <= r.stop_speed:
                tr.stopped_since = t if tr.stopped_since is None else tr.stopped_since
            else:
                tr.stopped_since, tr.alerted = None, False
        stopped = [tr for tr in tracks if tr.stopped_since is not None]
        if len(stopped) >= r.jam_count:
            return []
        incidents = []
        for tr in stopped:
            if not tr.alerted and t - tr.stopped_since >= r.stopped_s:
                tr.alerted = True
                incidents.append(
                    Incident("stopped_vehicle", round(t, 3), [tr.id], [tr.class_id],
                             details={"stopped_since": round(tr.stopped_since, 3)})
                )
        return incidents
