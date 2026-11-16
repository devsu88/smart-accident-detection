"""Video -> tracks, using the detector and SORT tracker from smart-surveillance.

This is the only module that needs a detector backend (and so torch). Everything after it works on
the saved tracks file.
"""

import cv2
import numpy as np
from smart_surveillance.detection import build_detector
from smart_surveillance.tracking import SortTracker

from .io import frame_row


def extract_tracks(video_path, config, detector=None, tracker=None, stride=1):
    """Yield one tracks row per processed frame (every ``stride``-th).

    Pass ``detector``/``tracker`` to inject your own (tests do).
    """
    detector = detector or build_detector(config)
    t = config.get("tracking", {})
    tracker = tracker or SortTracker(max_age=t.get("max_age", 5), min_hits=t.get("min_hits", 3),
                                     iou_threshold=0.3)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            detections = detector.detect(frame)
            tracks = tracker.update(detections if len(detections) else np.empty((0, 6)))
            yield frame_row(index / fps, tracks)
            index += 1
    finally:
        cap.release()
