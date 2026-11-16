"""Cut a short annotated-free clip around each incident, straight from the source video."""

from pathlib import Path

import cv2


def export_clips(video_path, incidents, out_dir, before=3.0, after=3.0):
    """Write one MP4 per incident and return the paths. Frames are streamed, never buffered."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for n, incident in enumerate(incidents):
            start = max(0.0, incident["t"] - before)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
            path = Path(out_dir) / f"incident_{n:03d}_{incident['type']}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            for _ in range(int((incident["t"] + after - start) * fps)):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
            writer.release()
            paths.append(path)
    finally:
        cap.release()
    return paths
