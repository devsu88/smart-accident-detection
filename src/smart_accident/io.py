"""JSON Lines readers and writers. One line per frame for tracks, one per incident for events."""

import json
from pathlib import Path


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def frame_row(t, tracks):
    """A tracks line: ``{"t": 1.2, "tracks": [[x1, y1, x2, y2, id, class], ...]}``."""
    rows = [[round(float(v), 1) for v in row[:4]] + [int(row[4]), int(row[5])] for row in tracks]
    return {"t": round(float(t), 3), "tracks": rows}
