import cv2
import numpy as np
import pytest

from smart_accident.cli import main
from smart_accident.clips import export_clips
from smart_accident.config import load_config, rule_config
from smart_accident.extract import extract_tracks
from smart_accident.io import read_jsonl, write_jsonl

from .scenes import frames, moving


class ScriptedDetector:
    """Returns one fixed car box per frame, moving 8 px right per frame."""

    def __init__(self):
        self.n = 0

    def detect(self, frame):
        x = 20 + 8 * self.n
        self.n += 1
        return np.array([[x, 40, x + 40, 70, 0.9, 3]], dtype=float)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "road.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 120))
    for _ in range(40):
        writer.write(np.zeros((120, 200, 3), dtype=np.uint8))
    writer.release()
    return path


def test_extract_tracks_yields_one_row_per_frame_with_video_time(video):
    rows = list(extract_tracks(video, {}, detector=ScriptedDetector()))
    assert len(rows) == 40
    assert rows[10]["t"] == pytest.approx(1.0)
    assert rows[-1]["tracks"] and rows[-1]["tracks"][0][5] == 3  # tracked car keeps its class


def test_extract_tracks_rejects_a_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(extract_tracks(tmp_path / "nope.mp4", {}, detector=ScriptedDetector()))


def test_events_command_end_to_end(tmp_path, capsys):
    scene = frames({1: moving(0, 100, 100, stop_at=6.3), 2: moving(1260, 100, -100, stop_at=6.3)}, 12)
    write_jsonl(tmp_path / "tracks.jsonl", scene)
    assert main(["events", str(tmp_path / "tracks.jsonl"), "-o", str(tmp_path / "events.jsonl")]) == 0
    events = read_jsonl(tmp_path / "events.jsonl")
    assert [e["type"] for e in events] == ["collision"]


def test_clips_are_cut_around_the_incident(video, tmp_path):
    paths = export_clips(video, [{"t": 2.0, "type": "collision"}], tmp_path / "clips", before=1.0, after=1.0)
    cap = cv2.VideoCapture(str(paths[0]))
    assert 15 <= cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 21  # ~2 s at 10 fps
    cap.release()


def test_default_config_is_valid_and_unknown_rule_keys_fail():
    assert rule_config(load_config()).drop_ratio == 0.6
    with pytest.raises(ValueError, match="Unknown rule"):
        rule_config({"rules": {"drop_ration": 0.5}})


def test_missing_input_is_reported_not_raised(tmp_path, capsys):
    assert main(["events", str(tmp_path / "missing.jsonl"), "-o", str(tmp_path / "o.jsonl")]) == 1
    assert "error" in capsys.readouterr().err
