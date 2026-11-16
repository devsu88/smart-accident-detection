import json

import pytest

from smart_accident.cli import main
from smart_accident.evaluate import evaluate, score_video

LABELS = {
    "videos": [
        {
            "name": "a",
            "duration_s": 1800,
            "incidents": [{"start": 100, "end": 110}, {"start": 400, "end": 405}],
        },
        {"name": "b", "duration_s": 1800, "incidents": []},
    ]
}


def test_matches_events_inside_the_tolerance_window():
    detected, missed, false_alarms, delays = score_video([{"start": 100, "end": 110}], [101.5, 102, 300])
    assert (detected, missed, false_alarms) == (1, 0, 1)  # a second hit on one incident is no false alarm
    assert delays == [1.5]


def test_scores_precision_recall_and_false_alarm_rate():
    score = evaluate(LABELS, {"a": [103.0, 700.0], "b": [50.0]})
    assert (score.detected, score.missed, score.false_alarms) == (1, 1, 2)
    assert score.precision == pytest.approx(1 / 3)
    assert score.recall == pytest.approx(0.5)
    assert score.false_alarms_per_hour == pytest.approx(2.0)
    assert score.mean_time_to_detect_s == pytest.approx(3.0)


def test_no_events_gives_zero_recall_and_no_division_error():
    score = evaluate(LABELS, {"a": [], "b": []})
    assert (score.precision, score.recall, score.f1) == (0.0, 0.0, 0.0)
    assert score.mean_time_to_detect_s is None


def test_unlabelled_video_is_an_error():
    with pytest.raises(KeyError):
        evaluate(LABELS, {"a": []})


def test_cli_eval_reads_one_events_file_per_video(tmp_path, capsys):
    (tmp_path / "labels.json").write_text(json.dumps(LABELS))
    (tmp_path / "ev").mkdir()
    (tmp_path / "ev" / "a.jsonl").write_text(json.dumps({"t": 103.0}) + "\n")
    (tmp_path / "ev" / "b.jsonl").write_text("")
    assert main(["eval", str(tmp_path / "labels.json"), str(tmp_path / "ev")]) == 0
    assert "detected 1  missed 1  false alarms 0" in capsys.readouterr().out


def test_cli_eval_reports_missing_events_file(tmp_path, capsys):
    (tmp_path / "labels.json").write_text(json.dumps(LABELS))
    assert main(["eval", str(tmp_path / "labels.json"), str(tmp_path)]) == 1
    assert "error" in capsys.readouterr().err
