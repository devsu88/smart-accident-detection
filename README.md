# smart-accident-detection

Flags possible road incidents on fixed-camera video by reasoning over **tracked vehicles**, and scores
itself against hand-labelled footage. It builds on the detector and SORT tracker of
[smart-surveillance](https://github.com/devsu88/smart-surveillance), used as a package dependency.

> **Status: the rules and the evaluation harness are tested on synthetic trajectories. They have not
> been evaluated on real footage yet**, so no real-world precision or recall is claimed. See
> [Evaluating on real video](#evaluating-on-real-video).

## How it works

```text
video ──tracks──▶ tracks.jsonl ──events──▶ events.jsonl ──eval──▶ precision / recall
 (detector + SORT,                (pure numpy rules,              (against labelled
  needs torch)                     no GPU)                         incident intervals)
                                       └──clips──▶ short MP4 around each event
```

The pipeline is split on purpose. Only `tracks` needs a detector (and torch). The rules and the
evaluation run on the saved tracks, so you can tune thresholds and re-score in seconds.

### Rules

| Incident | Fires when |
|---|---|
| `collision` | Two vehicles make **fresh contact** (boxes overlap) and, within `react_s`, **both** lose at least `drop_ratio` of the speed they had just before. One event per pair per cooldown. |
| `stopped_vehicle` | A vehicle stays still inside the configured `road_polygon` for `stopped_s`, unless `jam_count` or more vehicles are stopped (a jam). Off unless a polygon is set. |

Design choices that keep false alarms down:

- **No single signal fires.** Perspective occlusion overlaps boxes all the time, and traffic slows
  down all the time. Only overlap *together with* a sharp drop in both speeds counts.
- **Speeds are in box-heights per second**, so they do not depend on distance from the camera, and
  they use video timestamps, so results do not depend on processing speed. Per-frame speed is
  smoothed to absorb box jitter, and young tracks are ignored.
- **Vehicles that were not really moving cannot "slow down"**: a jam overlap does not trigger.

## Quick start

```bash
uv sync                      # rules, evaluation, clips: no torch needed
uv run pytest -q
```

To run on video you need a detector backend (this is the heavy part):

```bash
uv sync --extra rtdetr       # or --extra torchvision
uv run smart-accident tracks road.mp4 -o outputs/road.tracks.jsonl
uv run smart-accident events outputs/road.tracks.jsonl -o outputs/road.events.jsonl
uv run smart-accident clips road.mp4 outputs/road.events.jsonl -o outputs/clips
```

Settings (detector, thresholds, `road_polygon` in pixels) live in
[`default_config.yaml`](src/smart_accident/default_config.yaml); pass your own with `--config`.

Each event is one JSON line:

```json
{"type": "collision", "t": 6.4, "ids": [1, 2], "classes": [3, 3], "score": 0.97, "details": {"contact_since": 6.0}}
```

## Evaluating on real video

Label every video with its incident intervals (seconds), **including videos with no incident**:
without normal footage, precision is meaningless.

```json
{"videos": [
  {"name": "cam1", "duration_s": 600, "incidents": [{"start": 120.0, "end": 128.0}]},
  {"name": "cam2_normal", "duration_s": 900, "incidents": []}
]}
```

```bash
# one tracks + events file per video, named after it, then:
uv run smart-accident eval labels.json outputs/events/
```

```text
detected 1  missed 0  false alarms 0
precision 1.00  recall 1.00  F1 1.00
false alarms / hour 0.00  mean time to detect 1.2 s
```

An event counts for an incident if it falls within `[start - 2 s, end + 5 s]` (`--before`, `--after`).
Use **fixed-camera CCTV** footage; check each dataset's licence, and do not commit videos.

## Limitations

- **Not validated on real footage yet.** Thresholds are reasoned defaults tested on synthetic
  scenes, not tuned on data.
- **Fixed camera only.** Speeds assume a static view; dashcam footage breaks the rules.
- **Tracker dependence.** SORT can swap or drop IDs exactly at a collision, which can hide an
  incident. Detector misses in rain, night or heavy occlusion do the same.
- **Rear-end and head-on contact only through overlap.** Side-swipes with little box overlap, and
  incidents involving pedestrians, are not covered.
- **No learned model.** First rules and measurement; a classifier is a possible later step once
  there is labelled data to train and test it on.
- The `rtdetr`/`torchvision` extras forward to smart-surveillance's, but its PyTorch CUDA index pin
  does not carry over: torch comes from PyPI unless you add your own `[tool.uv.sources]` entry.

## Development

```bash
uv run ruff check .
uv run pytest -q
```

CI runs both without torch. Tests use scripted trajectories and a scripted detector.
