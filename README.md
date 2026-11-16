# smart-accident-detection

Flags possible road incidents on fixed-camera video by reasoning over **tracked vehicles**, and scores
itself against hand-labelled footage. It builds on the detector and SORT tracker of
[smart-surveillance](https://github.com/devsu88/smart-surveillance), used as a package dependency.

> **Status: evaluated on synthetic footage only** (SB-SIF, rendered in a simulator): no false alarms
> on 27 non-crash clips and 40% of crash clips detected on a held-out sample. **Not evaluated on real
> footage.** See [Results](#results) and [Limitations](#limitations).

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
| `collision` | Two vehicles make **fresh contact** (boxes overlap) and, within `react_s`, their **relative speed collapses**: it falls by at least `drop_ratio` from a value that was at least `min_speed`. Covers two moving vehicles stopping together and a moving vehicle hitting a stopped one. One event per pair per cooldown. |
| `stopped_vehicle` | A vehicle stays still inside the configured `road_polygon` for `stopped_s`, unless `jam_count` or more vehicles are stopped (a jam). Off unless a polygon is set. |

Design choices that keep false alarms down:

- **No single signal fires.** Perspective occlusion overlaps boxes all the time, and traffic slows
  down all the time. Only a fresh overlap *together with* a collapse of the relative speed counts.
- **Speeds are in box-heights per second**, so they do not depend on distance from the camera, and
  they use video timestamps, so results do not depend on processing speed. Relative speed comes
  from a line fit over the two tracks' centres, which averages out box jitter, and young tracks are
  ignored.
- **A pair that was not moving relative to each other cannot "stop"**: a jam overlap, or two
  vehicles side by side at the same speed, does not trigger.

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

## Results

Evaluated on [SB-SIF](https://zenodo.org/records/15267252) (Scenario Based - Synthetic Intersection
Footage, **CC BY 4.0**), rendered with the BeamNG.tech simulator from a fixed, elevated camera over an
intersection: 1280x720, 60 fps, about 6 s per clip. Labels are per video and come from the file
names: a crash, a near miss (vehicles almost collide) or normal traffic. There is **no impact
timestamp**, so the evaluation is at video level: a crash clip should raise an event, the others
should not.

```bash
# detector backend + opencv + remotezip needed; clips are fetched with HTTP range requests
python scripts/eval_sbsif.py --out data/sbsif --per-class 3
python scripts/eval_sbsif.py --out data/sbsif_test --exclude data/sbsif --seed 1 --per-class 3
```

The first sample was used to develop the rules, the second one was held out (different clips, never
looked at while changing the rules). Faster R-CNN detections, every 4th frame (15 fps), defaults of
`default_config.yaml`.

| Sample | Crash clips flagged | Normal clips flagged | Near-miss clips flagged |
|---|---|---|---|
| Development (15 / 15 / 15 clips) | 8 / 15 | 0 / 15 | 1 / 15 |
| **Held-out** (15 / 15 / 12 clips) | **6 / 15** | **0 / 15** | **0 / 12** |

Held-out: precision 1.00 (no false alarm in 27 non-crash clips), recall 0.40, F1 0.57.

How to read this:

- **It is a small, synthetic sample.** 15 crash clips per sample: the recall figure has a wide margin
  of error. Synthetic rendering is not real traffic, and a detector trained on real photos may behave
  differently on it.
- **Conservative by design.** Missing a crash is cheaper than raising alarms all day, and the rules
  trade recall for precision. Near misses, the hardest negatives, raised 1 alarm in 27 clips.
- **Why crashes are missed.** On the development sample, in all 7 missed crashes the two vehicles
  do touch, but one of the two tracks was born within a fraction of a second of the contact. SORT
  loses or splits tracks exactly when boxes merge, so there is no speed history for that vehicle and
  the rule cannot compare before and after. This is a tracker limit, not a threshold to tune; a
  tracker with re-identification or longer memory is the next thing to try.
- **Time to detect is not measured**, because the dataset has no impact timestamp. On one clip
  checked by eye, the event came about 0.3 s after contact.

To evaluate on your own footage, label every video with its incident intervals (seconds), including
videos with no incident (without normal footage precision is meaningless):

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

An event counts for an incident if it falls within `[start - 2 s, end + 5 s]` (`--before`, `--after`).
Use fixed-camera footage, check each dataset's licence, and do not commit videos.

Citation: *YoFlow Method for Scenario Based Automatic Accident Detection* (2025), the paper behind
SB-SIF. Videos are not redistributed here.

## Limitations

- **Not validated on real footage.** Results above are on simulator footage of one kind of scene
  (intersections), with a small sample. Thresholds were adjusted on the development sample only.
- **Fixed camera only.** Speeds assume a static view; dashcam footage breaks the rules.
- **Tracker dependence.** SORT can swap, split or drop IDs exactly at a collision, which hides the
  incident: this is the main cause of missed crashes (see [Results](#results)). Detector misses in
  rain, night or heavy occlusion do the same.
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
