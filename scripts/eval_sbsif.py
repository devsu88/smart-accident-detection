"""Evaluate the incident rules on a sample of SB-SIF (synthetic intersection footage, CC BY 4.0).

Dataset: https://zenodo.org/records/15267252 (cite: "YoFlow Method for Scenario Based Automatic
Accident Detection", 2025). Labels are per video and live in the file names: ``NT-*`` normal traffic,
``NM-*`` near miss (no collision), anything else a crash. There is no impact timestamp, so this is a
video-level evaluation: a crash clip should raise an event, the other clips should not.

Only the sampled clips are downloaded (HTTP range requests into the remote zips), a few MB each.
Needs a detector backend, ``opencv`` and ``remotezip`` (``pip install remotezip``). Videos and tracks stay in
``--out`` (cached, so a second run only re-scores) and are never meant to be committed.

    python scripts/eval_sbsif.py --out data/sbsif --per-class 3
"""

import argparse
import json
import random
from pathlib import Path

import cv2
from remotezip import RemoteZip
from smart_surveillance.detection import build_detector

from smart_accident.config import load_config, rule_config
from smart_accident.evaluate import evaluate
from smart_accident.events import detect_incidents
from smart_accident.extract import extract_tracks
from smart_accident.io import read_jsonl, write_jsonl

URL = "https://zenodo.org/records/15267252/files/{}.zip?download=1"
ARCHIVES = [f"crash_scenario_{i}" for i in range(1, 6)] + [f"NTNM_scenario_{i}" for i in range(1, 6)]


def kind(archive, member):
    """Crash archives hold crashes; NTNM archives hold normal (nt-...) and near-miss (everything else)."""
    if archive.startswith("crash"):
        return "crash"
    return "normal" if Path(member).name.lower().startswith("nt") else "near_miss"


def stem(archive, member):
    return f"{archive}__{Path(member).stem.replace(' ', '_')}"


def sample_clips(per_class, seed, exclude=frozenset()):
    rng = random.Random(seed)
    clips = []
    for archive in ARCHIVES:
        with RemoteZip(URL.format(archive)) as zf:
            names = sorted(n for n in zf.namelist() if n.endswith(".mp4"))
        for k in ("crash", "normal", "near_miss"):
            pool = [n for n in names if kind(archive, n) == k and stem(archive, n) not in exclude]
            clips += [(archive, n) for n in rng.sample(pool, min(per_class, len(pool)))]
    return clips


def fetch(archive, member, out):
    target = out / "videos" / f"{stem(archive, member)}.mp4"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with RemoteZip(URL.format(archive)) as zf:
            target.write_bytes(zf.read(member))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/sbsif"))
    parser.add_argument("--per-class", type=int, default=3, help="clips per class and scenario")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude", type=Path, help="earlier --out folder: do not sample its clips")
    parser.add_argument("--stride", type=int, default=4, help="every Nth frame (clips are 60 fps)")
    parser.add_argument("--model", default="FasterRCNN_ResNet50_FPN")
    parser.add_argument("--config")
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"]["name"] = args.model
    rules = rule_config(config)
    detector = build_detector(config)

    labels, events_by_video, kinds = {"videos": []}, {}, {}
    exclude = {p.stem for p in (args.exclude / "videos").glob("*.mp4")} if args.exclude else frozenset()
    for archive, member in sample_clips(args.per_class, args.seed, exclude):
        path = fetch(archive, member, args.out)
        cap = cv2.VideoCapture(str(path))
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 60.0)
        cap.release()
        tracks_file = args.out / "tracks" / f"{path.stem}.jsonl"
        if tracks_file.exists():  # re-score cached tracks without running the detector again
            frames = read_jsonl(tracks_file)
        else:
            frames = list(extract_tracks(path, config, detector=detector, stride=args.stride))
            write_jsonl(tracks_file, frames)
        incidents = detect_incidents(frames, rules)
        write_jsonl(args.out / "events" / f"{path.stem}.jsonl", incidents)
        k = kind(archive, member)
        kinds[path.stem] = k
        events_by_video[path.stem] = [i["t"] for i in incidents]
        window = [{"start": 0.0, "end": duration}] if k == "crash" else []
        labels["videos"].append({"name": path.stem, "duration_s": duration, "incidents": window})
        print(f"{k:9s} {path.stem:45s} events={len(incidents)}")
    (args.out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")

    score = evaluate(labels, events_by_video, before=0.0, after=0.0)
    for k in ("crash", "normal", "near_miss"):
        names = [n for n, v in kinds.items() if v == k]
        flagged = sum(1 for n in names if events_by_video[n])
        print(f"{k:9s} clips={len(names):3d} flagged={flagged}")
    print(f"detected {score.detected}  missed {score.missed}  false alarms {score.false_alarms}")
    print(f"precision {score.precision:.2f}  recall {score.recall:.2f}  F1 {score.f1:.2f}")


if __name__ == "__main__":
    main()
