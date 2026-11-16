"""Command line: ``tracks`` (video -> tracks), ``events`` (tracks -> incidents), ``eval``, ``clips``."""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, rule_config
from .evaluate import evaluate
from .events import detect_incidents
from .io import read_jsonl, write_jsonl


def _tracks(args):
    from .extract import extract_tracks  # needs a detector backend

    config = load_config(args.config)
    if args.model:
        config["model"]["name"] = args.model
    write_jsonl(args.output, extract_tracks(args.video, config))
    print(f"Tracks written to {args.output}")


def _events(args):
    incidents = detect_incidents(read_jsonl(args.tracks), rule_config(load_config(args.config)))
    write_jsonl(args.output, incidents)
    print(f"{len(incidents)} incident(s) written to {args.output}")


def _eval(args):
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    events = {
        video["name"]: [e["t"] for e in read_jsonl(Path(args.events_dir) / f"{video['name']}.jsonl")]
        for video in labels["videos"]
    }
    score = evaluate(labels, events, args.before, args.after)
    ttd = f"{score.mean_time_to_detect_s:.1f} s" if score.mean_time_to_detect_s is not None else "n/a"
    print(f"detected {score.detected}  missed {score.missed}  false alarms {score.false_alarms}")
    print(f"precision {score.precision:.2f}  recall {score.recall:.2f}  F1 {score.f1:.2f}")
    print(f"false alarms / hour {score.false_alarms_per_hour:.2f}  mean time to detect {ttd}")


def _clips(args):
    from .clips import export_clips

    paths = export_clips(args.video, read_jsonl(args.events), args.output)
    print(f"{len(paths)} clip(s) written to {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="smart-accident", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tracks", help="video -> tracks JSONL (needs a detector extra)")
    p.add_argument("video")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--config")
    p.add_argument("--model", help="detector name, e.g. rtdetr_r18vd or FasterRCNN_ResNet50_FPN")
    p.set_defaults(func=_tracks)

    p = sub.add_parser("events", help="tracks JSONL -> incident events JSONL")
    p.add_argument("tracks")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--config")
    p.set_defaults(func=_events)

    p = sub.add_parser("eval", help="score events against labelled incidents")
    p.add_argument("labels", help="labels JSON")
    p.add_argument("events_dir", help="folder with one <video name>.jsonl of events per labelled video")
    p.add_argument("--before", type=float, default=2.0, help="seconds an event may precede an incident")
    p.add_argument("--after", type=float, default=5.0, help="seconds an event may trail an incident")
    p.set_defaults(func=_eval)

    p = sub.add_parser("clips", help="cut a clip around each incident")
    p.add_argument("video")
    p.add_argument("events")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=_clips)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, KeyError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
