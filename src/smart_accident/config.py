"""Configuration: packaged defaults or a YAML file, split into detector settings and incident rules."""

from dataclasses import dataclass
from importlib import resources

import yaml


@dataclass
class RuleConfig:
    vehicle_classes: tuple = (3, 4, 6, 8)
    min_track_s: float = 0.5
    speed_ema: float = 0.4
    overlap_min: float = 0.10
    react_s: float = 2.0
    baseline_s: tuple = (0.5, 2.0)
    now_s: float = 0.3
    min_speed: float = 0.5
    drop_ratio: float = 0.6
    cooldown_s: float = 5.0
    stopped_s: float = 10.0
    stop_speed: float = 0.05
    jam_count: int = 3
    road_polygon: list | None = None


def load_config(path=None):
    """Return the raw config dict. With no path, the packaged defaults are used."""
    if path is None:
        text = resources.files("smart_accident").joinpath("default_config.yaml").read_text(encoding="utf-8")
    else:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    return yaml.safe_load(text)


def rule_config(config):
    """Build a RuleConfig from the ``rules`` section, rejecting unknown keys."""
    section = dict(config.get("rules") or {})
    unknown = set(section) - set(RuleConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown rule setting(s): {', '.join(sorted(unknown))}")
    for key in ("vehicle_classes", "baseline_s"):
        if key in section:
            section[key] = tuple(section[key])
    return RuleConfig(**section)
