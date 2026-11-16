"""Synthetic scenes: scripted vehicle trajectories turned into per-frame track rows."""

CAR, FPS, W, H = 3, 10, 60.0, 30.0


def frames(vehicles, seconds, fps=FPS):
    """``vehicles``: {track_id: position(t) -> (x, y)} (box centre). Returns tracks frames."""
    out = []
    for n in range(int(seconds * fps)):
        t = n / fps
        rows = []
        for track_id, position in vehicles.items():
            pos = position(t)
            if pos is None:
                continue
            x, y = pos
            rows.append([x - W / 2, y - H / 2, x + W / 2, y + H / 2, track_id, CAR])
        out.append({"t": t, "tracks": rows})
    return out


def moving(x0, y, speed, stop_at=None, start_at=0.0):
    """Constant speed (px/s) along x, optionally halting dead at ``stop_at``."""
    def position(t):
        if t < start_at:
            return None
        end = t if stop_at is None else min(t, stop_at)
        return (x0 + speed * end, y)
    return position
